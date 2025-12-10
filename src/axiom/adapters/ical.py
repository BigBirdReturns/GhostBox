"""
iCalendar Adapter for axiom-kg

Parses iCalendar (.ics) files into axiom-kg coordinates.

Use cases:
- "Find scheduling conflicts across these 5 calendars"
- "What events overlap between teams?"
- "Detect double-booked resources"
- "Analyze meeting patterns over time"

Usage:
    adapter = ICalAdapter()
    nodes = adapter.parse("calendar.ics")
    
    # Find conflicts
    conflicts = adapter.find_conflicts(nodes)
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import re

from .base import FileAdapter
from axiom.core import Node, SemanticID, Space, RelationType


# Event types → type_ coordinates
EVENT_TYPE_TO_TYPE = {
    "meeting": 1,
    "call": 2,
    "appointment": 3,
    "deadline": 4,
    "reminder": 5,
    "holiday": 6,
    "vacation": 7,
    "travel": 8,
    "conference": 9,
    "workshop": 10,
    "training": 11,
    "interview": 12,
    "review": 13,
    "standup": 14,
    "sync": 14,
    "1:1": 15,
    "one-on-one": 15,
}

# Status → subtype coordinates
STATUS_TO_SUBTYPE = {
    "confirmed": 1,
    "tentative": 2,
    "cancelled": 3,
}


class ICalAdapter(FileAdapter):
    """
    Adapter for iCalendar files.
    
    Converts calendar events into axiom-kg coordinates:
    - Events → Time nodes (Major 6)
    - Attendees → Entity nodes (Major 1)
    - Locations → Location nodes (Major 5)
    """
    
    DOMAIN_NAME = "ical"
    SUPPORTED_EXTENSIONS = [".ics", ".ical"]
    
    def __init__(self, space: Optional[Space] = None):
        super().__init__(space)
    
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse iCal datetime formats."""
        if not dt_str:
            return None
        
        # Remove TZID prefix if present
        dt_str = re.sub(r'^TZID=[^:]+:', '', dt_str)
        
        formats = [
            "%Y%m%dT%H%M%SZ",     # UTC
            "%Y%m%dT%H%M%S",       # Local
            "%Y%m%d",              # Date only
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _classify_event(self, summary: str, description: str = "") -> int:
        """Classify event type from summary/description."""
        text = (summary + " " + description).lower()
        
        for keyword, type_ in EVENT_TYPE_TO_TYPE.items():
            if keyword in text:
                return type_
        
        return 99  # Unknown
    
    def _parse_ical_content(self, content: str) -> List[Dict]:
        """Parse raw iCal content into event dictionaries."""
        events = []
        current_event = None
        current_key = None
        
        lines = content.replace('\r\n ', '').replace('\r\n\t', '').split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line == "BEGIN:VEVENT":
                current_event = {}
            elif line == "END:VEVENT":
                if current_event:
                    events.append(current_event)
                current_event = None
            elif current_event is not None:
                if ':' in line:
                    # Handle properties with parameters
                    if ';' in line.split(':')[0]:
                        key_part = line.split(':')[0]
                        key = key_part.split(';')[0]
                        value = ':'.join(line.split(':')[1:])
                    else:
                        key, value = line.split(':', 1)
                    
                    current_event[key] = value
        
        return events
    
    def parse(self, source: Any) -> List[Node]:
        """
        Parse iCalendar file into axiom-kg nodes.
        
        Args:
            source: .ics file path or content string
            
        Returns:
            List of Node objects
        """
        if isinstance(source, Path):
            content = source.read_text()
        elif isinstance(source, str):
            path = Path(source)
            if path.exists():
                content = path.read_text()
            else:
                content = source
        else:
            raise ValueError(f"Cannot parse {type(source)}")
        
        ical_events = self._parse_ical_content(content)
        nodes = []
        
        for event in ical_events:
            node = self._event_to_node(event)
            if node:
                nodes.append(node)
        
        return nodes
    
    def _event_to_node(self, event: Dict) -> Optional[Node]:
        """Convert iCal event dict to axiom-kg Node."""
        summary = event.get("SUMMARY", "")
        if not summary:
            return None
        
        # Events are Time (Major 6)
        major = 6
        type_ = self._classify_event(summary, event.get("DESCRIPTION", ""))
        
        status = event.get("STATUS", "CONFIRMED").lower()
        subtype = STATUS_TO_SUBTYPE.get(status, 1)
        
        sem_id = self.create_id(major, type_, subtype)
        
        # Parse times
        dtstart = self._parse_datetime(event.get("DTSTART", ""))
        dtend = self._parse_datetime(event.get("DTEND", ""))
        
        # Parse attendees
        attendees = []
        for key, value in event.items():
            if key == "ATTENDEE":
                # Extract email from mailto:
                email = re.sub(r'^mailto:', '', value, flags=re.IGNORECASE)
                attendees.append(email)
        
        metadata = {
            "summary": summary,
            "description": event.get("DESCRIPTION", "")[:500],
            "location": event.get("LOCATION"),
            "dtstart": dtstart.isoformat() if dtstart else None,
            "dtend": dtend.isoformat() if dtend else None,
            "duration_minutes": int((dtend - dtstart).total_seconds() / 60) if dtstart and dtend else None,
            "status": event.get("STATUS"),
            "attendees": attendees,
            "organizer": event.get("ORGANIZER"),
            "uid": event.get("UID"),
            "recurrence": event.get("RRULE"),
        }
        
        return Node(id=sem_id, label=summary, metadata=metadata)
    
    def find_conflicts(self, nodes: List[Node]) -> List[Dict]:
        """
        Find scheduling conflicts between events.
        
        Returns list of overlapping event pairs.
        """
        conflicts = []
        
        # Filter to events with valid times
        timed_events = []
        for node in nodes:
            dtstart = node.metadata.get("dtstart")
            dtend = node.metadata.get("dtend")
            if dtstart and dtend:
                timed_events.append({
                    "node": node,
                    "start": datetime.fromisoformat(dtstart),
                    "end": datetime.fromisoformat(dtend),
                })
        
        # Check all pairs
        for i, a in enumerate(timed_events):
            for b in timed_events[i+1:]:
                # Check overlap
                if a["start"] < b["end"] and b["start"] < a["end"]:
                    conflicts.append({
                        "event_a": a["node"].label,
                        "event_b": b["node"].label,
                        "overlap_start": max(a["start"], b["start"]).isoformat(),
                        "overlap_end": min(a["end"], b["end"]).isoformat(),
                        "coordinate_distance": a["node"].id.distance(b["node"].id),
                    })
        
        return conflicts
    
    def find_attendee_conflicts(self, nodes: List[Node], attendee: str) -> List[Dict]:
        """Find conflicts for a specific attendee."""
        # Filter to events with this attendee
        attendee_events = [
            n for n in nodes 
            if attendee.lower() in [a.lower() for a in n.metadata.get("attendees", [])]
        ]
        
        return self.find_conflicts(attendee_events)
    
    def meeting_patterns(self, nodes: List[Node]) -> Dict[str, Any]:
        """
        Analyze meeting patterns.
        
        Returns:
            Statistics about meeting frequency, duration, attendees
        """
        total_events = len(nodes)
        total_duration = 0
        by_type = {}
        by_day = {i: 0 for i in range(7)}  # Mon=0, Sun=6
        attendee_counts = []
        
        for node in nodes:
            # Duration
            duration = node.metadata.get("duration_minutes", 0)
            if duration:
                total_duration += duration
            
            # By type
            type_coord = node.id.type_
            by_type[type_coord] = by_type.get(type_coord, 0) + 1
            
            # By day of week
            dtstart = node.metadata.get("dtstart")
            if dtstart:
                dt = datetime.fromisoformat(dtstart)
                by_day[dt.weekday()] += 1
            
            # Attendee count
            attendees = node.metadata.get("attendees", [])
            if attendees:
                attendee_counts.append(len(attendees))
        
        return {
            "total_events": total_events,
            "total_duration_hours": total_duration / 60,
            "avg_duration_minutes": total_duration / total_events if total_events else 0,
            "by_day": {
                "monday": by_day[0],
                "tuesday": by_day[1],
                "wednesday": by_day[2],
                "thursday": by_day[3],
                "friday": by_day[4],
                "saturday": by_day[5],
                "sunday": by_day[6],
            },
            "avg_attendees": sum(attendee_counts) / len(attendee_counts) if attendee_counts else 0,
            "event_type_distribution": by_type,
        }
    
    def merge_calendars(self, *calendars: Any) -> Tuple[List[Node], List[Dict]]:
        """
        Merge multiple calendars and find all conflicts.
        
        Returns:
            Tuple of (all_nodes, conflicts)
        """
        all_nodes = []
        for cal in calendars:
            nodes = self.parse(cal)
            all_nodes.extend(nodes)
        
        conflicts = self.find_conflicts(all_nodes)
        
        return all_nodes, conflicts
