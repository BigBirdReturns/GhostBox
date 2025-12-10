"""
GhostBox Photonic Source - Screen Ghost Integration

This module bridges Screen Ghost (photonic) to GhostBox (semantic).

Screen Ghost sees pixels and extracts state.
GhostBox assigns coordinates and detects tension.

Usage:
    from ghostbox.sources import ScreenGhostSource
    
    # Single observation
    source = ScreenGhostSource()
    state = source.capture()
    node = source.to_node(state)
    
    # Continuous monitoring
    for state in source.watch(interval=5):
        node = source.to_node(state)
        engine.space.add(node)
        tension = engine.compute_tension(state.app)
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# Axiom-KG core (for Node and SemanticID)
from axiom.core import Node, SemanticID, Space, RelationType


# =============================================================================
# SCREEN STATE (matches Screen Ghost's ScreenState)
# =============================================================================

@dataclass
class UIElement:
    """A UI element extracted from screen."""
    type: str
    label: str
    value: Optional[str] = None
    bounds: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"type": self.type, "label": self.label}
        if self.value is not None:
            d["value"] = self.value
        return d


@dataclass
class ScreenState:
    """
    State captured from Screen Ghost.
    
    This mirrors screenghost.ScreenState but is defined here
    to avoid import dependency (Screen Ghost may be separate process).
    """
    app: str
    screen: str
    elements: List[UIElement] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.8
    device: Optional[str] = None
    screenshot_path: Optional[str] = None
    width: int = 0
    height: int = 0
    
    @classmethod
    def from_event(cls, event: Dict[str, Any]) -> "ScreenState":
        """Parse from Screen Ghost JSON event."""
        data = event.get("data", {})
        elements = [
            UIElement(
                type=el.get("type", "unknown"),
                label=el.get("label", ""),
                value=el.get("value"),
            )
            for el in data.get("elements", [])
        ]
        
        dims = data.get("dimensions", {})
        
        return cls(
            app=data.get("app", "unknown"),
            screen=data.get("screen", "unknown"),
            elements=elements,
            timestamp=datetime.fromisoformat(event.get("timestamp", datetime.utcnow().isoformat())),
            confidence=event.get("confidence", 0.8),
            device=event.get("device"),
            screenshot_path=event.get("screenshot"),
            width=dims.get("width", 0),
            height=dims.get("height", 0),
        )
    
    def to_event(self) -> Dict[str, Any]:
        """Convert to GhostBox event format."""
        return {
            "topic": f"{self.app}.{self.screen}",
            "source": "screen_ghost",
            "source_type": "photonic",
            "data": {
                "app": self.app,
                "screen": self.screen,
                "elements": [el.to_dict() for el in self.elements],
            },
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def get_value(self, label: str) -> Optional[str]:
        """Get element value by label."""
        for el in self.elements:
            if el.label.lower() == label.lower():
                return el.value
        return None
    
    def diff(self, other: "ScreenState") -> Dict[str, Any]:
        """Compare two states."""
        changes = {
            "app_changed": self.app != other.app,
            "screen_changed": self.screen != other.screen,
            "element_changes": [],
            "has_changes": False,
        }
        
        self_els = {el.label.lower(): el for el in self.elements}
        other_els = {el.label.lower(): el for el in other.elements}
        
        for label, el in self_els.items():
            if label not in other_els:
                changes["element_changes"].append({
                    "type": "removed", "label": el.label, "old_value": el.value
                })
            elif el.value != other_els[label].value:
                changes["element_changes"].append({
                    "type": "changed", "label": el.label,
                    "old_value": el.value, "new_value": other_els[label].value
                })
        
        for label, el in other_els.items():
            if label not in self_els:
                changes["element_changes"].append({
                    "type": "added", "label": el.label, "new_value": el.value
                })
        
        changes["has_changes"] = (
            changes["app_changed"] or
            changes["screen_changed"] or
            len(changes["element_changes"]) > 0
        )
        
        return changes


# =============================================================================
# COORDINATE MAPPING
# =============================================================================

# App categories → Major coordinate
APP_CATEGORY_COORDS = {
    # System apps
    "settings": (1, 9, 1),
    "phone": (1, 9, 2),
    "messages": (1, 9, 3),
    "contacts": (1, 9, 4),
    "camera": (1, 9, 5),
    "photos": (1, 9, 6),
    "files": (1, 9, 7),
    
    # Smart home
    "google home": (1, 8, 1),
    "smartthings": (1, 8, 2),
    "philips hue": (1, 8, 3),
    "nest": (1, 8, 4),
    "ring": (1, 8, 5),
    "ecobee": (1, 8, 6),
    "alexa": (1, 8, 7),
    
    # Productivity
    "chrome": (1, 7, 1),
    "gmail": (1, 7, 2),
    "calendar": (1, 7, 3),
    "drive": (1, 7, 4),
    "docs": (1, 7, 5),
    "sheets": (1, 7, 6),
    "slack": (1, 7, 7),
    
    # Social
    "twitter": (1, 6, 1),
    "facebook": (1, 6, 2),
    "instagram": (1, 6, 3),
    "linkedin": (1, 6, 4),
    "whatsapp": (1, 6, 5),
    "telegram": (1, 6, 6),
    
    # Finance
    "bank": (1, 5, 1),
    "venmo": (1, 5, 2),
    "paypal": (1, 5, 3),
    "robinhood": (1, 5, 4),
    
    # Default
    "home": (1, 1, 1),
    "launcher": (1, 1, 1),
    "unknown": (1, 99, 1),
}


def get_app_coords(app_name: str) -> Tuple[int, int, int]:
    """Get coordinates for an app."""
    app_lower = app_name.lower()
    
    for key, coords in APP_CATEGORY_COORDS.items():
        if key in app_lower:
            return coords
    
    return APP_CATEGORY_COORDS["unknown"]


# =============================================================================
# SCREEN GHOST SOURCE
# =============================================================================

class ScreenGhostSource:
    """
    Photonic source that wraps Screen Ghost.
    
    Can operate in two modes:
    1. Direct: Import and call Screen Ghost functions directly
    2. Subprocess: Run screenghost.py --observe and parse output
    
    Subprocess mode is default for isolation.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        screenghost_path: Path = Path("screenghost.py"),
        model_path: Optional[Path] = None,
        use_subprocess: bool = True,
    ):
        self.device = device
        self.screenghost_path = screenghost_path
        self.model_path = model_path
        self.use_subprocess = use_subprocess
        
        # For direct mode
        self._sg_module = None
    
    def _run_subprocess(self, args: List[str]) -> str:
        """Run screenghost.py with args and return stdout."""
        cmd = [sys.executable, str(self.screenghost_path)]
        
        if self.device:
            cmd.extend(["--device", self.device])
        
        if self.model_path:
            cmd.extend(["--model-path", str(self.model_path)])
        
        cmd.extend(args)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            raise RuntimeError(f"screenghost failed: {result.stderr}")
        
        return result.stdout
    
    def _load_direct(self):
        """Load Screen Ghost module for direct calls."""
        if self._sg_module is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("screenghost", self.screenghost_path)
            self._sg_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._sg_module)
        return self._sg_module
    
    def capture(self) -> ScreenState:
        """
        Capture current screen state.
        
        Returns ScreenState ready for conversion to Node.
        """
        if self.use_subprocess:
            output = self._run_subprocess(["--observe", "--format", "json"])
            event = json.loads(output)
            return ScreenState.from_event(event)
        else:
            sg = self._load_direct()
            state = sg.observe(device=self.device)
            return ScreenState.from_event(state.to_event())
    
    def watch(
        self,
        interval: float = 2.0,
        app_filter: Optional[str] = None,
    ) -> Generator[ScreenState, None, None]:
        """
        Continuously observe screen.
        
        Yields ScreenState on each change.
        """
        if self.use_subprocess:
            # Run continuous mode and parse streaming output
            cmd = [
                sys.executable, str(self.screenghost_path),
                "--observe", "--continuous",
                "--interval", str(interval),
                "--format", "json",
            ]
            
            if self.device:
                cmd.extend(["--device", self.device])
            
            if app_filter:
                cmd = [
                    sys.executable, str(self.screenghost_path),
                    "--watch", app_filter,
                    "--interval", str(interval),
                    "--format", "json",
                ]
                if self.device:
                    cmd.extend(["--device", self.device])
            
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
            
            try:
                for line in proc.stdout:
                    line = line.strip()
                    if line:
                        event = json.loads(line)
                        yield ScreenState.from_event(event)
            finally:
                proc.terminate()
        
        else:
            sg = self._load_direct()
            for state in sg.watch(
                device=self.device,
                interval=interval,
                app_filter=app_filter,
            ):
                yield ScreenState.from_event(state.to_event())
    
    def to_node(self, state: ScreenState, space: Optional[Space] = None) -> Node:
        """
        Convert ScreenState to Axiom-KG Node.
        
        This is the bridge from photonic (pixels) to semantic (coordinates).
        """
        # Get base coordinates from app
        major, type_, subtype = get_app_coords(state.app)
        
        # Instance from screen name hash
        instance = hash(state.screen) % 9999 + 1
        
        sem_id = SemanticID.create(
            major=major,
            type_=type_,
            subtype=subtype,
            instance=instance,
        )
        
        # Build metadata
        metadata = {
            "app": state.app,
            "screen": state.screen,
            "source_type": "photonic",
            "confidence": state.confidence,
            "timestamp": state.timestamp.isoformat(),
            "element_count": len(state.elements),
        }
        
        # Add element values as properties
        for el in state.elements:
            if el.value is not None:
                key = f"ui_{el.label.lower().replace(' ', '_')}"
                metadata[key] = el.value
        
        node = Node(
            id=sem_id,
            label=f"{state.app}.{state.screen}",
            metadata=metadata,
        )
        
        if space:
            space.add(node)
        
        return node
    
    def to_nodes(self, state: ScreenState, space: Optional[Space] = None) -> List[Node]:
        """
        Convert ScreenState to multiple Nodes (one per element).
        
        Use this for finer-grained tension detection.
        """
        nodes = []
        
        # Main screen node
        screen_node = self.to_node(state, space)
        nodes.append(screen_node)
        
        # Element nodes
        base_major, base_type, base_subtype = get_app_coords(state.app)
        
        for i, el in enumerate(state.elements):
            el_id = SemanticID.create(
                major=base_major,
                type_=base_type,
                subtype=base_subtype + 1,  # Elements get subtype + 1
                instance=i + 1,
            )
            
            el_node = Node(
                id=el_id,
                label=f"{state.app}.{state.screen}.{el.label}",
                metadata={
                    "element_type": el.type,
                    "label": el.label,
                    "value": el.value,
                    "source_type": "photonic",
                    "parent_screen": screen_node.id.code,
                },
            )
            
            # Add relation to parent screen
            el_node.add_relation(RelationType.IS_PART_OF, screen_node)
            
            if space:
                space.add(el_node)
            
            nodes.append(el_node)
        
        return nodes


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def observe(device: Optional[str] = None) -> ScreenState:
    """Quick observe without creating source object."""
    source = ScreenGhostSource(device=device)
    return source.capture()


def screen_to_node(state: ScreenState) -> Node:
    """Quick convert without creating source object."""
    source = ScreenGhostSource()
    return source.to_node(state)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate photonic source (requires Android device)."""
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  GhostBox Photonic Source - Screen Ghost Integration             ║")
    print("║                                                                   ║")
    print("║  The UI is the API                                               ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Check if Screen Ghost is available
    screenghost_path = Path("screenghost.py")
    if not screenghost_path.exists():
        screenghost_path = Path("../screenghost/screenghost.py")
    
    if not screenghost_path.exists():
        print("Screen Ghost not found. Demo using mock data.")
        print()
        
        # Mock state
        state = ScreenState(
            app="Settings",
            screen="Display",
            elements=[
                UIElement(type="toggle", label="Dark Mode", value="off"),
                UIElement(type="slider", label="Brightness", value="70%"),
                UIElement(type="toggle", label="Auto-brightness", value="on"),
            ],
        )
    else:
        print(f"Found Screen Ghost at {screenghost_path}")
        print("Capturing screen state...")
        print()
        
        source = ScreenGhostSource(screenghost_path=screenghost_path)
        state = source.capture()
    
    print(f"App: {state.app}")
    print(f"Screen: {state.screen}")
    print(f"Elements: {len(state.elements)}")
    print()
    
    for el in state.elements:
        if el.value:
            print(f"  [{el.type}] {el.label} = {el.value}")
        else:
            print(f"  [{el.type}] {el.label}")
    print()
    
    # Convert to node
    print("Converting to Axiom-KG Node...")
    node = screen_to_node(state)
    print(f"  ID: {node.id.code}")
    print(f"  Label: {node.label}")
    print(f"  Coordinates: major={node.id.major}, type={node.id.type_}, subtype={node.id.subtype}")
    print()
    
    # Show event format
    print("GhostBox Event:")
    print(json.dumps(state.to_event(), indent=2))


if __name__ == "__main__":
    demo()
