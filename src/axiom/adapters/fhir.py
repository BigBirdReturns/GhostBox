"""
FHIR Healthcare Adapter for axiom-kg

Parses HL7 FHIR (Fast Healthcare Interoperability Resources) into axiom-kg coordinates.

Use cases:
- "Does this patient record from Hospital A match Hospital B?"
- "Find duplicate patient records across systems"
- "Track a condition across encounters"
- "Compare treatment protocols"

FHIR Resources: https://www.hl7.org/fhir/resourcelist.html

Usage:
    adapter = FHIRAdapter()
    nodes = adapter.parse(patient_bundle)
    
    # Find potential duplicates
    dupes = adapter.find_duplicate_patients(nodes)
"""

from typing import Any, Dict, List, Optional, Set
from pathlib import Path
from datetime import datetime
import hashlib

from .base import JSONAdapter
from axiom.core import Node, SemanticID, Space, RelationType


# FHIR resource types → axiom-kg coordinates
RESOURCE_TYPE_TO_COORDS = {
    # Individuals (Major 1 - Entity)
    "Patient": (1, 1, 1),
    "Practitioner": (1, 1, 2),
    "RelatedPerson": (1, 1, 3),
    "Person": (1, 1, 4),
    
    # Organizations (Major 1 - Entity)
    "Organization": (1, 2, 1),
    "Location": (1, 2, 2),
    "HealthcareService": (1, 2, 3),
    
    # Clinical (Major 1 - Entity, type 3)
    "Condition": (1, 3, 1),
    "Observation": (1, 3, 2),
    "DiagnosticReport": (1, 3, 3),
    "Procedure": (1, 3, 4),
    "Specimen": (1, 3, 5),
    
    # Medications (Major 1 - Entity, type 4)
    "Medication": (1, 4, 1),
    "MedicationRequest": (1, 4, 2),
    "MedicationDispense": (1, 4, 3),
    "MedicationAdministration": (1, 4, 4),
    "Immunization": (1, 4, 5),
    
    # Care Management (Major 2 - Action)
    "Encounter": (2, 1, 1),
    "EpisodeOfCare": (2, 1, 2),
    "CarePlan": (2, 1, 3),
    "CareTeam": (2, 1, 4),
    "Appointment": (2, 2, 1),
    "Task": (2, 2, 2),
    
    # Financial (Major 7 - Quantity)
    "Claim": (7, 1, 1),
    "Coverage": (7, 1, 2),
    "ExplanationOfBenefit": (7, 1, 3),
    
    # Documents (Major 8 - Abstract)
    "DocumentReference": (8, 1, 1),
    "Composition": (8, 1, 2),
    "Bundle": (8, 1, 3),
}


class FHIRAdapter(JSONAdapter):
    """
    Adapter for FHIR healthcare resources.
    
    Converts FHIR resources into axiom-kg coordinates:
    - Patients, Practitioners → Entity nodes (Major 1)
    - Encounters, Procedures → Action nodes (Major 2)
    - Conditions, Observations → Property/Entity nodes
    """
    
    DOMAIN_NAME = "fhir"
    
    def __init__(self, space: Optional[Space] = None):
        super().__init__(space)
    
    def _get_coords(self, resource_type: str) -> tuple:
        """Get coordinates for a FHIR resource type."""
        return RESOURCE_TYPE_TO_COORDS.get(resource_type, (8, 99, 1))
    
    def _extract_identifier(self, resource: Dict) -> Optional[str]:
        """Extract primary identifier from resource."""
        identifiers = resource.get("identifier", [])
        if identifiers:
            # Prefer MRN (medical record number)
            for ident in identifiers:
                if ident.get("type", {}).get("coding", [{}])[0].get("code") == "MR":
                    return ident.get("value")
            # Fall back to first identifier
            return identifiers[0].get("value")
        return resource.get("id")
    
    def _extract_name(self, resource: Dict) -> str:
        """Extract human-readable name from resource."""
        resource_type = resource.get("resourceType", "")
        
        # Patient/Practitioner names
        names = resource.get("name", [])
        if names:
            name = names[0]
            given = " ".join(name.get("given", []))
            family = name.get("family", "")
            return f"{given} {family}".strip() or resource_type
        
        # Organization/Location names
        if "name" in resource and isinstance(resource["name"], str):
            return resource["name"]
        
        # Condition/Observation codes
        code = resource.get("code", {})
        codings = code.get("coding", [])
        if codings:
            return codings[0].get("display", codings[0].get("code", resource_type))
        
        return resource.get("id", resource_type)
    
    def _parse_resource(self, resource: Dict) -> Optional[Node]:
        """Parse a single FHIR resource into a Node."""
        resource_type = resource.get("resourceType")
        if not resource_type:
            return None
        
        major, type_, subtype = self._get_coords(resource_type)
        sem_id = self.create_id(major, type_, subtype)
        
        label = self._extract_name(resource)
        identifier = self._extract_identifier(resource)
        
        # Extract common metadata
        metadata = {
            "resource_type": resource_type,
            "fhir_id": resource.get("id"),
            "identifier": identifier,
        }
        
        # Resource-specific metadata
        if resource_type == "Patient":
            metadata.update({
                "birth_date": resource.get("birthDate"),
                "gender": resource.get("gender"),
                "deceased": resource.get("deceasedBoolean", resource.get("deceasedDateTime")),
                "address": self._extract_address(resource.get("address", [])),
            })
        
        elif resource_type == "Encounter":
            metadata.update({
                "status": resource.get("status"),
                "class": resource.get("class", {}).get("code"),
                "period_start": resource.get("period", {}).get("start"),
                "period_end": resource.get("period", {}).get("end"),
                "subject": resource.get("subject", {}).get("reference"),
            })
        
        elif resource_type == "Condition":
            metadata.update({
                "clinical_status": resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
                "verification_status": resource.get("verificationStatus", {}).get("coding", [{}])[0].get("code"),
                "category": self._extract_codings(resource.get("category", [])),
                "code": self._extract_codings([resource.get("code", {})]),
                "subject": resource.get("subject", {}).get("reference"),
                "onset": resource.get("onsetDateTime"),
            })
        
        elif resource_type == "Observation":
            metadata.update({
                "status": resource.get("status"),
                "category": self._extract_codings(resource.get("category", [])),
                "code": self._extract_codings([resource.get("code", {})]),
                "value": self._extract_value(resource),
                "effective": resource.get("effectiveDateTime"),
                "subject": resource.get("subject", {}).get("reference"),
            })
        
        elif resource_type == "MedicationRequest":
            metadata.update({
                "status": resource.get("status"),
                "intent": resource.get("intent"),
                "medication": self._extract_medication(resource),
                "subject": resource.get("subject", {}).get("reference"),
                "authored_on": resource.get("authoredOn"),
            })
        
        return Node(id=sem_id, label=label, metadata=metadata)
    
    def _extract_address(self, addresses: List[Dict]) -> Optional[str]:
        """Extract address string."""
        if not addresses:
            return None
        addr = addresses[0]
        parts = []
        parts.extend(addr.get("line", []))
        parts.append(addr.get("city", ""))
        parts.append(addr.get("state", ""))
        parts.append(addr.get("postalCode", ""))
        return ", ".join(p for p in parts if p)
    
    def _extract_codings(self, codeable_concepts: List[Dict]) -> List[str]:
        """Extract display values from CodeableConcepts."""
        displays = []
        for cc in codeable_concepts:
            for coding in cc.get("coding", []):
                display = coding.get("display", coding.get("code"))
                if display:
                    displays.append(display)
        return displays
    
    def _extract_value(self, observation: Dict) -> Any:
        """Extract observation value."""
        if "valueQuantity" in observation:
            q = observation["valueQuantity"]
            return f"{q.get('value')} {q.get('unit', '')}"
        if "valueString" in observation:
            return observation["valueString"]
        if "valueCodeableConcept" in observation:
            return self._extract_codings([observation["valueCodeableConcept"]])
        return None
    
    def _extract_medication(self, med_request: Dict) -> Optional[str]:
        """Extract medication name."""
        if "medicationCodeableConcept" in med_request:
            codings = self._extract_codings([med_request["medicationCodeableConcept"]])
            return codings[0] if codings else None
        if "medicationReference" in med_request:
            return med_request["medicationReference"].get("display")
        return None
    
    def parse(self, source: Any) -> List[Node]:
        """
        Parse FHIR resource(s) into axiom-kg nodes.
        
        Accepts:
        - Single resource dict
        - Bundle dict
        - File path to JSON
        """
        data = self.load_json(source)
        nodes = []
        
        # Handle Bundle
        if data.get("resourceType") == "Bundle":
            entries = data.get("entry", [])
            for entry in entries:
                resource = entry.get("resource", {})
                node = self._parse_resource(resource)
                if node:
                    nodes.append(node)
        else:
            # Single resource
            node = self._parse_resource(data)
            if node:
                nodes.append(node)
        
        return nodes
    
    def find_duplicate_patients(self, nodes: List[Node], threshold: float = 0.8) -> List[Dict]:
        """
        Find potential duplicate patient records.
        
        Uses name similarity, birthdate, and address matching.
        """
        patients = [n for n in nodes if n.metadata.get("resource_type") == "Patient"]
        duplicates = []
        
        for i, a in enumerate(patients):
            for b in patients[i+1:]:
                score = self._match_score(a, b)
                if score >= threshold:
                    duplicates.append({
                        "patient_a": {
                            "id": a.metadata.get("fhir_id"),
                            "name": a.label,
                            "birth_date": a.metadata.get("birth_date"),
                        },
                        "patient_b": {
                            "id": b.metadata.get("fhir_id"),
                            "name": b.label,
                            "birth_date": b.metadata.get("birth_date"),
                        },
                        "match_score": score,
                        "coordinate_distance": a.id.distance(b.id),
                    })
        
        return sorted(duplicates, key=lambda x: x["match_score"], reverse=True)
    
    def _match_score(self, a: Node, b: Node) -> float:
        """Calculate match score between two patients."""
        score = 0.0
        weights = 0.0
        
        # Name similarity (simple)
        if a.label and b.label:
            weights += 0.4
            if a.label.lower() == b.label.lower():
                score += 0.4
            elif self._name_similarity(a.label, b.label) > 0.8:
                score += 0.3
        
        # Birth date match
        dob_a = a.metadata.get("birth_date")
        dob_b = b.metadata.get("birth_date")
        if dob_a and dob_b:
            weights += 0.3
            if dob_a == dob_b:
                score += 0.3
        
        # Gender match
        gender_a = a.metadata.get("gender")
        gender_b = b.metadata.get("gender")
        if gender_a and gender_b:
            weights += 0.1
            if gender_a == gender_b:
                score += 0.1
        
        # Address similarity
        addr_a = a.metadata.get("address")
        addr_b = b.metadata.get("address")
        if addr_a and addr_b:
            weights += 0.2
            if addr_a == addr_b:
                score += 0.2
        
        return score / weights if weights > 0 else 0
    
    def _name_similarity(self, name_a: str, name_b: str) -> float:
        """Simple name similarity using character overlap."""
        a = set(name_a.lower().split())
        b = set(name_b.lower().split())
        
        if not a or not b:
            return 0
        
        intersection = len(a & b)
        union = len(a | b)
        
        return intersection / union if union > 0 else 0
    
    def link_patient_records(self, nodes: List[Node]) -> Dict[str, List[Node]]:
        """
        Group all records by patient reference.
        
        Returns dict mapping patient ID → list of related records.
        """
        patient_records = {}
        
        for node in nodes:
            subject = node.metadata.get("subject", "")
            
            # Extract patient ID from reference
            if "Patient/" in subject:
                patient_id = subject.split("Patient/")[-1].split("/")[0]
                
                if patient_id not in patient_records:
                    patient_records[patient_id] = []
                patient_records[patient_id].append(node)
        
        return patient_records
    
    def patient_timeline(self, nodes: List[Node], patient_id: str) -> List[Dict]:
        """
        Build chronological timeline for a patient.
        """
        timeline = []
        
        for node in nodes:
            subject = node.metadata.get("subject", "")
            if patient_id not in subject:
                continue
            
            # Get date
            date = (
                node.metadata.get("period_start") or
                node.metadata.get("effective") or
                node.metadata.get("onset") or
                node.metadata.get("authored_on")
            )
            
            timeline.append({
                "date": date,
                "type": node.metadata.get("resource_type"),
                "description": node.label,
                "coordinate": node.id.code,
            })
        
        # Sort by date
        timeline.sort(key=lambda x: x["date"] or "")
        
        return timeline
