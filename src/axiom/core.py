"""
Axiom: Semantic Coordinates for Derivable Knowledge

The core insight: knowledge should not be stored. It should be derivable.
Store the minimal structure from which all answers can be constructed.

Like mathematics: you don't store every equation. You store axioms and 
derivation rules. Everything else is generated on demand.

Like DNA: you don't store the organism. You store the instructions for 
building it. The organism is rehydrated from the code.

This module provides the primitives:
- SemanticID: coordinates in knowledge space
- Node: positioned concepts with minimal stored relations  
- Fork: explicit ambiguity branching
- Space: the geometry in which derivation happens
- AuditLog: cryptographic proof of every operation

No embeddings. No retrieval. No guessing.
The system derives. It knows how to know.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, Tuple, Callable
import hashlib
import json
import time
import re


# =============================================================================
# SEMANTIC ID: Coordinates in Knowledge Space
# =============================================================================

# The 8 axioms - everything else derives from these categories
MAJOR_CATEGORIES = {
    1: "Entity",      # things that exist
    2: "Action",      # things that happen  
    3: "Property",    # qualities of things
    4: "Relation",    # connections between things
    5: "Location",    # where things are
    6: "Time",        # when things happen
    7: "Quantity",    # how much/many
    8: "Abstract",    # ideas, concepts, patterns
}


class SemanticIDError(ValueError):
    """Raised when a SemanticID is malformed."""
    pass


@dataclass(frozen=True, slots=True)
class SemanticID:
    """
    A coordinate in semantic space.
    
    Format: MM-TT-SS-XXXX
    - MM: Major category (01-08, the axioms)
    - TT: Type within category (01-99)
    - SS: Subtype within type (01-99)
    - XXXX: Instance (0001-9999)
    
    This is not a database key. It's a position from which
    relationships can be DERIVED, not looked up.
    
    Two concepts with the same MM share ontological category.
    Two concepts with the same MM-TT are siblings.
    Two concepts with the same MM-TT-SS are close cousins.
    
    Distance is computed from coordinates, not stored.
    """
    
    major: int      # 1-8
    type_: int      # 1-99
    subtype: int    # 1-99
    instance: int   # 1-9999
    
    _PATTERN = re.compile(r'^(\d{2})-(\d{2})-(\d{2})-(\d{4})$')
    
    def __post_init__(self):
        if not (1 <= self.major <= 8):
            raise SemanticIDError(f"Major must be 1-8, got {self.major}")
        if not (1 <= self.type_ <= 99):
            raise SemanticIDError(f"Type must be 1-99, got {self.type_}")
        if not (1 <= self.subtype <= 99):
            raise SemanticIDError(f"Subtype must be 1-99, got {self.subtype}")
        if not (1 <= self.instance <= 9999):
            raise SemanticIDError(f"Instance must be 1-9999, got {self.instance}")
    
    @property
    def code(self) -> str:
        """The canonical string representation."""
        return f"{self.major:02d}-{self.type_:02d}-{self.subtype:02d}-{self.instance:04d}"
    
    @property
    def category_name(self) -> str:
        """Human-readable category name."""
        return MAJOR_CATEGORIES.get(self.major, "Unknown")
    
    @classmethod
    def parse(cls, code: str) -> "SemanticID":
        """Parse a string like '01-02-03-0004' into a SemanticID."""
        match = cls._PATTERN.match(code)
        if not match:
            raise SemanticIDError(f"Invalid format: {code}. Expected MM-TT-SS-XXXX")
        major, type_, subtype, instance = map(int, match.groups())
        return cls(major, type_, subtype, instance)
    
    @classmethod
    def create(cls, major: int, type_: int, subtype: int, instance: int) -> "SemanticID":
        """Create a new SemanticID from components."""
        return cls(major, type_, subtype, instance)
    
    # -------------------------------------------------------------------------
    # DERIVATION: These relationships are COMPUTED, not stored
    # -------------------------------------------------------------------------
    
    def shares_category(self, other: "SemanticID") -> bool:
        """Same major category = same ontological kind."""
        return self.major == other.major
    
    def shares_type(self, other: "SemanticID") -> bool:
        """Same major+type = siblings in the taxonomy."""
        return self.major == other.major and self.type_ == other.type_
    
    def shares_subtype(self, other: "SemanticID") -> bool:
        """Same major+type+subtype = close cousins."""
        return (self.major == other.major and 
                self.type_ == other.type_ and 
                self.subtype == other.subtype)
    
    def distance(self, other: "SemanticID") -> int:
        """
        Semantic distance derived from coordinates alone.
        No embedding lookup. No vector math. Pure geometry.
        
        0 = identical
        1 = same subtype (close cousins)
        2 = same type (siblings)
        3 = same category (distant relatives)
        4 = different categories
        """
        if self.code == other.code:
            return 0
        if self.shares_subtype(other):
            return 1
        if self.shares_type(other):
            return 2
        if self.shares_category(other):
            return 3
        return 4
    
    def __str__(self) -> str:
        return self.code
    
    def __repr__(self) -> str:
        return f"SemanticID({self.code})"
    
    def __hash__(self) -> int:
        return hash(self.code)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, SemanticID):
            return self.code == other.code
        return False


# =============================================================================
# RELATION TYPES: The edges in the graph (start small, grow later)
# =============================================================================

class RelationType(Enum):
    """
    Seed set of relation types.
    Start with 8, grow to 128 as needed.
    """
    IS_A = auto()           # taxonomic
    PART_OF = auto()        # mereological
    HAS_PROPERTY = auto()   # attributive
    CAUSES = auto()         # causal
    LOCATED_IN = auto()     # spatial
    OCCURS_AT = auto()      # temporal
    SIMILAR_TO = auto()     # analogical
    CONTRADICTS = auto()    # oppositional
    FORKED_FROM = auto()    # ambiguity lineage
    DERIVED_FROM = auto()   # derivation lineage


# =============================================================================
# NODE: A positioned concept in semantic space
# =============================================================================

@dataclass
class Node:
    """
    A concept positioned in semantic space.
    
    Stores only:
    - id: the SemanticID coordinate
    - label: human-readable name
    - metadata: optional tags/notes
    - relations: explicit edges (sparse - most relationships are derived)
    
    Everything else is computed from position and graph structure.
    """
    
    id: SemanticID
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[RelationType, List[str]] = field(default_factory=dict)  # type -> [target codes]
    
    def add_relation(self, rel_type: RelationType, target: "Node") -> None:
        """Add an explicit relation to another node."""
        if rel_type not in self.relations:
            self.relations[rel_type] = []
        target_code = target.id.code
        if target_code not in self.relations[rel_type]:
            self.relations[rel_type].append(target_code)
    
    def get_relations(self, rel_type: RelationType) -> List[str]:
        """Get all target codes for a relation type."""
        return self.relations.get(rel_type, [])
    
    def relation_count(self) -> int:
        """Total number of explicit relations."""
        return sum(len(targets) for targets in self.relations.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/transmission."""
        return {
            "id": self.id.code,
            "label": self.label,
            "metadata": self.metadata,
            "relations": {
                rel_type.name: targets 
                for rel_type, targets in self.relations.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Deserialize from storage/transmission."""
        relations = {
            RelationType[name]: targets
            for name, targets in data.get("relations", {}).items()
        }
        return cls(
            id=SemanticID.parse(data["id"]),
            label=data["label"],
            metadata=data.get("metadata", {}),
            relations=relations
        )
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        return False


# =============================================================================
# FORK: Explicit ambiguity handling
# =============================================================================

@dataclass
class Fork:
    """
    Explicit branching when a concept has multiple incompatible meanings.
    
    Example: "jaguar" forks into jaguar:animal and jaguar:car
    
    Forks attach to nodes, not IDs. The ID stays stable.
    The fork tracks the divergent meanings and their resolution.
    """
    
    source_id: SemanticID           # The ambiguous concept
    branches: List[SemanticID]      # The divergent meanings
    created_at: float = field(default_factory=time.time)
    resolved_to: Optional[SemanticID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_resolved(self) -> bool:
        return self.resolved_to is not None
    
    @property
    def branch_count(self) -> int:
        return len(self.branches)
    
    def resolve(self, chosen: SemanticID) -> None:
        """Mark the fork as resolved to a specific branch."""
        if chosen not in self.branches:
            raise ValueError(f"{chosen} is not a branch of this fork")
        self.resolved_to = chosen
    
    def add_branch(self, branch_id: SemanticID) -> None:
        """Add a new branch to the fork."""
        if branch_id not in self.branches:
            self.branches.append(branch_id)


# =============================================================================
# AUDIT LOG: Cryptographic proof of every operation
# =============================================================================

@dataclass
class AuditEntry:
    """A single entry in the audit log."""
    index: int
    timestamp: float
    action: str
    args: Tuple[Any, ...]
    prev_hash: str
    hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "action": self.action,
            "args": self.args,
            "prev_hash": self.prev_hash,
            "hash": self.hash
        }


class AuditLog:
    """
    Append-only cryptographic audit chain.
    
    Every operation is logged with:
    - What happened
    - When it happened
    - Hash of the previous entry (chain integrity)
    - Hash of this entry (tamper evidence)
    
    You can replay the log to reconstruct any historical state.
    You can verify the chain to prove no entries were modified.
    """
    
    GENESIS_HASH = "0" * 64
    
    def __init__(self):
        self._entries: List[AuditEntry] = []
    
    def _compute_hash(self, payload: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a payload."""
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    
    def append(self, action: str, *args: Any) -> AuditEntry:
        """Log an action with its arguments."""
        index = len(self._entries)
        timestamp = time.time()
        prev_hash = self._entries[-1].hash if self._entries else self.GENESIS_HASH
        
        payload = {
            "index": index,
            "timestamp": timestamp,
            "action": action,
            "args": args,
            "prev_hash": prev_hash
        }
        entry_hash = self._compute_hash(payload)
        
        entry = AuditEntry(
            index=index,
            timestamp=timestamp,
            action=action,
            args=args,
            prev_hash=prev_hash,
            hash=entry_hash
        )
        self._entries.append(entry)
        return entry
    
    def verify(self) -> bool:
        """Verify the entire chain is intact."""
        for i, entry in enumerate(self._entries):
            # Check prev_hash linkage
            expected_prev = self._entries[i - 1].hash if i > 0 else self.GENESIS_HASH
            if entry.prev_hash != expected_prev:
                return False
            
            # Recompute hash
            payload = {
                "index": entry.index,
                "timestamp": entry.timestamp,
                "action": entry.action,
                "args": entry.args,
                "prev_hash": entry.prev_hash
            }
            if self._compute_hash(payload) != entry.hash:
                return False
        
        return True
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __iter__(self):
        return iter(self._entries)
    
    def last(self, n: int = 5) -> List[AuditEntry]:
        """Get the last n entries."""
        return self._entries[-n:]


# =============================================================================
# SPACE: The semantic geometry where derivation happens
# =============================================================================

class Space:
    """
    The semantic space in which concepts exist and from which knowledge is derived.
    
    This is the SEED. Everything else grows from here.
    
    Key principle: store minimal structure, derive everything else.
    - Siblings are derived from shared coordinates
    - Categories are derived from major component
    - Paths are derived by traversing relations
    - Tension is derived from fork count and relation density
    
    What RAG gets wrong: it stores chunks and retrieves them.
    What embeddings get wrong: they encode similarity, not structure.
    What transformers get wrong: they bake knowledge into weights.
    
    This DERIVES. It doesn't retrieve. It doesn't guess.
    """
    
    def __init__(self):
        self._nodes: Dict[str, Node] = {}  # code -> Node
        self._forks: Dict[str, Fork] = {}  # source_code -> Fork
        self._audit = AuditLog()
    
    @property
    def audit(self) -> AuditLog:
        return self._audit
    
    # -------------------------------------------------------------------------
    # STORAGE: The minimal structure
    # -------------------------------------------------------------------------
    
    def add(self, node: Node) -> Node:
        """Add a node to the space."""
        code = node.id.code
        if code in self._nodes:
            raise ValueError(f"Node {code} already exists")
        self._nodes[code] = node
        self._audit.append("ADD", code, node.label)
        return node
    
    def get(self, id_or_code: SemanticID | str) -> Optional[Node]:
        """Get a node by ID or code string."""
        code = id_or_code.code if isinstance(id_or_code, SemanticID) else id_or_code
        return self._nodes.get(code)
    
    def find_by_label(self, label: str, case_sensitive: bool = False) -> List[Node]:
        """Find nodes by label."""
        if case_sensitive:
            return [n for n in self._nodes.values() if n.label == label]
        label_lower = label.lower()
        return [n for n in self._nodes.values() if n.label.lower() == label_lower]
    
    def add_relation(self, source: Node, rel_type: RelationType, target: Node) -> None:
        """Add a relation between two nodes."""
        if source.id.code not in self._nodes:
            raise ValueError(f"Source node {source.id} not in space")
        if target.id.code not in self._nodes:
            raise ValueError(f"Target node {target.id} not in space")
        source.add_relation(rel_type, target)
        self._audit.append("RELATE", source.id.code, rel_type.name, target.id.code)
    
    def create_fork(self, source: Node, branch_labels: List[str]) -> Tuple[Fork, List[Node]]:
        """
        Create a fork for an ambiguous concept.
        
        Returns the Fork and the new branch Nodes.
        Branch nodes get new instance numbers in the same subtype.
        """
        if source.id.code not in self._nodes:
            raise ValueError(f"Source node {source.id} not in space")
        
        branches: List[Node] = []
        branch_ids: List[SemanticID] = []
        
        # Find the next available instance numbers
        existing_instances = [
            n.id.instance for n in self._nodes.values()
            if n.id.shares_subtype(source.id)
        ]
        next_instance = max(existing_instances, default=0) + 1
        
        for label in branch_labels:
            branch_id = SemanticID.create(
                source.id.major,
                source.id.type_,
                source.id.subtype,
                next_instance
            )
            branch_node = Node(
                id=branch_id,
                label=f"{source.label}:{label}",
                metadata={"forked_from": source.id.code}
            )
            branch_node.add_relation(RelationType.FORKED_FROM, source)
            
            self._nodes[branch_id.code] = branch_node
            branches.append(branch_node)
            branch_ids.append(branch_id)
            next_instance += 1
        
        fork = Fork(source_id=source.id, branches=branch_ids)
        self._forks[source.id.code] = fork
        
        self._audit.append("FORK", source.id.code, [b.code for b in branch_ids])
        
        return fork, branches
    
    def get_fork(self, source: Node) -> Optional[Fork]:
        """Get the fork for a node, if any."""
        return self._forks.get(source.id.code)
    
    # -------------------------------------------------------------------------
    # DERIVATION: The physics of semantic space
    # -------------------------------------------------------------------------
    
    def derive_siblings(self, node: Node) -> List[Node]:
        """
        DERIVED: all nodes sharing type with this one.
        Not stored - computed from coordinates.
        """
        self._audit.append("DERIVE", "siblings", node.id.code)
        return [
            n for n in self._nodes.values()
            if n.id.shares_type(node.id) and n.id.code != node.id.code
        ]
    
    def derive_cousins(self, node: Node) -> List[Node]:
        """
        DERIVED: all nodes sharing subtype with this one.
        """
        self._audit.append("DERIVE", "cousins", node.id.code)
        return [
            n for n in self._nodes.values()
            if n.id.shares_subtype(node.id) and n.id.code != node.id.code
        ]
    
    def derive_category(self, major: int) -> List[Node]:
        """
        DERIVED: all nodes in a major category.
        """
        self._audit.append("DERIVE", "category", major)
        return [n for n in self._nodes.values() if n.id.major == major]
    
    def derive_path(
        self, 
        start: Node, 
        end: Node, 
        visited: Optional[Set[str]] = None
    ) -> Optional[List[Node | str]]:
        """
        DERIVED: find a path between two nodes.
        This is REASONING - following structure to construct an answer.
        """
        if visited is None:
            visited = set()
        
        if start.id.code == end.id.code:
            return [start]
        
        if start.id.code in visited:
            return None
        
        visited.add(start.id.code)
        
        # Try explicit relations first
        for rel_type, targets in start.relations.items():
            for target_code in targets:
                if target_code == end.id.code:
                    self._audit.append("DERIVE", "path", start.id.code, "->", end.id.code)
                    return [start, end]
                
                target_node = self._nodes.get(target_code)
                if target_node:
                    sub_path = self.derive_path(target_node, end, visited)
                    if sub_path:
                        return [start] + sub_path
        
        # Try structural similarity (siblings might connect)
        for sibling in self.derive_siblings(start):
            if sibling.id.code not in visited:
                sub_path = self.derive_path(sibling, end, visited)
                if sub_path:
                    return [start, "(via sibling)"] + sub_path
        
        return None
    
    def derive_tension(self, node: Node) -> float:
        """
        DERIVED: how contested/unstable is this concept?
        
        High tension = many forks + few relations = needs attention
        Low tension = stable meaning, well-connected
        
        Tension = (fork_count + 1) / (relation_count + 1)
        """
        fork = self._forks.get(node.id.code)
        fork_count = fork.branch_count if fork else 0
        relation_count = node.relation_count()
        
        tension = (fork_count + 1) / (relation_count + 1)
        self._audit.append("DERIVE", "tension", node.id.code, f"{tension:.2f}")
        return tension
    
    def derive_neighbors(
        self, 
        node: Node, 
        max_distance: int = 2
    ) -> List[Tuple[Node, int]]:
        """
        DERIVED: all nodes within semantic distance.
        """
        self._audit.append("DERIVE", "neighbors", node.id.code, max_distance)
        result = []
        for other in self._nodes.values():
            if other.id.code != node.id.code:
                dist = node.id.distance(other.id)
                if dist <= max_distance:
                    result.append((other, dist))
        return sorted(result, key=lambda x: x[1])
    
    # -------------------------------------------------------------------------
    # STATISTICS: Understanding the space
    # -------------------------------------------------------------------------
    
    @property
    def node_count(self) -> int:
        return len(self._nodes)
    
    @property 
    def relation_count(self) -> int:
        return sum(n.relation_count() for n in self._nodes.values())
    
    @property
    def fork_count(self) -> int:
        return len(self._forks)
    
    @property
    def derivation_ratio(self) -> float:
        """
        How much more can we derive than we store?
        
        Derivable queries ≈ n² (any node to any node)
        Stored facts = nodes + relations
        
        This ratio should be >> 1 for the system to be worthwhile.
        """
        stored = self.node_count + self.relation_count
        derivable = self.node_count ** 2  # pairwise queries
        return derivable / stored if stored > 0 else 0
    
    def nodes(self) -> List[Node]:
        """All nodes in the space."""
        return list(self._nodes.values())
    
    def forks(self) -> List[Fork]:
        """All forks in the space."""
        return list(self._forks.values())
    
    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        return {
            "nodes": self.node_count,
            "relations": self.relation_count,
            "forks": self.fork_count,
            "derivation_ratio": f"{self.derivation_ratio:.0f}x",
            "audit_entries": len(self._audit),
            "chain_valid": self._audit.verify()
        }


# =============================================================================
# WRAPPER: The deterministic constraint layer
# =============================================================================

class Strategy(Enum):
    """
    The finite set of actions the system can take.
    
    Strategies are explicit, testable, auditable.
    The neural component (if any) selects among these.
    It cannot invent new strategies.
    """
    CREATE_NODE = auto()
    RETURN_EXISTING = auto()
    CREATE_FORK = auto()
    RESOLVE_FORK = auto()
    ADD_RELATION = auto()
    DERIVE_PATH = auto()
    DERIVE_SIBLINGS = auto()
    DERIVE_TENSION = auto()
    ESCALATE = auto()  # defer to human


@dataclass
class Decision:
    """A logged decision with full context."""
    strategy: Strategy
    input_data: Any
    context: Dict[str, Any]
    result: Any
    audit_hash: str


class DeterministicWrapper:
    """
    The constraint layer that makes neural systems accountable.
    
    Instead of allowing arbitrary outputs, the system must:
    1. Propose a strategy from a finite set
    2. Apply the strategy to the semantic space
    3. Log everything with cryptographic signatures
    
    The wrapper inverts the typical "AI wrapper" meaning.
    In VC-speak, "wrapper" = thin layer over someone else's model.
    Here, the wrapper IS the value. The neural component is the commodity.
    
    The model provides intelligence. The wrapper provides accountability.
    """
    
    def __init__(self, space: Optional[Space] = None):
        self.space = space or Space()
        self._decisions: List[Decision] = []
    
    @property
    def audit(self) -> AuditLog:
        return self.space.audit
    
    def propose_strategy(self, input_data: Any, context: Dict[str, Any]) -> Strategy:
        """
        Propose a strategy for handling input.
        
        Override this in subclasses to add neural selection.
        Default implementation uses simple rules.
        """
        # This is where a neural component would plug in
        # For now, simple rule-based selection
        
        if isinstance(input_data, str):
            label = input_data.strip()
            existing = self.space.find_by_label(label)
            
            if existing:
                # Check if ambiguous (multiple matches)
                if len(existing) > 1:
                    return Strategy.CREATE_FORK
                return Strategy.RETURN_EXISTING
            
            return Strategy.CREATE_NODE
        
        return Strategy.ESCALATE
    
    def apply_strategy(
        self, 
        strategy: Strategy, 
        input_data: Any, 
        context: Dict[str, Any]
    ) -> Any:
        """
        Apply a strategy. Must be deterministic.
        Same space state + same input = same result.
        """
        if strategy == Strategy.CREATE_NODE:
            return self._create_node(input_data, context)
        
        if strategy == Strategy.RETURN_EXISTING:
            return self.space.find_by_label(input_data)
        
        if strategy == Strategy.CREATE_FORK:
            return self._create_fork(input_data, context)
        
        if strategy == Strategy.ESCALATE:
            return {"escalated": True, "input": input_data}
        
        raise ValueError(f"Unhandled strategy: {strategy}")
    
    def _create_node(self, label: str, context: Dict[str, Any]) -> Node:
        """Create a new node with auto-assigned ID."""
        major = context.get("major", 8)  # Default to Abstract
        type_ = context.get("type", 1)
        subtype = context.get("subtype", 1)
        
        # Find next available instance
        existing = [
            n.id.instance for n in self.space.nodes()
            if n.id.major == major and n.id.type_ == type_ and n.id.subtype == subtype
        ]
        instance = max(existing, default=0) + 1
        
        node = Node(
            id=SemanticID.create(major, type_, subtype, instance),
            label=label,
            metadata=context.get("metadata", {})
        )
        return self.space.add(node)
    
    def _create_fork(self, label: str, context: Dict[str, Any]) -> Tuple[Fork, List[Node]]:
        """Create a fork for an ambiguous label."""
        existing = self.space.find_by_label(label)
        if not existing:
            raise ValueError(f"Cannot fork non-existent label: {label}")
        
        source = existing[0]
        branch_labels = context.get("branch_labels", ["sense_1", "sense_2"])
        return self.space.create_fork(source, branch_labels)
    
    def handle(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Decision:
        """
        Full handling pipeline:
        1. Propose strategy
        2. Apply strategy
        3. Log decision
        4. Return result with audit proof
        """
        context = context or {}
        
        strategy = self.propose_strategy(input_data, context)
        result = self.apply_strategy(strategy, input_data, context)
        
        # Log the decision
        entry = self.audit.append(
            "DECISION",
            strategy.name,
            str(input_data)[:100],  # Truncate for log
            str(result)[:100]
        )
        
        decision = Decision(
            strategy=strategy,
            input_data=input_data,
            context=context,
            result=result,
            audit_hash=entry.hash
        )
        self._decisions.append(decision)
        
        return decision
    
    @property
    def decisions(self) -> List[Decision]:
        return list(self._decisions)


# =============================================================================
# CONVENIENCE: Quick creation helpers
# =============================================================================

def create_space() -> Space:
    """Create an empty semantic space."""
    return Space()


def create_wrapper(space: Optional[Space] = None) -> DeterministicWrapper:
    """Create a wrapper around a space."""
    return DeterministicWrapper(space)


def quick_node(
    major: int, 
    type_: int, 
    subtype: int, 
    instance: int, 
    label: str,
    **metadata: Any
) -> Node:
    """Quickly create a node."""
    return Node(
        id=SemanticID.create(major, type_, subtype, instance),
        label=label,
        metadata=metadata
    )
