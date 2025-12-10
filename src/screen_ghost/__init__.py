"""
Screen Ghost v0.7.0

The perception layer of the GhostBox stack.
Captures events from adversarial information environments.

New in v0.7:
- Real capture sources (file, database, webhook, polling)
- Attention geometry (shapes that show where to look)
- Analyst session continuity (thread doesn't reset on rotation)
"""

from .simulated import Event, generate_demo_events
from .source import ScreenSource, SimulatedScreenSource, ScreenGhostSource

from .capture import (
    CaptureSource,
    JSONFileSource,
    CSVFileSource,
    LogFileSource,
    SQLiteSource,
    ScreenGhostDBSource,
    WebhookSource,
    PollingSource,
    MultiplexSource,
    app_based_topic,
    keyword_topic,
)

from .attention import (
    ShapeType,
    AttentionShape,
    AttentionMap,
    AttentionGeometry,
    ContradictionDetector,
    VelocityDetector,
    ConvergenceDetector,
    ConfidenceDropDetector,
    SilenceDetector,
)

from .session import (
    HandoffStatus,
    AnalystNote,
    FocusArea,
    AnalystSession,
    SessionManager,
)

__version__ = "0.7.0"

__all__ = [
    # Version
    "__version__",
    # Core events
    "Event",
    "generate_demo_events",
    # Sources (v0.6)
    "ScreenSource",
    "SimulatedScreenSource",
    "ScreenGhostSource",
    # Capture sources (v0.7)
    "CaptureSource",
    "JSONFileSource",
    "CSVFileSource",
    "LogFileSource",
    "SQLiteSource",
    "ScreenGhostDBSource",
    "WebhookSource",
    "PollingSource",
    "MultiplexSource",
    "app_based_topic",
    "keyword_topic",
    # Attention geometry (v0.7)
    "ShapeType",
    "AttentionShape",
    "AttentionMap",
    "AttentionGeometry",
    "ContradictionDetector",
    "VelocityDetector",
    "ConvergenceDetector",
    "ConfidenceDropDetector",
    "SilenceDetector",
    # Session continuity (v0.7)
    "HandoffStatus",
    "AnalystNote",
    "FocusArea",
    "AnalystSession",
    "SessionManager",
]
