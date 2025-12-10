"""
ScreenGhost simulated layer.

This module generates a small adversarial event stream that stands in
for real screen captures and automation logs.

It is designed so that the rest of the stack can run without any OS
specific setup, but you can later replace this with your real ScreenGhost
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Event:
    topic: str
    source: str
    text: str
    confidence: float


def generate_demo_events() -> List[Event]:
    """
    Generate a fixed set of conflicting events for one topic.
    These events are intentionally inconsistent, so the tension score
    will be high enough to create an alert.
    """
    topic = "election_integrity"
    return [
        Event(topic, "feed_a", "The election servers were never exposed to the internet.", 0.9),
        Event(topic, "feed_b", "Independent audit found remote access enabled on six machines.", 0.8),
        Event(topic, "feed_c", "No evidence of remote access was recorded in the final report.", 0.6),
        Event(topic, "feed_d", "Screenshots show remote sessions during voting hours.", 0.7),
    ]


def main() -> None:
    print("ScreenGhost simulated demo events:")
    events = generate_demo_events()
    for e in events:
        print(f"- [{e.topic}] ({e.source}, {e.confidence:.2f}) {e.text}")


if __name__ == "__main__":
    main()
