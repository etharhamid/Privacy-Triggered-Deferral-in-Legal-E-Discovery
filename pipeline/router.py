"""
router.py
Implements the three governance strategies (§3.5):
  1. autonomous           — always AI, no human in the loop
  2. confidence_only      — defer if c(x) < tau_c
  3. privacy_triggered    — defer if c(x) < tau_c OR r(x) > tau_r
"""

from dataclasses import dataclass


@dataclass
class RoutingDecision:
    defer: bool
    reason: str   # "auto" | "low_confidence" | "high_risk"


def route_autonomous(conf: float = 0, risk: float = 0, **_) -> RoutingDecision:
    return RoutingDecision(defer=False, reason="auto")


def route_confidence_only(conf: float, tau_c: float = 0.7, **_) -> RoutingDecision:
    if conf < tau_c:
        return RoutingDecision(defer=True, reason="low_confidence")
    return RoutingDecision(defer=False, reason="auto")


def route_privacy_triggered(
    conf: float, risk: float, tau_c: float = 0.7, tau_r: float = 0.5, **_
) -> RoutingDecision:
    if conf < tau_c:
        return RoutingDecision(defer=True, reason="low_confidence")
    if risk > tau_r:
        return RoutingDecision(defer=True, reason="high_risk")
    return RoutingDecision(defer=False, reason="auto")


POLICIES = {
    "autonomous":         route_autonomous,
    "confidence_only":    route_confidence_only,
    "privacy_triggered":  route_privacy_triggered,
}