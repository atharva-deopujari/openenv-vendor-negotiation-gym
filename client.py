"""
WebSocket client for the Vendor Negotiation Gym.
Typed, async-first interface via EnvClient subclass.
"""
from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import NegotiationAction, NegotiationObservation, NegotiationState


class NegotiationEnv(EnvClient[NegotiationAction, NegotiationObservation, NegotiationState]):
    """Typed client for the Vendor Negotiation Gym."""

    def _step_payload(self, action: NegotiationAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"action_type": action.action_type}
        for field in [
            "focus_terms", "risk_notes", "reservation_value", "price", "contract_months",
            "payment_terms_days", "sla_uptime", "support_tier",
            "onboarding_days", "service_credits_percent", "message",
            "term_name", "new_value", "concession_reason", "question",
            "give_term", "give_value", "get_term", "get_value",
            "package_terms", "package_name", "meso_offers",
            "requested_term", "offered_term", "reason",
        ]:
            val = getattr(action, field, None)
            if val is not None:
                payload[field] = val
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[NegotiationObservation]:
        obs_data = payload.get("observation", {})
        observation = NegotiationObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> NegotiationState:
        return NegotiationState(**payload)
