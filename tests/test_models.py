"""Tests for Pydantic models."""
import pytest
from models import NegotiationAction, NegotiationObservation, NegotiationState


class TestNegotiationAction:
    def test_propose_terms(self):
        action = NegotiationAction(action_type="propose_terms", price=70000)
        assert action.action_type == "propose_terms"
        assert action.price == 70000

    def test_invalid_action_type(self):
        with pytest.raises(Exception):
            NegotiationAction(action_type="invalid")

    def test_walk_away(self):
        action = NegotiationAction(action_type="walk_away", reason="Too expensive")
        assert action.reason == "Too expensive"


class TestNegotiationObservation:
    def test_defaults(self):
        obs = NegotiationObservation()
        assert obs.done is False
        assert obs.negotiation_status == "in_progress"


class TestNegotiationState:
    def test_defaults(self):
        state = NegotiationState()
        assert state.is_done is False
        assert state.accepted is False
        assert state.walked_away is False

    def test_no_hidden_fields(self):
        state_dict = NegotiationState().model_dump()
        assert "cp_utility_weights" not in state_dict
