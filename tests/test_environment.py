"""Tests for VendorNegotiationEnvironment."""
import pytest
from server.environment import VendorNegotiationEnvironment
from models import NegotiationAction, NegotiationObservation, NegotiationState


class TestReset:
    def test_reset_returns_observation(self):
        env = VendorNegotiationEnvironment()
        obs = env.reset(seed=42)
        assert isinstance(obs, NegotiationObservation)
        assert obs.done is False
        assert obs.scenario_id != ""
        assert obs.counterparty_type != ""
        assert len(obs.current_terms) > 0

    def test_reset_with_task_id(self):
        env = VendorNegotiationEnvironment()
        obs = env.reset(seed=42, task_id="strategic_contract_close")
        assert obs.max_rounds == 8

    def test_reset_is_reproducible(self):
        env = VendorNegotiationEnvironment()
        obs1 = env.reset(seed=42)
        obs2 = env.reset(seed=42)
        assert obs1.scenario_id == obs2.scenario_id

    def test_state_after_reset(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42)
        state = env.state
        assert isinstance(state, NegotiationState)
        assert state.round_index == 0
        assert state.is_done is False

    def test_state_hides_counterparty_weights(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42)
        state_dict = env.state.model_dump()
        assert "cp_utility_weights" not in state_dict
        assert "cp_acceptance_threshold" not in state_dict


class TestStep:
    def test_analyze_deal(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42, task_id="deal_qualification")
        obs = env.step(NegotiationAction(action_type="analyze_deal", focus_terms=["price"]))
        assert obs.reward > 0  # +0.1 minus small round cost

    def test_analyze_deal_idempotent(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42, task_id="multi_term_negotiation")
        env.step(NegotiationAction(action_type="analyze_deal"))
        obs2 = env.step(NegotiationAction(action_type="analyze_deal"))
        assert obs2.reward <= 0  # no reward for re-analyzing

    def test_propose_terms(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42, task_id="multi_term_negotiation")
        obs = env.step(NegotiationAction(action_type="propose_terms", price=70000))
        assert isinstance(obs, NegotiationObservation)
        assert obs.counterparty_message != ""

    def test_accept_deal_with_constraint_violation(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42, task_id="deal_qualification")
        obs = env.step(NegotiationAction(action_type="accept_deal"))
        # Initial terms may violate constraints, so close may fail
        assert isinstance(obs, NegotiationObservation)
        assert "constraint" in obs.last_action_feedback.lower() or obs.done is True

    def test_accept_counterparty_offer(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42, task_id="strategic_contract_close")
        favorable = dict(env._internal.current_terms)
        constraints = env._internal.company_constraints
        for key, limit in constraints.items():
            term, direction = key.rsplit("_", 1)
            if term not in favorable:
                continue
            if direction == "max" and favorable[term] > limit:
                favorable[term] = limit
            elif direction == "min" and favorable[term] < limit:
                favorable[term] = limit
        env._internal.reservation_value = 0.0
        env._internal.counterparty_last_offer = dict(favorable)
        obs = env.step(NegotiationAction(action_type="accept_counterparty_offer"))
        assert obs.done is True
        assert env.state.accepted is True
        assert env.state.current_terms == favorable

    def test_walk_away(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42, task_id="deal_qualification")
        obs = env.step(NegotiationAction(action_type="walk_away", reason="Too expensive"))
        assert obs.done is True

    def test_max_rounds_ends_episode(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42, task_id="deal_qualification")  # 4 rounds
        for _ in range(3):
            env.step(NegotiationAction(action_type="ask_clarification", question="Tell me more"))
        obs = env.step(NegotiationAction(action_type="ask_clarification", question="More"))
        assert obs.done is True

    def test_step_before_reset_raises(self):
        env = VendorNegotiationEnvironment()
        with pytest.raises(RuntimeError):
            env.step(NegotiationAction(action_type="accept_deal"))

    def test_step_after_done_raises(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42)
        env.step(NegotiationAction(action_type="walk_away", reason="test"))
        with pytest.raises(RuntimeError):
            env.step(NegotiationAction(action_type="accept_deal"))

    def test_available_actions(self):
        env = VendorNegotiationEnvironment()
        obs = env.reset(seed=42)
        assert "analyze_deal" in obs.available_actions
        assert "accept_deal" in obs.available_actions

    def test_request_tradeoff_requires_offered_term(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42, task_id="multi_term_negotiation")
        obs = env.step(NegotiationAction(action_type="request_tradeoff", requested_term="price"))
        assert obs.reward < 0
        assert "offered_term" in obs.last_action_feedback

    def test_immediate_walkaway_is_low_quality(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42, task_id="deal_qualification")
        env.step(NegotiationAction(action_type="walk_away", reason="nope"))
        assert env.grade() < 0.4


class TestGrading:
    def test_grade_returns_float_0_to_1(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42, task_id="deal_qualification")
        env.step(NegotiationAction(action_type="accept_deal"))
        score = env.grade()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_grade_score_on_state_after_done(self):
        env = VendorNegotiationEnvironment()
        env.reset(seed=42, task_id="deal_qualification")
        env.step(NegotiationAction(action_type="walk_away", reason="test"))
        state = env.state
        assert state.grade_score is not None

    def test_scores_vary_across_episodes(self):
        env = VendorNegotiationEnvironment()
        scores = []
        for seed in range(10):
            env.reset(seed=seed, task_id="deal_qualification")
            env.step(NegotiationAction(action_type="accept_deal"))
            scores.append(env.grade())
        assert len(set(scores)) > 1
