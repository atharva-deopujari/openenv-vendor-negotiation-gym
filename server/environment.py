"""
Core Environment class for the Vendor Negotiation Gym.
Implements reset(), step(), state per OpenEnv spec.

Features:
- Multi-term structured negotiation
- Deterministic counterparty response engine
- Hidden counterparty utility model
- available_actions guidance
- Idempotent analysis (no reward for re-analyzing)
- Dense reward shaping
- SLA, payment, onboarding, pricing across real deal dimensions
"""
import uuid
from typing import Optional, Any, List, Dict

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import (
    NegotiationAction, NegotiationObservation, NegotiationState, _InternalState,
)
from scenarios import ScenarioDatabase
from deal_engine import DealEngine

TASKS = {
    "deal_qualification": {
        "name": "Deal Qualification",
        "difficulty": "easy",
        "description": (
            "Identify a viable deal posture without giving away value too early. "
            "Analyze the deal, ask clarifications if useful, make an initial "
            "reasonable proposal, and close or walk away appropriately."
        ),
        "max_rounds": 4,
        "grading_weights": {
            "target_alignment": 0.45,
            "constraint_compliance": 0.35,
            "action_efficiency": 0.20,
        },
    },
    "multi_term_negotiation": {
        "name": "Structured Negotiation",
        "difficulty": "medium",
        "description": (
            "Negotiate across multiple terms using concessions and bundles, "
            "not only price. Improve deal quality while preserving must-have terms."
        ),
        "max_rounds": 6,
        "grading_weights": {
            "deal_quality": 0.35,
            "strategic_concessions": 0.25,
            "constraint_compliance": 0.20,
            "efficiency": 0.20,
        },
    },
    "strategic_contract_close": {
        "name": "Contract Close Under Hidden Preferences",
        "difficulty": "hard",
        "description": (
            "Close the best possible deal against a counterparty with hidden "
            "priorities and limited rounds. Infer hidden preferences from signals, "
            "trade low-value concessions for high-value gains, and decide when "
            "to close or walk away."
        ),
        "max_rounds": 8,
        "grading_weights": {
            "final_deal_utility": 0.40,
            "counterparty_signal_exploitation": 0.20,
            "constraint_compliance": 0.20,
            "close_decision_quality": 0.10,
            "efficiency": 0.10,
        },
    },
}


class VendorNegotiationEnvironment(
    Environment[NegotiationAction, NegotiationObservation, NegotiationState]
):
    """Vendor Negotiation Gym -- enterprise deal negotiation environment."""

    def __init__(self):
        super().__init__()
        self.db = ScenarioDatabase()
        self._internal: Optional[_InternalState] = None
        self._scenario: Optional[dict] = None
        self._engine: Optional[DealEngine] = None
        self._task_id: str = "deal_qualification"
        self._last_cp_message: str = ""
        self._last_signals: List[str] = []
        self._analyzed: bool = False

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs):
        if task_id:
            self._task_id = task_id

        scenario = self.db.get_random_scenario(seed=seed)
        self._scenario = scenario
        self._engine = DealEngine(scenario)
        self._analyzed = False

        ep_id = episode_id or str(uuid.uuid4())
        task_config = TASKS.get(self._task_id, TASKS["deal_qualification"])
        max_rounds = task_config["max_rounds"]

        initial_terms = dict(scenario["initial_terms"])
        initial_company_util = self._engine.compute_company_utility(
            initial_terms, scenario["company_targets"]
        )
        batna_value = self._compute_batna_value(initial_company_util)

        self._internal = _InternalState(
            episode_id=ep_id,
            step_count=0,
            task_id=self._task_id,
            round_index=0,
            max_rounds=max_rounds,
            is_done=False,
            action_log=[],
            current_terms=initial_terms,
            cumulative_reward=0.0,
            estimated_deal_quality=0.0,
            accepted=False,
            walked_away=False,
            grade_score=None,
            reservation_value=round(min(0.95, batna_value + 0.08), 4),
            # Internal
            scenario_id=scenario["id"],
            counterparty_type=scenario["counterparty_type"],
            cp_utility_weights=scenario["cp_utility_weights"],
            cp_acceptance_threshold=scenario["cp_acceptance_threshold"],
            cp_ideal_terms=scenario["cp_ideal_terms"],
            cp_floor_terms=scenario.get("cp_floor_terms", {}),
            company_targets=scenario["company_targets"],
            company_constraints=scenario["company_constraints"],
            must_have_terms=scenario["must_have_terms"],
            initial_terms=initial_terms,
            clarifications_asked=0,
            concessions_made=[],
            terms_touched=[],
            successful_tradeoffs=0,
            close_attempts=0,
            successful_packages=0,
            successful_mesos=0,
            package_offer_attempts=0,
            meso_offer_attempts=0,
            reservation_value_set=False,
            reservation_breached=False,
            batna_value=batna_value,
            counterparty_last_offer=None,
            offer_history=[],
            inferred_counterparty_priorities=[],
            constraint_violations=0,
            deal_rejected=False,
        )

        # Initial counterparty message
        self._last_cp_message = f"Welcome. We'd like to discuss {scenario['title']}. Here are our initial terms."
        self._last_signals = scenario.get("signals", [])[:2]

        return self._make_observation(
            reward=None,
            done=False,
            feedback="Negotiation started. Review the terms and begin.",
        )

    def step(self, action, timeout_s=None, **kwargs):
        if self._internal is None:
            raise RuntimeError("Must call reset() before step()")
        if self._internal.is_done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        s = self._internal
        s.step_count += 1
        s.round_index += 1
        s.action_log.append(action.action_type)
        reward = 0.0
        feedback = ""
        done = False

        # flat per-round cost
        round_cost = 0.012
        reward -= round_cost

        prev_company_util = self._engine.compute_company_utility(
            s.current_terms, s.company_targets
        )


        if action.action_type == "analyze_deal":
            if self._analyzed:
                reward += 0.0
                feedback = "Already analyzed. No new information."
            else:
                self._analyzed = True
                reward += 0.1
                s.inferred_counterparty_priorities = self._infer_priorities()
                feedback = "Deal analyzed. Key terms identified."

        elif action.action_type == "set_reservation_value":
            if action.reservation_value is None:
                reward += -0.05
                feedback = "Missing reservation_value."
            else:
                value = float(action.reservation_value)
                s.reservation_value = round(value, 4)
                s.reservation_value_set = True
                if value < s.batna_value:
                    reward += -0.08
                    feedback = "Reservation value is below BATNA. This weakens your negotiating stance."
                elif value > 0.95:
                    reward += -0.03
                    feedback = "Reservation value is unrealistically high."
                else:
                    reward += 0.04
                    feedback = "Reservation value set."

        elif action.action_type == "propose_terms":
            proposed = self._extract_proposed_terms(action, s.current_terms)
            reward_delta, done, feedback = self._process_offer(
                s, proposed, prev_company_util, offer_kind="proposal"
            )
            reward += reward_delta

        elif action.action_type == "counter_offer":
            baseline_terms = s.counterparty_last_offer or s.current_terms
            proposed = self._extract_proposed_terms(action, baseline_terms)
            reward_delta, done, feedback = self._process_offer(
                s, proposed, prev_company_util, offer_kind="counter_offer"
            )
            reward += reward_delta

        elif action.action_type == "concede_term":
            if not action.term_name or action.new_value is None:
                reward += -0.05
                feedback = "Missing term_name or new_value."
            elif action.term_name not in s.current_terms:
                reward += -0.05
                feedback = f"Unknown term '{action.term_name}'."
            else:
                terms = dict(s.current_terms)
                terms[action.term_name] = action.new_value
                s.current_terms = terms
                s.concessions_made.append(action.term_name)
                s.terms_touched.append(action.term_name)

                new_company_util = self._engine.compute_company_utility(
                    terms, s.company_targets
                )
                delta = new_company_util - prev_company_util

                # Concession should hurt your utility but help the deal
                cp_util = self._engine.compute_counterparty_utility(terms)
                if delta < 0 and cp_util > 0.4:
                    reward += 0.05  # strategic concession
                    feedback = f"Conceded {action.term_name}. Counterparty receptive."
                elif delta < -0.1:
                    reward += -0.1  # gave away too much
                    feedback = f"Conceded {action.term_name}. Significant value given away."
                else:
                    reward += 0.0
                    feedback = f"Conceded {action.term_name}."

                msg, sigs = self._engine.generate_response(
                    terms, s.round_index, s.max_rounds
                )
                self._last_cp_message = msg
                self._last_signals = sigs

        elif action.action_type == "ask_clarification":
            if s.clarifications_asked >= 2:
                reward += -0.05
                feedback = "Too many clarifications. Counterparty is impatient."
            else:
                s.clarifications_asked += 1
                reward += 0.02
                # Reveal a new signal
                all_signals = self._scenario.get("signals", [])
                idx = min(s.clarifications_asked + 1, len(all_signals) - 1)
                if idx < len(all_signals):
                    self._last_signals = [all_signals[idx]]
                feedback = "Clarification received."
                self._last_cp_message = "Good question. Here's some additional context."
                s.inferred_counterparty_priorities = self._infer_priorities(limit=3)

        elif action.action_type == "bundle_offer":
            if not all([action.give_term, action.give_value,
                       action.get_term, action.get_value]):
                reward += -0.05
                feedback = "Bundle offer requires give_term, give_value, get_term, get_value."
            elif action.give_term == action.get_term:
                reward += -0.05
                feedback = "Bundle offer must exchange different terms."
            elif action.give_term not in s.current_terms or action.get_term not in s.current_terms:
                reward += -0.05
                feedback = "Bundle offer references unknown term(s)."
            else:
                terms = dict(s.current_terms)
                terms[action.give_term] = action.give_value
                terms[action.get_term] = action.get_value

                violations = self._engine.check_constraint_violations(
                    terms, s.company_constraints
                )
                if violations:
                    s.constraint_violations += len(violations)
                    reward += -0.15
                    feedback = f"Bundle violates constraints: {'; '.join(violations)}"
                else:
                    previous_terms = dict(s.current_terms)
                    prev_cp_util = self._engine.compute_counterparty_utility(
                        previous_terms
                    )
                    s.current_terms = terms
                    s.terms_touched.extend(
                        self._changed_terms(previous_terms, terms)
                    )
                    new_company_util = self._engine.compute_company_utility(
                        terms, s.company_targets
                    )
                    delta = new_company_util - prev_company_util
                    cp_delta = self._engine.compute_counterparty_utility(terms) - prev_cp_util
                    if delta >= 0 and cp_delta > 0:
                        reward += 0.15
                        s.successful_tradeoffs += 1
                        feedback = "Bundle improved the deal for both sides."
                    elif cp_delta <= 0:
                        reward += -0.05
                        feedback = "Bundle did not create a credible tradeoff."
                    else:
                        reward += delta * 0.3
                        feedback = "Bundle submitted."

                    msg, sigs = self._engine.generate_response(
                        terms, s.round_index, s.max_rounds
                    )
                    self._last_cp_message = msg
                    self._last_signals = sigs

        elif action.action_type == "make_package_offer":
            s.package_offer_attempts += 1
            package_terms = action.package_terms or {}
            if len(package_terms) < 2:
                reward += -0.06
                feedback = "Package offer requires at least two terms."
            else:
                proposed = dict(s.current_terms)
                proposed.update(package_terms)
                reward_delta, done, feedback = self._process_offer(
                    s,
                    proposed,
                    prev_company_util,
                    offer_kind="package_offer",
                    require_multi_term=True,
                )
                reward += reward_delta
                if "accepted" in feedback.lower() or "counterparty is favorable" in feedback.lower():
                    s.successful_packages += 1

        elif action.action_type == "make_meso_offer":
            s.meso_offer_attempts += 1
            offers = action.meso_offers or []
            valid_offers = []
            for offer in offers[:3]:
                candidate = dict(s.current_terms)
                candidate.update(offer)
                if not self._engine.check_constraint_violations(candidate, s.company_constraints):
                    valid_offers.append(candidate)
            if len(valid_offers) < 2:
                reward += -0.08
                feedback = "MESO requires at least two compliant package offers."
            else:
                chosen = max(
                    valid_offers,
                    key=lambda terms: self._engine.compute_counterparty_utility(terms),
                )
                reward_delta, done, feedback = self._process_offer(
                    s,
                    chosen,
                    prev_company_util,
                    offer_kind="meso_offer",
                    require_multi_term=True,
                )
                reward += reward_delta + 0.03
                if "accepted" in feedback.lower() or "counterparty is favorable" in feedback.lower():
                    s.successful_mesos += 1

        elif action.action_type == "request_tradeoff":
            if not action.requested_term:
                reward += -0.05
                feedback = "Missing requested_term."
            elif not action.offered_term:
                reward += -0.08
                feedback = "Tradeoff request requires an offered_term."
            elif action.requested_term == action.offered_term:
                reward += -0.08
                feedback = "Tradeoff request must exchange different terms."
            elif action.requested_term not in s.current_terms or action.offered_term not in s.current_terms:
                reward += -0.08
                feedback = "Tradeoff request references unknown term(s)."
            else:
                requested_weight = abs(self._engine.cp_weights.get(action.requested_term, 0.5))
                offered_weight = abs(self._engine.cp_weights.get(action.offered_term, 0.0))
                terms = dict(s.current_terms)
                moved = False

                offered_ideal = self._scenario["cp_ideal_terms"].get(action.offered_term)
                offered_current = terms.get(action.offered_term)
                if (
                    offered_ideal is not None
                    and isinstance(offered_current, (int, float))
                    and isinstance(offered_ideal, (int, float))
                ):
                    terms[action.offered_term] = round(
                        offered_current + 0.35 * (offered_ideal - offered_current), 2
                    )
                    if isinstance(self._scenario["initial_terms"].get(action.offered_term), int):
                        terms[action.offered_term] = int(round(terms[action.offered_term]))
                    moved = True

                requested_target = self._scenario["company_targets"].get(action.requested_term)
                requested_current = terms.get(action.requested_term)
                if (
                    requested_target is not None
                    and isinstance(requested_current, (int, float))
                    and isinstance(requested_target, (int, float))
                    and offered_weight >= max(0.12, requested_weight * 0.8)
                ):
                    terms[action.requested_term] = round(
                        requested_current + 0.35 * (requested_target - requested_current), 2
                    )
                    if isinstance(self._scenario["initial_terms"].get(action.requested_term), int):
                        terms[action.requested_term] = int(round(terms[action.requested_term]))
                    moved = True

                if not moved:
                    reward += -0.05
                    feedback = "Tradeoff request was not credible."
                else:
                    violations = self._engine.check_constraint_violations(
                        terms, s.company_constraints
                    )
                    prev_cp_util = self._engine.compute_counterparty_utility(s.current_terms)
                    prev_util = prev_company_util
                    if violations:
                        s.constraint_violations += len(violations)
                        reward += -0.12
                        feedback = f"Tradeoff violates constraints: {'; '.join(violations)}"
                    else:
                        new_util = self._engine.compute_company_utility(
                            terms, s.company_targets
                        )
                        new_cp_util = self._engine.compute_counterparty_utility(terms)
                        previous_terms = dict(s.current_terms)
                        s.current_terms = terms
                        s.terms_touched.extend(
                            self._changed_terms(previous_terms, terms)
                        )
                        if new_util >= prev_util and new_cp_util > prev_cp_util:
                            reward += 0.12
                            s.successful_tradeoffs += 1
                            feedback = f"Counterparty traded on {action.requested_term} for movement on {action.offered_term}."
                        else:
                            reward += -0.03
                            feedback = "Tradeoff moved terms but did not materially improve the deal."

                msg, sigs = self._engine.generate_response(
                    s.current_terms, s.round_index, s.max_rounds
                )
                self._last_cp_message = msg
                self._last_signals = sigs

        elif action.action_type in {"accept_deal", "final_offer"}:
            s.close_attempts += 1
            reward_delta, done, feedback = self._attempt_close(
                s, s.current_terms, accept_counterparty=False
            )
            reward += reward_delta

        elif action.action_type == "accept_counterparty_offer":
            s.close_attempts += 1
            if not s.counterparty_last_offer:
                reward += -0.08
                feedback = "There is no counterparty offer to accept."
            else:
                reward_delta, done, feedback = self._attempt_close(
                    s, s.counterparty_last_offer, accept_counterparty=True
                )
                reward += reward_delta

        elif action.action_type == "walk_away":
            s.walked_away = True
            done = True
            company_util = self._engine.compute_company_utility(
                s.current_terms, s.company_targets
            )
            if s.reservation_value is not None and company_util < s.reservation_value:
                reward += 0.08
                feedback = "Walked away from a deal below your reservation value."
            if company_util < 0.2:
                reward += 0.15  # smart walk-away from bad deal
                feedback = "Walked away from an unfavorable deal. Good decision."
            elif company_util > 0.5:
                reward += -0.2  # walked away from a good deal
                feedback = "Walked away from a viable deal. Opportunity lost."
            else:
                reward += 0.0
                feedback = "Walked away. Neutral outcome."


        if s.round_index >= s.max_rounds and not done:
            done = True
            feedback += " Maximum rounds reached."


        s.is_done = done
        s.cumulative_reward += reward
        s.estimated_deal_quality = self._engine.compute_company_utility(
            s.current_terms, s.company_targets
        )

        if done:
            s.grade_score = self._compute_grade()

        return self._make_observation(reward=reward, done=done, feedback=feedback)

    @property
    def state(self) -> NegotiationState:
        if self._internal is None:
            return NegotiationState()
        s = self._internal
        return NegotiationState(
            episode_id=s.episode_id,
            step_count=s.step_count,
            task_id=s.task_id,
            round_index=s.round_index,
            max_rounds=s.max_rounds,
            is_done=s.is_done,
            action_log=list(s.action_log),
            current_terms=dict(s.current_terms),
            cumulative_reward=s.cumulative_reward,
            estimated_deal_quality=s.estimated_deal_quality,
            accepted=s.accepted,
            walked_away=s.walked_away,
            grade_score=s.grade_score,
            reservation_value=s.reservation_value,
        )

    def grade(self) -> float:
        return self._compute_grade()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="vendor-negotiation-gym",
            description=(
                "An enterprise vendor negotiation environment where AI agents "
                "negotiate multi-term contracts with deterministic counterparties "
                "under budget, SLA, and policy constraints."
            ),
            version="0.1.0",
            author="Atharva Deopujari",
        )


    def _compute_grade(self) -> float:
        if self._internal is None:
            return 0.0
        s = self._internal
        task = TASKS.get(s.task_id)
        if not task:
            return 0.0

        weights = task["grading_weights"]
        score = 0.0

        company_util = self._engine.compute_company_utility(
            s.current_terms, s.company_targets
        )
        initial_company_util = self._engine.compute_company_utility(
            s.initial_terms, s.company_targets
        )
        progress = max(0.0, company_util - initial_company_util)
        max_progress = max(1e-6, 1.0 - initial_company_util)
        progress_score = min(1.0, progress / max_progress)
        alignment_score = round(0.6 * company_util + 0.4 * progress_score, 4)
        violations = self._engine.check_constraint_violations(
            s.current_terms, s.company_constraints
        )
        compliance = 1.0 if not violations else max(0.0, 1.0 - len(violations) * 0.3)
        efficiency = max(0.0, 1.0 - s.round_index / s.max_rounds)
        engagement = self._engagement_score(s)
        outcome = self._outcome_quality(s, company_util, violations)

        if "target_alignment" in weights:
            score += alignment_score * weights["target_alignment"]
        if "deal_quality" in weights:
            score += alignment_score * weights["deal_quality"]
        if "final_deal_utility" in weights:
            # Full credit if accepted, partial credit for disciplined walk-away
            if s.accepted:
                util = company_util
            elif s.walked_away and company_util < 0.3:
                util = 0.3  # credit for recognizing a bad deal
            else:
                util = company_util * 0.2  # small partial credit
            score += util * weights["final_deal_utility"]
        if "constraint_compliance" in weights:
            score += compliance * weights["constraint_compliance"]
        if "action_efficiency" in weights:
            score += (efficiency * engagement) * weights["action_efficiency"]
        if "efficiency" in weights:
            score += (efficiency * engagement) * weights["efficiency"]
        if "strategic_concessions" in weights:
            concession_quality = 0.0
            if s.successful_tradeoffs > 0 or s.successful_packages > 0 or s.successful_mesos > 0:
                concession_quality = min(
                    1.0,
                    0.35
                    + 0.18 * s.successful_tradeoffs
                    + 0.18 * s.successful_packages
                    + 0.22 * s.successful_mesos,
                )
            elif s.concessions_made:
                concession_quality = max(0.1, 0.6 - len(s.concessions_made) * 0.15)
            score += concession_quality * weights["strategic_concessions"]
        if "counterparty_signal_exploitation" in weights:
            # Did the agent ask clarifications and use tradeoffs?
            signal_use = min(1.0, (
                (0.3 if s.clarifications_asked > 0 else 0.0)
                + (0.4 if "request_tradeoff" in s.action_log else 0.0)
                + (0.15 if "bundle_offer" in s.action_log else 0.0)
                + (0.15 if "make_package_offer" in s.action_log else 0.0)
                + (0.15 if "make_meso_offer" in s.action_log else 0.0)
            ))
            score += signal_use * weights["counterparty_signal_exploitation"]
        if "close_decision_quality" in weights:
            # Good close: accepted a good deal or walked away from a bad one
            if s.accepted and company_util > 0.4 and not violations:
                close_q = 1.0
            elif s.walked_away and company_util < 0.3:
                close_q = 0.8
            elif s.accepted and violations:
                close_q = 0.0
            elif s.walked_away and company_util > 0.5:
                close_q = 0.1
            else:
                close_q = 0.3  # timed out
            score += close_q * weights["close_decision_quality"]

        # Penalty for deals rejected by the counterparty
        if s.deal_rejected:
            score *= 0.8

        # Apply quality multipliers via geometric mean (prevents cascading collapse)
        reservation_quality = self._reservation_discipline_score(s, company_util)
        multiplier = (reservation_quality * engagement * outcome) ** (1 / 3)
        multiplier = max(0.15, multiplier)  # floor so scores don't collapse to 0
        score *= multiplier

        # Clamp to (0, 1) exclusive -- the evaluation harness requires
        # scores strictly between 0 and 1.
        return round(min(max(score, 0.0001), 0.9999), 4)


    def _make_observation(self, reward, done, feedback) -> NegotiationObservation:
        s = self._internal
        sc = self._scenario

        if done:
            status = "accepted" if s.accepted else ("walked_away" if s.walked_away else ("rejected" if s.deal_rejected else "timed_out"))
        else:
            status = "in_progress"

        return NegotiationObservation(
            done=done,
            reward=reward,
            scenario_id=sc["id"],
            task_id=s.task_id,
            round_index=s.round_index,
            max_rounds=s.max_rounds,
            counterparty_type=sc["counterparty_type"],
            counterparty_message=self._last_cp_message,
            signals=self._last_signals,
            inferred_counterparty_priorities=list(s.inferred_counterparty_priorities),
            current_terms=dict(s.current_terms),
            counterparty_last_offer=dict(s.counterparty_last_offer) if s.counterparty_last_offer else None,
            offer_history=list(s.offer_history[-6:]),
            company_targets=sc["company_targets"],
            company_constraints=sc["company_constraints"],
            must_have_terms=sc["must_have_terms"],
            nice_to_have_terms=sc.get("nice_to_have_terms", []),
            batna_summary=self._batna_summary(s),
            reservation_value_hint=round(min(0.95, s.batna_value + 0.08), 4),
            compliance_risks=self._engine.check_constraint_violations(
                s.current_terms, s.company_constraints
            ),
            rounds_remaining=max(0, s.max_rounds - s.round_index),
            available_actions=self._get_available_actions() if not done else [],
            last_action_feedback=feedback,
            negotiation_status=status,
            negotiation_phase=self._negotiation_phase(s),
        )

    def _get_available_actions(self) -> List[str]:
        s = self._internal
        available = []

        if not self._analyzed:
            available.append("analyze_deal")
        if not s.reservation_value_set:
            available.append("set_reservation_value")
        if s.clarifications_asked < 2:
            available.append("ask_clarification")
        available.append("propose_terms")
        if s.counterparty_last_offer is not None:
            available.append("counter_offer")
        available.append("concede_term")
        available.append("bundle_offer")
        available.append("make_package_offer")
        available.append("make_meso_offer")
        available.append("request_tradeoff")
        available.append("accept_deal")
        if s.counterparty_last_offer is not None:
            available.append("accept_counterparty_offer")
        available.append("final_offer")
        available.append("walk_away")

        return available

    @staticmethod
    def _extract_proposed_terms(action: NegotiationAction, current: dict) -> dict:
        terms = dict(current)
        if action.price is not None:
            terms["price"] = action.price
        if action.contract_months is not None:
            terms["contract_months"] = action.contract_months
        if action.payment_terms_days is not None:
            terms["payment_terms_days"] = action.payment_terms_days
        if action.sla_uptime is not None:
            terms["sla_uptime"] = action.sla_uptime
        if action.support_tier is not None:
            terms["support_tier"] = action.support_tier
        if action.onboarding_days is not None:
            terms["onboarding_days"] = action.onboarding_days
        if action.service_credits_percent is not None:
            terms["service_credits_percent"] = action.service_credits_percent
        return terms

    @staticmethod
    def _changed_terms(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
        return [term for term, value in after.items() if before.get(term) != value]

    def _process_offer(
        self,
        s: _InternalState,
        proposed: Dict[str, Any],
        prev_company_util: float,
        *,
        offer_kind: str,
        require_multi_term: bool = False,
    ) -> tuple[float, bool, str]:
        reward = 0.0
        done = False
        changed_terms = self._changed_terms(s.current_terms, proposed)
        if not changed_terms:
            return -0.03, False, "Offer did not change any deal terms."
        if require_multi_term and len(changed_terms) < 2:
            return -0.05, False, "This offer should move at least two terms."

        violations = self._engine.check_constraint_violations(
            proposed, s.company_constraints
        )
        if violations:
            s.constraint_violations += len(violations)
            return -0.2, False, f"Offer violates constraints: {'; '.join(violations)}"

        previous_terms = dict(s.current_terms)
        s.terms_touched.extend(changed_terms)
        s.current_terms = proposed
        s.offer_history.append(
            {"actor": "agent", "kind": offer_kind, "terms": dict(proposed)}
        )

        new_company_util = self._engine.compute_company_utility(
            proposed, s.company_targets
        )
        delta = new_company_util - prev_company_util
        reward += delta * 0.5

        if self._engine.should_reject(proposed):
            s.deal_rejected = True
            done = True
            reward += -0.3
            self._last_cp_message = self._scenario["responses"].get(
                "reject", "We cannot proceed."
            )
            self._last_signals = []
            return reward, done, "Counterparty rejected the proposal."

        if self._engine.should_accept(proposed):
            msg, sigs = self._engine.generate_response(
                proposed, s.round_index, s.max_rounds
            )
            self._last_cp_message = msg
            self._last_signals = sigs
            s.counterparty_last_offer = dict(proposed)
            s.offer_history.append(
                {"actor": "counterparty", "kind": "ready_to_sign", "terms": dict(proposed)}
            )
            return reward, done, "Proposal submitted. Counterparty is favorable."

        # Store counteroffer separately -- do NOT overwrite agent's current_terms
        counter = self._engine.generate_counteroffer(proposed)
        s.counterparty_last_offer = dict(counter)
        s.offer_history.append(
            {"actor": "counterparty", "kind": "counter_offer", "terms": dict(counter)}
        )
        msg, sigs = self._engine.generate_response(
            counter, s.round_index, s.max_rounds
        )
        self._last_cp_message = msg + " Here's our counteroffer."
        self._last_signals = sigs
        return reward, done, "Counterparty made a counteroffer. Review counterparty_last_offer."

    def _attempt_close(
        self,
        s: _InternalState,
        terms: Dict[str, Any],
        *,
        accept_counterparty: bool,
    ) -> tuple[float, bool, str]:
        reward = 0.0
        done = False
        company_util = self._engine.compute_company_utility(
            terms, s.company_targets
        )
        violations = self._engine.check_constraint_violations(
            terms, s.company_constraints
        )
        reservation = s.reservation_value if s.reservation_value is not None else s.batna_value
        if violations:
            return -0.2, False, f"Cannot close: terms violate constraints: {'; '.join(violations)}"

        if company_util < reservation:
            s.reservation_breached = True
            reward -= 0.1  # penalty for closing below reservation, but allow it

        if accept_counterparty:
            s.current_terms = dict(terms)
            s.accepted = True
            done = True
            reward += 0.05
            self._last_cp_message = "Accepted. We'll proceed on the latest counterparty offer."
            self._last_signals = ["You accepted the counterparty's last offer."]
        elif self._engine.should_accept(terms):
            s.accepted = True
            done = True
            reward += 0.05
            self._last_cp_message = self._scenario["responses"].get(
                "accept", "We're happy to proceed."
            )
            self._last_signals = ["Counterparty accepted the final deal."]
        elif self._engine.should_reject(terms):
            s.deal_rejected = True
            done = True
            reward += -0.25
            self._last_cp_message = self._scenario["responses"].get(
                "reject", "We cannot proceed under these terms."
            )
            self._last_signals = []
            return reward, done, "Counterparty rejected the close attempt."
        else:
            self._last_cp_message = "We're not ready to sign yet. We need better terms."
            self._last_signals = ["Counterparty is still evaluating the offer."]
            return -0.08, False, "Close attempt failed. Counterparty did not accept the current terms."

        if company_util > 0.6:
            reward += 0.25
            feedback = "Deal accepted by both sides. Strong outcome."
        elif company_util > 0.3:
            reward += 0.1
            feedback = "Deal accepted by both sides. Reasonable outcome."
        else:
            reward += -0.05
            feedback = "Deal accepted by both sides, but economics are weak."
        return reward, done, feedback

    def _compute_batna_value(self, initial_company_util: float) -> float:
        return round(max(0.18, min(0.75, initial_company_util + 0.03)), 4)

    def _infer_priorities(self, limit: int = 2) -> List[str]:
        ranked = sorted(
            self._scenario["cp_utility_weights"].items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        return [term for term, _ in ranked[:limit]]

    def _batna_summary(self, s: _InternalState) -> str:
        return (
            f"Your BATNA is a fallback deal worth approximately {s.batna_value:.2f} utility. "
            "Do not close below that threshold."
        )

    def _negotiation_phase(self, s: _InternalState) -> str:
        if s.round_index <= 1:
            return "discovery"
        if s.round_index < max(2, s.max_rounds - 2):
            return "bargaining"
        return "closing"

    def _engagement_score(self, s: _InternalState) -> float:
        negotiation_actions = sum(
            1
            for action in s.action_log
            if action in {
                "propose_terms",
                "counter_offer",
                "concede_term",
                "bundle_offer",
                "make_package_offer",
                "make_meso_offer",
                "request_tradeoff",
                "final_offer",
                "accept_counterparty_offer",
            }
        )
        distinct_terms = len(set(s.terms_touched))
        analyzed = "analyze_deal" in s.action_log
        clarified = s.clarifications_asked > 0

        if s.task_id == "deal_qualification":
            score = 0.0
            score += 0.35 if analyzed else 0.0
            score += 0.45 if negotiation_actions >= 1 else 0.0
            score += 0.20 if (clarified or s.accepted or s.walked_away) else 0.0
            return max(0.15, min(score, 1.0))

        if s.task_id == "multi_term_negotiation":
            score = 0.0
            score += 0.20 if analyzed else 0.0
            score += 0.40 if negotiation_actions >= 2 else (0.15 if negotiation_actions == 1 else 0.0)
            score += 0.25 if distinct_terms >= 2 else (0.10 if distinct_terms == 1 else 0.0)
            score += 0.15 if (s.accepted or s.walked_away) else 0.0
            score += 0.10 if (s.successful_tradeoffs > 0 or s.successful_packages > 0) else 0.0
            return max(0.10, min(score, 1.0))

        score = 0.0
        score += 0.15 if analyzed else 0.0
        score += 0.20 if clarified else 0.0
        score += 0.35 if negotiation_actions >= 2 else (0.10 if negotiation_actions == 1 else 0.0)
        score += 0.15 if distinct_terms >= 3 else (0.05 if distinct_terms >= 1 else 0.0)
        score += 0.15 if (s.accepted or s.walked_away or s.deal_rejected) else 0.0
        return max(0.05, min(score, 1.0))

    def _reservation_discipline_score(
        self,
        s: _InternalState,
        company_util: float,
    ) -> float:
        reservation = s.reservation_value if s.reservation_value is not None else s.batna_value
        if not s.reservation_value_set:
            base = 0.95  # don't penalize tasks that don't prompt for reservation
        elif reservation < s.batna_value:
            base = 0.55
        else:
            base = 1.0

        if s.accepted and company_util < reservation:
            return 0.0
        if s.walked_away and company_util < reservation:
            return min(1.0, base + 0.05)
        if s.walked_away and company_util > reservation + 0.1:
            return max(0.2, base - 0.35)
        if s.reservation_breached:
            return max(0.2, base - 0.4)
        return base

    def _outcome_quality(
        self,
        s: _InternalState,
        company_util: float,
        violations: List[str],
    ) -> float:
        counterparty_accepts = self._engine.should_accept(s.current_terms)
        economically_bad = company_util < 0.3 or bool(violations)

        if s.accepted:
            return 1.0 if counterparty_accepts and not violations else 0.0

        if s.walked_away:
            if s.round_index <= 1 and not s.deal_rejected:
                return 0.35  # too early walk-away, but not terrible
            if economically_bad or s.deal_rejected or not counterparty_accepts:
                return 0.92  # smart walk-away from bad deal
            return 0.4  # walked away from closeable deal

        if s.deal_rejected:
            return 0.5  # counterparty rejected, but agent tried

        return 0.4  # timed out
