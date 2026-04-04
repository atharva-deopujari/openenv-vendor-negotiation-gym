"""
Baseline inference agent for the Vendor Negotiation Gym.                                                                              
                                                          
Two-stage architecture:
 1. LLM Planner: generates a high-level negotiation plan once per episode                                                              
    (priority terms, tradeoff pairs, close threshold).                   
 2. Deterministic Controller: executes round-by-round actions guided by the                                                            
    plan, using strategic actions (package offers, tradeoffs, MESO offers)                                                           
    to maximize the environment's grading rubric.
"""
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from client import NegotiationEnv
from models import NegotiationAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
EPISODES_PER_TASK = int(os.getenv("EPISODES_PER_TASK", "5"))
INFERENCE_TEMPERATURE = float(os.getenv("INFERENCE_TEMPERATURE", "0"))

TASKS = [
    "deal_qualification",
    "multi_term_negotiation",
    "strategic_contract_close",
]

PLANNER_PROMPT = """You are planning a negotiation strategy for an enterprise procurement agent.

Given the scenario, produce a compact JSON object with:
- priority_terms: an ordered list of 2-4 terms to focus on
- tradeoff_request: the best term to request
- tradeoff_offer: the best term to offer in exchange
- ask_clarification: true or false
- close_threshold: a float from 0.0 to 1.0 for how strong the deal should be before attempting to close

Rules:
- prioritize realistic procurement logic
- prefer payment terms, price, onboarding, SLA, and contract duration over cosmetic terms
- offer something the counterparty may plausibly value, typically contract_months or implementation_fee
- never suggest violating hard constraints

Respond ONLY with a single JSON object."""

TERM_PRIORITY = [
    "price",
    "payment_terms_days",
    "contract_months",
    "sla_uptime",
    "onboarding_days",
    "service_credits_percent",
    "support_tier",
]

SUPPORT_TIER_ORDER = {
    "basic": 0,
    "standard": 1,
    "premium": 2,
    "enterprise": 3,
}

COUNTERPARTY_FRIENDLY_DIRECTION = {
    "price": 1,
    "contract_months": 1,
    "payment_terms_days": -1,
    "sla_uptime": -1,
    "support_tier": -1,
    "onboarding_days": 1,
    "service_credits_percent": -1,
}


class LLMBackend:
    """Text-generation backend interface."""

    provider_name: str = "unknown"

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        history: List[Dict[str, str]] | None = None,
        max_tokens: int = 400,
    ) -> str:
        raise NotImplementedError


class OpenAICompatibleBackend(LLMBackend):
    provider_name = "openai_compatible"

    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        history: List[Dict[str, str]] | None = None,
        max_tokens: int = 400,
    ) -> str:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        # Newer models (gpt-5+, o-series) require max_completion_tokens
        # and some don't support temperature != 1
        is_new_model = any(
            self.model_name.startswith(p) for p in ("gpt-5", "o1", "o3", "o4")
        )
        token_param = "max_completion_tokens" if is_new_model else "max_tokens"
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            token_param: max_tokens,
        }
        # Only set temperature if model supports it
        if not is_new_model or INFERENCE_TEMPERATURE == 1.0:
            kwargs["temperature"] = INFERENCE_TEMPERATURE
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()


def create_llm_backend() -> LLMBackend:
    """Build the OpenAI-compatible backend from env vars."""
    if not HF_TOKEN:
        raise RuntimeError(
            "Missing API credentials. Set HF_TOKEN or OPENAI_API_KEY before running inference.py."
        )
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    return OpenAICompatibleBackend(client=client, model_name=MODEL_NAME)


def _extract_json_object(content: str) -> str:
    """Pull a JSON object out of an LLM response, stripping code fences if present."""
    text = content.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{") and cleaned.endswith("}"):
                return cleaned

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_comparable(term: str, value: Any) -> Any:
    if term == "support_tier":
        return SUPPORT_TIER_ORDER.get(str(value), 1)
    return value


def _interpolate_term(term: str, current: Any, target: Any, fraction: float) -> Any:
    if term == "support_tier":
        cur = _to_comparable(term, current)
        tgt = _to_comparable(term, target)
        mixed = round(cur + (tgt - cur) * fraction)
        for tier, idx in SUPPORT_TIER_ORDER.items():
            if idx == mixed:
                return tier
        return current

    if _is_numeric(current) and _is_numeric(target):
        value = current + (target - current) * fraction
        if isinstance(current, int) and isinstance(target, int):
            return int(round(value))
        return round(float(value), 2)

    return target


def _move_toward_bound(
    term: str,
    current: Any,
    constraints: Dict[str, Any],
    direction: int,
    fraction: float,
) -> Any:
    if term == "support_tier":
        cur = _to_comparable(term, current)
        min_key = f"{term}_min"
        max_key = f"{term}_max"
        bound = None
        if direction > 0 and max_key in constraints:
            bound = _to_comparable(term, constraints[max_key])
        elif direction < 0 and min_key in constraints:
            bound = _to_comparable(term, constraints[min_key])
        if bound is None:
            bound = max(0, min(3, cur + direction))
        moved = round(cur + (bound - cur) * fraction)
        for tier, idx in SUPPORT_TIER_ORDER.items():
            if idx == moved:
                return tier
        return current

    if not _is_numeric(current):
        return current

    min_key = f"{term}_min"
    max_key = f"{term}_max"
    bound = current
    if direction > 0 and max_key in constraints and _is_numeric(constraints[max_key]):
        bound = constraints[max_key]
    elif direction < 0 and min_key in constraints and _is_numeric(constraints[min_key]):
        bound = constraints[min_key]
    else:
        span = max(abs(float(current)) * 0.1, 1.0)
        bound = float(current) + (span * direction)

    moved = float(current) + (float(bound) - float(current)) * fraction
    if isinstance(current, int):
        return int(round(moved))
    return round(moved, 2)


def _estimate_company_utility(
    current_terms: Dict[str, Any],
    company_targets: Dict[str, Any],
    company_constraints: Dict[str, Any],
) -> float:
    scores: List[float] = []
    for term, target in company_targets.items():
        if term not in current_terms:
            continue
        current = current_terms[term]
        cur = _to_comparable(term, current)
        tgt = _to_comparable(term, target)

        if term == "price":
            price_max = company_constraints.get("price_max")
            if price_max and _is_numeric(cur):
                denom = max(price_max - tgt, 1.0)
                score = (price_max - cur) / denom
            else:
                score = 0.5
        elif term == "payment_terms_days":
            minimum = company_constraints.get("payment_terms_days_min")
            baseline = minimum if minimum is not None else min(cur, tgt)
            denom = max(tgt - baseline, 1.0)
            score = (cur - baseline) / denom
        elif term == "contract_months":
            maximum = company_constraints.get("contract_months_max")
            baseline = maximum if maximum is not None else max(cur, tgt)
            denom = max(baseline - tgt, 1.0)
            score = (baseline - cur) / denom
        elif term == "onboarding_days":
            maximum = company_constraints.get("onboarding_days_max")
            baseline = maximum if maximum is not None else max(cur, tgt)
            denom = max(baseline - tgt, 1.0)
            score = (baseline - cur) / denom
        elif term in {"sla_uptime", "service_credits_percent", "support_tier"}:
            minimum = company_constraints.get(f"{term}_min")
            baseline = minimum if minimum is not None else min(cur, tgt)
            denom = max(tgt - baseline, 1.0)
            score = (cur - baseline) / denom
        else:
            score = 0.5

        scores.append(max(0.0, min(float(score), 1.0)))

    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _violates_constraints(
    terms: Dict[str, Any], company_constraints: Dict[str, Any]
) -> bool:
    for key, limit in company_constraints.items():
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        term, direction = parts
        if term not in terms:
            continue
        value = _to_comparable(term, terms[term])
        bound = _to_comparable(term, limit)
        if direction == "max" and value > bound:
            return True
        if direction == "min" and value < bound:
            return True
    return False


def _recommend_reservation_value(observation: Dict[str, Any], task_id: str) -> float:
    hint = float(observation.get("reservation_value_hint", 0.5) or 0.5)
    if task_id == "deal_qualification":
        return round(hint, 4)
    if task_id == "multi_term_negotiation":
        return round(min(0.95, hint + 0.01), 4)
    return round(min(0.95, hint + 0.02), 4)


def _close_threshold(observation: Dict[str, Any], plan: Dict[str, Any]) -> float:
    reservation_hint = float(observation.get("reservation_value_hint", 0.3) or 0.3)
    return max(float(plan.get("close_threshold", 0.35)), reservation_hint + 0.02)


def _requested_terms(
    observation: Dict[str, Any],
    plan: Dict[str, Any],
) -> List[str]:
    current_terms = observation.get("current_terms", {})
    targets = observation.get("company_targets", {})
    terms: List[str] = []
    for term in (plan.get("priority_terms") or []) + observation.get("must_have_terms", []) + TERM_PRIORITY:
        if term in current_terms and term in targets and term not in terms:
            terms.append(term)
    return terms[:4]


def _concession_terms(observation: Dict[str, Any]) -> List[str]:
    inferred = list(observation.get("inferred_counterparty_priorities", []))
    current_terms = observation.get("current_terms", {})
    ordered = inferred + ["contract_months", "payment_terms_days", "onboarding_days", "price"]
    return [term for term in ordered if term in current_terms]


def _build_package_terms(
    observation: Dict[str, Any],
    plan: Dict[str, Any],
    *,
    request_fraction: float,
    concession_fraction: float,
) -> Dict[str, Any]:
    current_terms = observation.get("current_terms", {})
    targets = observation.get("company_targets", {})
    constraints = observation.get("company_constraints", {})
    package_terms: Dict[str, Any] = {}

    requested = _requested_terms(observation, plan)
    for term in requested[:2]:
        package_terms[term] = _interpolate_term(
            term,
            current_terms[term],
            targets[term],
            request_fraction,
        )

    for term in _concession_terms(observation):
        if term in package_terms:
            continue
        direction = COUNTERPARTY_FRIENDLY_DIRECTION.get(term, 1)
        package_terms[term] = _move_toward_bound(
            term,
            current_terms[term],
            constraints,
            direction,
            concession_fraction,
        )
        break

    if _violates_constraints({**current_terms, **package_terms}, constraints):
        return {}
    return package_terms


def _build_meso_offers(
    observation: Dict[str, Any],
    plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    variants = []
    for request_fraction, concession_fraction in [(0.28, 0.22), (0.34, 0.28), (0.22, 0.35)]:
        package_terms = _build_package_terms(
            observation,
            plan,
            request_fraction=request_fraction,
            concession_fraction=concession_fraction,
        )
        if package_terms:
            variants.append(package_terms)
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for offer in variants:
        key = json.dumps(offer, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(offer)
    return deduped[:3]


def _derive_focus_terms(observation: Dict[str, Any]) -> List[str]:
    current_terms = observation.get("current_terms", {})
    company_targets = observation.get("company_targets", {})
    focus = list(observation.get("must_have_terms", []))
    gaps = []
    for term in TERM_PRIORITY:
        if term not in current_terms or term not in company_targets:
            continue
        cur = _to_comparable(term, current_terms[term])
        tgt = _to_comparable(term, company_targets[term])
        if _is_numeric(cur) and _is_numeric(tgt):
            gaps.append((abs(cur - tgt), term))
    focus.extend(term for _, term in sorted(gaps, reverse=True)[:2])
    seen = []
    for term in focus:
        if term and term not in seen:
            seen.append(term)
    return seen[:4]


def _build_proposal(
    observation: Dict[str, Any],
    fraction: float,
    include_nice_to_have: bool = True,
) -> Dict[str, Any]:
    current_terms = dict(observation.get("current_terms", {}))
    targets = observation.get("company_targets", {})
    must_have = observation.get("must_have_terms", [])
    nice_to_have = observation.get("nice_to_have_terms", []) if include_nice_to_have else []

    proposal: Dict[str, Any] = {"action_type": "propose_terms"}
    chosen_terms: List[str] = []
    for term in must_have + nice_to_have:
        if term not in current_terms or term not in targets:
            continue
        proposal[term] = _interpolate_term(
            term, current_terms[term], targets[term], fraction
        )
        chosen_terms.append(term)
        if len(chosen_terms) >= 4:
            break
    return proposal


def _close_signal_present(message: str, signals: List[str]) -> bool:
    text = f"{message} {' '.join(signals)}".lower()
    markers = [
        "finalize",
        "move forward",
        "close to agreement",
        "let's finalize",
        "prepare the contract",
        "prepare the order form",
        "happy to move forward",
        "agreed",
    ]
    return any(marker in text for marker in markers)


def get_episode_plan(
    llm: LLMBackend,
    observation: Dict[str, Any],
    task_id: str,
) -> Dict[str, Any]:
    """Ask the LLM for a per-episode plan (priority terms, tradeoff pair, close threshold)."""
    obs_text = format_observation(observation, task_id)
    content = llm.generate_text(
        system_prompt=PLANNER_PROMPT,
        user_prompt=obs_text,
        max_tokens=300,
    )
    try:
        plan = json.loads(_extract_json_object(content))
        return {
            "priority_terms": list(plan.get("priority_terms", []))[:4],
            "tradeoff_request": plan.get("tradeoff_request"),
            "tradeoff_offer": plan.get("tradeoff_offer"),
            "ask_clarification": bool(plan.get("ask_clarification", False)),
            "close_threshold": float(plan.get("close_threshold", 0.35)),
        }
    except Exception:
        return {
            "priority_terms": _derive_focus_terms(observation),
            "tradeoff_request": "payment_terms_days",
            "tradeoff_offer": "contract_months",
            "ask_clarification": task_id != "deal_qualification",
            "close_threshold": 0.35,
        }


def _best_tradeoff_pair(observation: Dict[str, Any], plan: Dict[str, Any]) -> tuple:
    """Pick a (requested, offered) term pair for a tradeoff.

    Request a term we care about (large gap to target) and offer a term
    the CP likely values based on inferred priorities.
    """
    inferred_cp = list(observation.get("inferred_counterparty_priorities", []))
    current_terms = observation.get("current_terms", {})
    targets = observation.get("company_targets", {})
    nice_to_have = observation.get("nice_to_have_terms", [])

    # Best term to request: biggest gap in must-have terms
    request_candidates = []
    for term in TERM_PRIORITY:
        if term not in current_terms or term not in targets:
            continue
        cur = _to_comparable(term, current_terms[term])
        tgt = _to_comparable(term, targets[term])
        if _is_numeric(cur) and _is_numeric(tgt):
            gap = abs(tgt - cur) / max(abs(tgt), 1.0)
            request_candidates.append((gap, term))
    request_candidates.sort(reverse=True)

    # Best term to offer: counterparty priorities that are nice-to-have for us
    offer_candidates = inferred_cp + nice_to_have + ["contract_months", "implementation_fee", "onboarding_days"]
    offer_candidates = [t for t in offer_candidates if t in current_terms]

    requested = plan.get("tradeoff_request") or (request_candidates[0][1] if request_candidates else "price")
    offered = plan.get("tradeoff_offer") or "contract_months"

    # Prefer offering a term from CP priorities that differs from requested
    for t in offer_candidates:
        if t != requested:
            offered = t
            break

    return requested, offered


def choose_action_with_plan(
    observation: Dict[str, Any],
    task_id: str,
    plan: Dict[str, Any],
    step: int,
) -> NegotiationAction:
    """Per-step action selection, driven by the LLM plan.

    Progression per task:
      deal_qualification:       analyze -> propose -> propose/accept -> close/walk
      multi_term_negotiation:   analyze -> reservation+clarify -> package -> tradeoff -> meso -> close/walk
      strategic_contract_close: analyze -> reservation -> clarify -> tradeoff -> bundle -> package -> meso -> close/walk
    """
    available_actions = set(observation.get("available_actions", []))
    current_terms = observation.get("current_terms", {})
    company_targets = observation.get("company_targets", {})
    company_constraints = observation.get("company_constraints", {})
    round_index = int(observation.get("round_index", 0))
    max_rounds = int(observation.get("max_rounds", 1))
    message = observation.get("counterparty_message", "")
    signals = observation.get("signals", [])
    feedback = observation.get("last_action_feedback", "")
    must_have = observation.get("must_have_terms", [])
    nice_to_have = observation.get("nice_to_have_terms", [])
    counterparty_last_offer = observation.get("counterparty_last_offer")

    estimated_utility = _estimate_company_utility(
        current_terms, company_targets, company_constraints
    )
    close_signal = _close_signal_present(message, signals)
    close_threshold = _close_threshold(observation, plan)
    compliant = not _violates_constraints(current_terms, company_constraints)

    # Always analyze on the first step
    if step == 0 and "analyze_deal" in available_actions:
        focus_terms = plan.get("priority_terms") or _derive_focus_terms(observation)
        return NegotiationAction(
            action_type="analyze_deal",
            focus_terms=focus_terms[:4],
            risk_notes="Protect hard constraints while preserving room to trade on lower-priority terms.",
        )

    # Accept counterparty offer only if it clears reservation with a safety margin.
    # Accepting below reservation collapses the reservation discipline multiplier.
    reservation_hint = float(observation.get("reservation_value_hint", 0.3) or 0.3)
    min_accept_util = max(close_threshold + 0.10, reservation_hint + 0.15, 0.55)
    if counterparty_last_offer and "accept_counterparty_offer" in available_actions:
        if not _violates_constraints(counterparty_last_offer, company_constraints):
            cp_offer_util = _estimate_company_utility(
                counterparty_last_offer, company_targets, company_constraints
            )
            # Also check the offer is at least as good as current terms
            if cp_offer_util >= min_accept_util and cp_offer_util >= estimated_utility * 0.90:
                return NegotiationAction(action_type="accept_counterparty_offer")

    # Dispatch to task-specific strategy

    if task_id == "deal_qualification":
        return _deal_qualification_strategy(
            observation, plan, step, round_index, max_rounds,
            available_actions, estimated_utility, close_threshold,
            close_signal, compliant, feedback,
        )

    if task_id == "multi_term_negotiation":
        return _multi_term_strategy(
            observation, plan, step, round_index, max_rounds,
            available_actions, estimated_utility, close_threshold,
            close_signal, compliant, feedback,
        )

    return _strategic_close_strategy(
        observation, plan, step, round_index, max_rounds,
        available_actions, estimated_utility, close_threshold,
        close_signal, compliant, feedback,
    )


def _deal_qualification_strategy(
    observation, plan, step, round_index, max_rounds,
    available_actions, estimated_utility, close_threshold,
    close_signal, compliant, feedback,
) -> NegotiationAction:
    # 4 rounds: quick analysis, then decisive close or walk.
    current_terms = observation.get("current_terms", {})
    company_targets = observation.get("company_targets", {})
    company_constraints = observation.get("company_constraints", {})

    # Aggressive opening proposal
    if round_index <= 1 and "propose_terms" in available_actions:
        proposal = {"action_type": "propose_terms"}
        chosen = []
        for term in (plan.get("priority_terms") or []) + observation.get("must_have_terms", []) + TERM_PRIORITY:
            if term not in current_terms or term not in company_targets or term in chosen:
                continue
            proposal[term] = _interpolate_term(term, current_terms[term], company_targets[term], 0.55)
            chosen.append(term)
            if len(chosen) >= 3:
                break
        return NegotiationAction(**proposal)

    # Moderate follow-up proposal
    if round_index == 2 and "propose_terms" in available_actions:
        proposal = {"action_type": "propose_terms"}
        chosen = []
        for term in (plan.get("priority_terms") or []) + observation.get("must_have_terms", []) + TERM_PRIORITY:
            if term not in current_terms or term not in company_targets or term in chosen:
                continue
            proposal[term] = _interpolate_term(term, current_terms[term], company_targets[term], 0.35)
            chosen.append(term)
            if len(chosen) >= 3:
                break
        return NegotiationAction(**proposal)

    # Close only when counterparty signals readiness (avoids rejection penalty)
    if close_signal and compliant and estimated_utility >= close_threshold and "final_offer" in available_actions:
        return NegotiationAction(action_type="final_offer")

    # Disciplined walk-away from weak deals
    if estimated_utility < 0.35 and "walk_away" in available_actions:
        return NegotiationAction(
            action_type="walk_away",
            reason="Deal does not meet minimum qualification threshold.",
        )

    # Preserve BATNA if no close signal materialized
    if "walk_away" in available_actions and not close_signal:
        return NegotiationAction(
            action_type="walk_away",
            reason="No clear close signal from counterparty. Walking away to preserve BATNA.",
        )

    if "propose_terms" in available_actions:
        return NegotiationAction(**_build_proposal(observation, 0.30))

    return NegotiationAction(action_type="walk_away", reason="No viable path to close.")


def _multi_term_strategy(
    observation, plan, step, round_index, max_rounds,
    available_actions, estimated_utility, close_threshold,
    close_signal, compliant, feedback,
) -> NegotiationAction:
    # 6 rounds: reservation -> clarify -> package -> tradeoff -> meso -> close/walk.
    current_terms = observation.get("current_terms", {})
    company_targets = observation.get("company_targets", {})
    company_constraints = observation.get("company_constraints", {})

    # Set reservation value early to anchor walk-away discipline
    if step == 1 and "set_reservation_value" in available_actions:
        return NegotiationAction(
            action_type="set_reservation_value",
            reservation_value=_recommend_reservation_value(observation, task_id="multi_term_negotiation"),
        )

    # Probe counterparty priorities via clarification
    if "ask_clarification" in available_actions and round_index <= 2 and step <= 2:
        return NegotiationAction(
            action_type="ask_clarification",
            question="Which commercial term matters most for you if we want to move toward agreement quickly?",
        )

    # Multi-term package offer (linked requests + concessions)
    if round_index <= 3 and "make_package_offer" in available_actions:
        package_terms = _build_package_terms(
            observation, plan,
            request_fraction=0.40,
            concession_fraction=0.25,
        )
        if len(package_terms) >= 2:
            return NegotiationAction(
                action_type="make_package_offer",
                package_terms=package_terms,
            )

    # Explicit tradeoff request after package anchoring
    if (
        round_index <= 4
        and "request_tradeoff" in available_actions
        and "tradeoff" not in feedback.lower()
        and "failed" not in feedback.lower()
    ):
        requested, offered = _best_tradeoff_pair(observation, plan)
        if requested in current_terms and offered in current_terms and requested != offered:
            return NegotiationAction(
                action_type="request_tradeoff",
                requested_term=requested,
                offered_term=offered,
            )

    # MESO offer to probe counterparty preferences via multiple equivalent packages
    if round_index >= 3 and "make_meso_offer" in available_actions:
        meso_offers = _build_meso_offers(observation, plan)
        if len(meso_offers) >= 2:
            return NegotiationAction(
                action_type="make_meso_offer",
                meso_offers=meso_offers,
            )

    # Close phase -- only attempt final_offer when counterparty signals readiness
    if round_index >= max_rounds - 2:
        if close_signal and compliant and estimated_utility >= close_threshold and "final_offer" in available_actions:
            return NegotiationAction(action_type="final_offer")
        if estimated_utility < 0.35 and "walk_away" in available_actions:
            return NegotiationAction(
                action_type="walk_away",
                reason="Multi-term structure remains below acceptable threshold after exhausting strategic options.",
            )

    # Keep engaging with proposals if no strategic action applies
    if "propose_terms" in available_actions:
        fraction = 0.40 if round_index < max_rounds // 2 else 0.25
        return NegotiationAction(**_build_proposal(observation, fraction))

    if "walk_away" in available_actions:
        reason = (
            "No viable negotiation path remains."
            if estimated_utility < 0.45
            else "Cannot close despite reasonable terms; preserving BATNA."
        )
        return NegotiationAction(action_type="walk_away", reason=reason)
    return NegotiationAction(**_build_proposal(observation, 0.30))


def _strategic_close_strategy(
    observation, plan, step, round_index, max_rounds,
    available_actions, estimated_utility, close_threshold,
    close_signal, compliant, feedback,
) -> NegotiationAction:
    # 8 rounds: full strategic toolkit (tradeoff + bundle + package + meso).
    current_terms = observation.get("current_terms", {})
    company_targets = observation.get("company_targets", {})
    company_constraints = observation.get("company_constraints", {})

    # Anchor discipline with an explicit reservation value
    if step <= 1 and "set_reservation_value" in available_actions:
        return NegotiationAction(
            action_type="set_reservation_value",
            reservation_value=_recommend_reservation_value(observation, task_id="strategic_contract_close"),
        )

    # Elicit hidden counterparty priorities via clarification
    if "ask_clarification" in available_actions and round_index <= 2 and step <= 2:
        return NegotiationAction(
            action_type="ask_clarification",
            question="If we tighten one commercial term for you, which term would matter most for getting this signed?",
        )

    # Explicit tradeoff request -- gives CP a credible exchange
    if (
        round_index <= 3
        and "request_tradeoff" in available_actions
        and "not credible" not in feedback.lower()
        and "failed" not in feedback.lower()
    ):
        requested, offered = _best_tradeoff_pair(observation, plan)
        if requested in current_terms and offered in current_terms and requested != offered:
            return NegotiationAction(
                action_type="request_tradeoff",
                requested_term=requested,
                offered_term=offered,
            )

    # One structured bundle offer (give CP-priority term, get company-priority term)
    if round_index <= 5 and "bundle_offer" in available_actions and "Bundle" not in feedback:
        inferred_cp = list(observation.get("inferred_counterparty_priorities", []))
        give_term = None
        for t in inferred_cp + ["contract_months", "onboarding_days"]:
            if t in current_terms and t in COUNTERPARTY_FRIENDLY_DIRECTION:
                give_term = t
                break
        get_term = None
        for t in (plan.get("priority_terms") or []) + ["price", "payment_terms_days"]:
            if t in current_terms and t != give_term:
                get_term = t
                break
        if give_term and get_term:
            give_val = _move_toward_bound(
                give_term, current_terms[give_term], company_constraints,
                COUNTERPARTY_FRIENDLY_DIRECTION.get(give_term, 1), 0.20,
            )
            get_val = _interpolate_term(get_term, current_terms[get_term], company_targets.get(get_term, current_terms[get_term]), 0.30)
            test_terms = dict(current_terms)
            test_terms[give_term] = give_val
            test_terms[get_term] = get_val
            if not _violates_constraints(test_terms, company_constraints):
                return NegotiationAction(
                    action_type="bundle_offer",
                    give_term=give_term,
                    give_value=give_val,
                    get_term=get_term,
                    get_value=get_val,
                )

    # Multi-term package offer
    if round_index <= 6 and "make_package_offer" in available_actions:
        package_terms = _build_package_terms(
            observation, plan,
            request_fraction=0.35,
            concession_fraction=0.20,
        )
        if len(package_terms) >= 2:
            return NegotiationAction(
                action_type="make_package_offer",
                package_terms=package_terms,
            )

    # MESO offer to probe counterparty preferences
    if round_index >= 4 and "make_meso_offer" in available_actions:
        meso_offers = _build_meso_offers(observation, plan)
        if len(meso_offers) >= 2:
            return NegotiationAction(
                action_type="make_meso_offer",
                meso_offers=meso_offers,
            )

    # Continue tightening terms in the mid-game
    if round_index < max_rounds - 2 and "propose_terms" in available_actions:
        fraction = 0.35 if round_index < 5 else 0.20
        return NegotiationAction(**_build_proposal(observation, fraction))

    # Close phase: final_offer only with a clear close signal
    if round_index >= max_rounds - 2:
        if close_signal and compliant and estimated_utility >= close_threshold and "final_offer" in available_actions:
            if "failed" not in feedback.lower():
                return NegotiationAction(action_type="final_offer")
        if estimated_utility < 0.35 and "walk_away" in available_actions:
            return NegotiationAction(
                action_type="walk_away",
                reason="Deal economics remain below threshold after exhausting all strategic options.",
            )

    if "propose_terms" in available_actions:
        return NegotiationAction(**_build_proposal(observation, 0.25))

    if "walk_away" in available_actions:
        return NegotiationAction(action_type="walk_away", reason="No viable path to close.")
    return NegotiationAction(**_build_proposal(observation, 0.20))


def format_observation(observation: dict, task_id: str) -> str:
    """Render an observation as a prompt for the planner."""
    parts = [
        f"Scenario: {observation.get('scenario_id', 'N/A')}",
        f"Counterparty type: {observation.get('counterparty_type', 'N/A')}",
        f"Counterparty message: {observation.get('counterparty_message', '')}",
    ]

    signals = observation.get("signals", [])
    if signals:
        parts.append(f"Signals: {', '.join(signals)}")

    current_terms = observation.get("current_terms", {})
    if current_terms:
        terms_str = json.dumps(current_terms, indent=2)
        parts.append(f"Current terms:\n{terms_str}")

    company_targets = observation.get("company_targets", {})
    if company_targets:
        targets_str = json.dumps(company_targets, indent=2)
        parts.append(f"Company targets:\n{targets_str}")

    company_constraints = observation.get("company_constraints", {})
    if company_constraints:
        constraints_str = json.dumps(company_constraints, indent=2)
        parts.append(f"Company constraints:\n{constraints_str}")

    must_have = observation.get("must_have_terms", [])
    if must_have:
        parts.append(f"Must-have terms: {', '.join(must_have)}")

    nice_to_have = observation.get("nice_to_have_terms", [])
    if nice_to_have:
        parts.append(f"Nice-to-have terms: {', '.join(nice_to_have)}")

    inferred = observation.get("inferred_counterparty_priorities", [])
    if inferred:
        parts.append(f"Inferred counterparty priorities: {', '.join(inferred)}")

    batna_summary = observation.get("batna_summary", "")
    if batna_summary:
        parts.append(f"BATNA: {batna_summary}")

    reservation_value_hint = observation.get("reservation_value_hint")
    if reservation_value_hint is not None:
        parts.append(f"Reservation value hint: {reservation_value_hint}")

    rounds_remaining = observation.get("rounds_remaining")
    if rounds_remaining is not None:
        parts.append(f"Rounds remaining: {rounds_remaining}")

    available_actions = observation.get("available_actions", [])
    if available_actions:
        parts.append(f"Available actions: {', '.join(available_actions)}")
        parts.append(
            "IMPORTANT: Pick your next action from the available_actions list above."
        )

    round_index = observation.get("round_index", 0)
    max_rounds = observation.get("max_rounds", 5)
    parts.append(f"Round: {round_index}/{max_rounds}")

    status = observation.get("negotiation_status", "in_progress")
    parts.append(f"Negotiation status: {status}")

    feedback = observation.get("last_action_feedback", "")
    if feedback:
        parts.append(f"Feedback: {feedback}")

    parts.append(f"Task: {task_id}")

    return "\n".join(parts)


BENCHMARK_NAME = "vendor_negotiation_gym"
SUCCESS_SCORE_THRESHOLD = 0.30  # score ≥ threshold → success=true


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(bool(done)).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _format_action_for_log(action: NegotiationAction) -> str:
    # Build a single-line action repr for the [STEP] log.
    data = action.model_dump(exclude_none=True, exclude_defaults=True)
    action_type = data.pop("action_type", "unknown")
    # Drop empty metadata dict which adds noise without signal
    if data.get("metadata") == {}:
        data.pop("metadata", None)
    if not data:
        return f"{action_type}()"
    args = ",".join(f"{k}={json.dumps(v, separators=(',', ':'))}" for k, v in data.items())
    return f"{action_type}({args})"


def _run_episode(
    env: NegotiationEnv,
    llm: LLMBackend,
    task_id: str,
    episode_seed: int,
    model_name: str,
) -> float:
    """Play one episode end-to-end and emit the required stdout logs."""
    log_start(task=task_id, env_name=BENCHMARK_NAME, model=model_name)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = env.reset(seed=episode_seed, task_id=task_id)
        plan = get_episode_plan(llm, result.observation.model_dump(), task_id)
        step_index = 0

        while not result.done:
            obs_dict = result.observation.model_dump()
            action = choose_action_with_plan(obs_dict, task_id, plan, step_index)
            result = env.step(action)
            step_index += 1
            steps_taken = step_index

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            action_str = _format_action_for_log(action)
            error = getattr(result.observation, "last_action_error", None) if result.observation else None
            log_step(
                step=step_index,
                action=action_str,
                reward=reward,
                done=bool(result.done),
                error=error,
            )

        state = env.state()
        score = float(state.grade_score) if state.grade_score is not None else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    except Exception as exc:
        print(f"[DEBUG] Episode failed: {exc}", flush=True)
        return score
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def run_inference() -> Dict[str, Dict[str, Any]]:
    """Main entry point: run EPISODES_PER_TASK episodes on each task."""
    llm = create_llm_backend()
    env = NegotiationEnv(base_url=ENV_URL).sync()
    all_results: Dict[str, Dict[str, Any]] = {}

    try:
        with env:
            for task_id in TASKS:
                scores: List[float] = []
                for ep in range(EPISODES_PER_TASK):
                    episode_score = _run_episode(
                        env=env,
                        llm=llm,
                        task_id=task_id,
                        episode_seed=ep,
                        model_name=MODEL_NAME,
                    )
                    scores.append(episode_score)
                avg = sum(scores) / len(scores) if scores else 0.0
                all_results[task_id] = {
                    "average_score": round(avg, 4),
                    "scores": [round(s, 4) for s in scores],
                }
    finally:
        print(f"[SUMMARY] results={json.dumps(all_results)}", flush=True)

    return all_results


if __name__ == "__main__":
    run_inference()
