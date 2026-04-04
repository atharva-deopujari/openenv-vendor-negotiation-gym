"""
Typed Pydantic models for the Vendor Negotiation Gym.
Inherits from OpenEnv base classes: Action, Observation, State.
"""
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field

from openenv.core.env_server import Action, Observation, State


VALID_ACTION_TYPES = Literal[
    "analyze_deal",
    "set_reservation_value",
    "propose_terms",
    "counter_offer",
    "concede_term",
    "ask_clarification",
    "bundle_offer",
    "make_package_offer",
    "make_meso_offer",
    "request_tradeoff",
    "accept_deal",
    "accept_counterparty_offer",
    "final_offer",
    "walk_away",
]

VALID_SUPPORT_TIERS = Literal["basic", "standard", "premium", "enterprise"]

VALID_TERM_NAMES = Literal[
    "price",
    "contract_months",
    "payment_terms_days",
    "sla_uptime",
    "support_tier",
    "onboarding_days",
    "termination_flexibility",
    "service_credits_percent",
    "seat_commitment",
    "implementation_fee",
]


class NegotiationAction(Action):
    """Agent's action in the negotiation environment."""

    action_type: VALID_ACTION_TYPES

    # analyze_deal
    focus_terms: Optional[List[str]] = None
    risk_notes: Optional[str] = None
    reservation_value: Optional[float] = None

    # propose_terms
    price: Optional[float] = None
    contract_months: Optional[int] = None
    payment_terms_days: Optional[int] = None
    sla_uptime: Optional[float] = None
    support_tier: Optional[VALID_SUPPORT_TIERS] = None
    onboarding_days: Optional[int] = None
    service_credits_percent: Optional[float] = None
    message: Optional[str] = None

    # concede_term
    term_name: Optional[str] = None
    new_value: Optional[Union[str, float, int]] = None
    concession_reason: Optional[str] = None

    # ask_clarification
    question: Optional[str] = None

    # bundle_offer
    give_term: Optional[str] = None
    give_value: Optional[Union[str, float, int]] = None
    get_term: Optional[str] = None
    get_value: Optional[Union[str, float, int]] = None
    package_terms: Optional[Dict[str, Union[str, float, int]]] = None
    package_name: Optional[str] = None
    meso_offers: Optional[List[Dict[str, Union[str, float, int]]]] = None

    # request_tradeoff
    requested_term: Optional[str] = None
    offered_term: Optional[str] = None

    # walk_away
    reason: Optional[str] = None


class NegotiationObservation(Observation):
    """What the agent observes after each action."""

    # Scenario metadata
    scenario_id: str = ""
    task_id: str = ""
    round_index: int = 0
    max_rounds: int = 5
    counterparty_type: str = ""

    # Counterparty communication
    counterparty_message: str = ""
    signals: List[str] = Field(default_factory=list)
    inferred_counterparty_priorities: List[str] = Field(default_factory=list)

    # Deal state
    current_terms: Dict[str, Any] = Field(default_factory=dict)
    counterparty_last_offer: Optional[Dict[str, Any]] = None
    offer_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Company objectives
    company_targets: Dict[str, Any] = Field(default_factory=dict)
    company_constraints: Dict[str, Any] = Field(default_factory=dict)
    must_have_terms: List[str] = Field(default_factory=list)
    nice_to_have_terms: List[str] = Field(default_factory=list)
    batna_summary: str = ""
    reservation_value_hint: float = 0.0
    compliance_risks: List[str] = Field(default_factory=list)
    rounds_remaining: int = 0

    # Action guidance
    available_actions: List[str] = Field(default_factory=list)
    last_action_feedback: str = ""

    # Status
    negotiation_status: str = "in_progress"  # in_progress | accepted | rejected | walked_away
    negotiation_phase: str = "discovery"


class NegotiationState(State):
    """Public episode state -- hidden counterparty weights not exposed."""

    task_id: str = ""
    round_index: int = 0
    max_rounds: int = 5
    is_done: bool = False
    action_log: List[str] = Field(default_factory=list)
    current_terms: Dict[str, Any] = Field(default_factory=dict)
    cumulative_reward: float = 0.0
    estimated_deal_quality: float = 0.0
    accepted: bool = False
    walked_away: bool = False
    grade_score: Optional[float] = None
    reservation_value: Optional[float] = None


class _InternalState(NegotiationState):
    """Internal state with hidden counterparty model -- never sent to clients."""

    # Scenario ground truth
    scenario_id: str = ""
    counterparty_type: str = ""

    # Hidden counterparty preferences (weights sum to ~1.0)
    cp_utility_weights: Dict[str, float] = Field(default_factory=dict)
    cp_acceptance_threshold: float = 0.5
    cp_ideal_terms: Dict[str, Any] = Field(default_factory=dict)
    cp_floor_terms: Dict[str, Any] = Field(default_factory=dict)

    # Company objectives (kept for grading)
    company_targets: Dict[str, Any] = Field(default_factory=dict)
    company_constraints: Dict[str, Any] = Field(default_factory=dict)
    must_have_terms: List[str] = Field(default_factory=list)

    # Starting terms
    initial_terms: Dict[str, Any] = Field(default_factory=dict)

    # Tracking
    clarifications_asked: int = 0
    concessions_made: List[str] = Field(default_factory=list)
    terms_touched: List[str] = Field(default_factory=list)
    successful_tradeoffs: int = 0
    close_attempts: int = 0
    successful_packages: int = 0
    successful_mesos: int = 0
    package_offer_attempts: int = 0
    meso_offer_attempts: int = 0
    reservation_value_set: bool = False
    reservation_breached: bool = False
    batna_value: float = 0.0
    counterparty_last_offer: Optional[Dict[str, Any]] = None
    offer_history: List[Dict[str, Any]] = Field(default_factory=list)
    inferred_counterparty_priorities: List[str] = Field(default_factory=list)
    constraint_violations: int = 0
    deal_rejected: bool = False
