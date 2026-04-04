---
title: Vendor Negotiation Gym
emoji: 🤝
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Vendor Negotiation Gym

An OpenEnv environment for multi-dimensional enterprise contract negotiation under partial observability. Agents negotiate vendor contracts across nine deal dimensions against deterministic counterparties whose utility weights are hidden and must be inferred from dialogue.

## Overview

Enterprise procurement is a high-stakes, multi-dimensional decision problem: buyers and vendors negotiate across price, contract length, SLAs, payment terms, support tiers, onboarding timelines, and more, under budget and compliance constraints. This environment models that domain with 52 hand-written scenarios across five counterparty archetypes, a 14-action negotiation interface, and a grading rubric that rewards strategic competence rather than raw model capability.

The environment is OpenEnv v1 spec-compliant and exposes a FastAPI server with `/reset`, `/step`, `/state`, `/schema`, `/metadata`, and `/health` endpoints. It is fully containerized and deployable to HuggingFace Spaces without modification.

## Quick Stats

| | |
|---|---|
| Scenarios | 52 hand-written (SCN-001 … SCN-052) |
| Counterparty types | 5 (software vendor, cloud provider, logistics vendor, marketing agency, enterprise buyer) |
| Actions | 14 (information, proposal, strategic, closure) |
| Deal dimensions | 9 (price, contract months, payment terms, SLA uptime, support tier, onboarding days, service credits, seat commitment, implementation fee) |
| Tasks | 3 with calibrated difficulty gradient |
| Grading | Weighted task components × geometric-mean quality multiplier |
| Partial observability | Counterparty utility weights, acceptance threshold, and ideal terms are hidden |

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │            Agent / LLM               │
                    │  (plans, reasons, selects actions)   │
                    └──────────────┬───────────────────────┘
                                   │ NegotiationAction
                                   ▼
 ┌──────────────────────────────────────────────────────────────┐
 │             FastAPI Server (OpenEnv spec v1)                 │
 │                                                              │
 │   ┌────────────┐   ┌──────────────┐   ┌─────────────────┐    │
 │   │  Scenario  │──▶│ Deal Engine  │──▶│ Hidden CP Model │    │
 │   │  Database  │   │ (utilities,  │   │  (weights,      │    │
 │   │   (52)     │   │ constraints) │   │ thresholds)     │    │
 │   └────────────┘   └──────┬───────┘   └─────────────────┘    │
 │                           │                                  │
 │                           ▼                                  │
 │                  ┌──────────────────┐                        │
 │                  │   Reward Shaper  │  per-step dense reward │
 │                  └─────────┬────────┘                        │
 │                            ▼                                 │
 │                  ┌──────────────────┐                        │
 │                  │ Geometric-Mean   │  final grade ∈ [0, 1]  │
 │                  │     Grader       │                        │
 │                  └──────────────────┘                        │
 └──────────────────────────┬───────────────────────────────────┘
                            │ NegotiationObservation (public fields only)
                            ▼
                    ┌───────────────┐
                    │     Agent     │
                    └───────────────┘
```

## Tasks

| # | Task ID | Difficulty | Max Rounds | Tests |
|---|---|---|---|---|
| 1 | `deal_qualification` | Easy | 4 | Deal reading, constraint discipline, efficient action use |
| 2 | `multi_term_negotiation` | Medium | 6 | Multi-dimensional tradeoffs, concession sequencing |
| 3 | `strategic_contract_close` | Hard | 8 | Inference of hidden CP preferences, close-vs-walk timing, signal exploitation |

Grading weights are task-specific:

| Task | Weights |
|---|---|
| `deal_qualification` | target_alignment 0.45 · constraint_compliance 0.35 · action_efficiency 0.20 |
| `multi_term_negotiation` | deal_quality 0.35 · strategic_concessions 0.25 · constraint_compliance 0.20 · efficiency 0.20 |
| `strategic_contract_close` | final_deal_utility 0.40 · counterparty_signal_exploitation 0.20 · constraint_compliance 0.20 · close_decision_quality 0.10 · efficiency 0.10 |

## Negotiation Concepts

The action space and grading rubric borrow several terms from the academic negotiation literature (Fisher & Ury's *Getting to Yes*, the Harvard Program on Negotiation, Wharton integrative bargaining research). Brief definitions for readers outside the field:

| Term | Meaning |
|---|---|
| **BATNA** | *Best Alternative To a Negotiated Agreement.* Your fallback if the deal fails, e.g. "we already have a competing offer at $80k." Agents should never close below their BATNA. |
| **Reservation value** | The minimum utility the agent is willing to accept. Should be anchored to BATNA; closing below it is a grading failure. |
| **Integrative bargaining** | Creating joint value by trading across terms (e.g. give longer contract, get lower price) rather than haggling on a single number. |
| **Bundle offer** | A structured give/get across two terms: "if you move on X, I'll move on Y." |
| **Package offer** | A multi-term proposal where several terms are linked; the counterparty must accept or reject the package as a whole. |
| **MESO** | *Multiple Equivalent Simultaneous Offers.* Present several packages that are all equally valuable to you. The counterparty picks one, and their choice reveals their hidden priorities, without you having to concede. A standard information-gathering technique. |
| **Tradeoff request** | An explicit ask for a term exchange: "we'll concede on contract length if you tighten payment terms." |
| **Signal exploitation** | Using counterparty messages, offer history, and responses to infer the hidden utility weights the environment never reveals. |

## Action Space

Actions are grouped by role in the negotiation loop.

**Information**

| Action | Purpose |
|---|---|
| `analyze_deal` | Inspect current terms against targets and constraints; first use grants an exploration bonus |
| `ask_clarification` | Probe the counterparty for hidden priorities; answers feed `inferred_counterparty_priorities` |
| `set_reservation_value` | Commit to a minimum utility floor; a disciplined reservation value is a grading input |

**Proposal**

| Action | Purpose |
|---|---|
| `propose_terms` | Submit revised terms on any subset of the 9 dimensions |
| `counter_offer` | Respond to the counterparty's last structured offer with a revision |
| `concede_term` | Move a single term toward the counterparty's position |

**Strategic**

| Action | Purpose |
|---|---|
| `bundle_offer` | Structured give/get across two terms |
| `make_package_offer` | Multi-term linked package |
| `make_meso_offer` | Multiple equivalent package variants; probes hidden CP weights |
| `request_tradeoff` | Ask for movement on one term in exchange for movement on another |

**Closure**

| Action | Purpose |
|---|---|
| `accept_deal` | Close on current terms (checks reservation value and counterparty threshold) |
| `accept_counterparty_offer` | Accept the counterparty's last structured offer |
| `final_offer` | Last-chance close attempt |
| `walk_away` | Exit with an optional reason; credited when counterparty utility was poor |

**Example MESO offer.** The agent presents three packages that are all roughly equivalent from its own utility perspective:

```json
{
  "action_type": "make_meso_offer",
  "package_name": "A/B/C variants",
  "meso_offers": [
    {"price": 72000, "contract_months": 24, "payment_terms_days": 45},
    {"price": 75000, "contract_months": 36, "payment_terms_days": 30},
    {"price": 68000, "contract_months": 18, "payment_terms_days": 60}
  ]
}
```

If the counterparty picks variant B, the agent learns they value long contracts. If they pick C, they're price-sensitive. The agent extracted hidden preference information without conceding anything. The environment rewards successful MESO usage through the `strategic_concessions` scoring component.

## Observation Space

The environment exposes a rich public observation while withholding the counterparty's hidden model. Agents must infer unknowns from messages, signals, and offer history.

| Exposed | Hidden |
|---|---|
| Current terms, offer history, counterparty's last structured offer | `cp_utility_weights` (true preference vector) |
| Counterparty messages and soft signals | `cp_acceptance_threshold` |
| Company targets, constraints, must-haves, nice-to-haves | `cp_ideal_terms`, `cp_floor_terms` |
| `inferred_counterparty_priorities` (updated via clarifications) | Internal engagement and outcome trackers |
| `reservation_value_hint`, `batna_summary` | Internal reservation discipline state |
| `compliance_risks`, `negotiation_phase`, `rounds_remaining`, `available_actions` | (none) |

Agents that only optimize against visible signals tend to overshoot and get close attempts rejected. Agents that actively probe, cross-reference offer history, and use MESO or tradeoff actions to extract hidden preference information consistently score higher.

## Reward Design

Per-step reward components provide dense shaping suitable for RL training:

| Component | Value | Fires when |
|---|---|---|
| `deal_delta` | ± variable | Terms move toward or away from company targets |
| `smart_tradeoff` | +bonus | Concession on a low-priority term gains a high-priority term |
| `successful_package` | +bonus | Package offer clears the counterparty acceptance threshold |
| `successful_meso` | +bonus | MESO variant is accepted by the counterparty |
| `constraint_violation` | − scaled | Proposed terms breach a hard constraint |
| `round_cost` | −0.012 | Every round (discourages drift) |
| `close_bonus` | +0.20 | Successful close within constraints |
| `close_penalty` | −0.30 | Episode times out with no resolution |

The final episode grade combines weighted task components with a geometric-mean quality multiplier:

```python
base = sum(weight_i * component_i for i in task_components)
multiplier = (reservation_quality * engagement * outcome) ** (1 / 3)
multiplier = max(0.15, multiplier)
final_score = clip(base * multiplier, 0.0, 1.0)
```

The geometric mean is deliberate. Arithmetic averages let agents optimize a single dimension (for example, always accepting quickly to farm the `close_bonus`). The geometric mean requires balanced performance across reservation discipline, engagement, and outcome quality, and a weakness in any one axis drags the whole score down. A 0.15 floor prevents catastrophic collapse from a single early mistake.

## Baseline Scores

Measured with `inference.py` using a two-stage baseline agent: the LLM produces a per-episode plan (priority terms, tradeoff pair, close threshold), and a rule-based controller selects actions each round using strategic moves such as package offers, tradeoffs, and MESO offers. Five episodes per task, temperature 0, fully deterministic.

| Task | Difficulty | GPT-4.1 | GPT-5.3 |
|---|---|---|---|
| `deal_qualification` | Easy | 0.339 | 0.386 |
| `multi_term_negotiation` | Medium | 0.332 | 0.337 |
| `strategic_contract_close` | Hard | 0.303 | 0.294 |

Per-episode score range: 0.26 – 0.56.

**Interpreting the numbers.** The scores sit around 0.30 – 0.40 for a reason: the environment is calibrated to challenge current frontier LLMs while leaving headroom for better agents, improved prompting, and RL fine-tuning.

- Monotonic difficulty progression (0.39 → 0.34 → 0.29 on GPT-5.3) confirms the task gradient.
- The 0.26 – 0.56 per-episode spread shows that strategic action choice, reservation discipline, and close-timing materially affect outcomes within a single model.
- The geometric-mean grader ensures single-dimension optimization cannot produce a high score.

For comparison: an ablation agent that skips `analyze_deal`, never sets a reservation value, and accepts the opening offer scores around 0.18 on the same scenario, roughly 2.6× lower than the strategic baseline. Strategic action choice, not raw model capability, dominates outcomes in this environment.

## Running the Baseline

The baseline agent uses the OpenAI Python client and reads the three standard hackathon environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py
```

Compatible with any OpenAI-API-compatible provider: OpenAI, the HuggingFace Inference Router, vLLM, and NVIDIA NIM (Nemotron). Tunables: `EPISODES_PER_TASK` (default 5), `INFERENCE_TEMPERATURE` (default 0).

The script emits the required structured stdout logs for Phase 2 evaluation:

```
[START] task=deal_qualification env=vendor_negotiation_gym model=gpt-4.1
[STEP]  step=1 action=analyze_deal(...) reward=0.09 done=false error=null
[STEP]  step=2 action=propose_terms(price=82800.0,payment_terms_days=38) reward=0.11 done=false error=null
[STEP]  step=3 action=propose_terms(price=79020.0,payment_terms_days=40) reward=0.02 done=false error=null
[STEP]  step=4 action=final_offer() reward=-0.09 done=true error=null
[END]   success=true steps=4 score=0.358 rewards=0.09,0.11,0.02,-0.09
```

## Setup

```bash
pip install uv
uv pip install -e .

# Run locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Or build and run the container
docker build -t negotiation-env .
docker run -p 7860:7860 negotiation-env

# Validate OpenEnv spec compliance
openenv validate .
```

This repository ships with the HuggingFace Spaces frontmatter above. Push to a Space with SDK `docker` and port `7860` to deploy unmodified.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/health`   | Liveness probe |
| `GET`  | `/metadata` | Environment metadata (name, version, tasks) |
| `GET`  | `/schema`   | JSON schemas for action, observation, and state |
| `POST` | `/reset`    | Start a new episode. Body: `{"seed": int, "task_id": "..."}` |
| `POST` | `/step`     | Submit a `NegotiationAction`; returns observation, reward, done, info |
| `GET`  | `/state`    | Current public episode state (hidden counterparty model excluded) |

## Example Episode

Scenario SCN-001, Enterprise CRM License Renewal (software vendor).

```
Round 0 ─ RESET
  Initial terms: price=$85k, contract=36mo, SLA=99.5%, support=standard
  Company targets: price=$70k, SLA=99.9%, support=premium
  Hidden: CP weights price:0.30, contract_months:0.25 (not visible to agent)

Round 1 ─ analyze_deal(focus_terms=["price", "sla_uptime"])
  → reward +0.10; inferred_counterparty_priorities populated

Round 2 ─ set_reservation_value(0.45)
  → reward +0.05; reservation discipline tracker armed

Round 3 ─ make_meso_offer([
             {price: 72k, contract: 24, support: premium},
             {price: 75k, contract: 36, support: premium},
             {price: 68k, contract: 18, support: standard}])
  → CP selects variant B → reveals contract length preference
  → reward +0.15 (successful_meso)

Round 4 ─ request_tradeoff(requested="sla_uptime=99.9", offered="payment_terms_days=30")
  → CP accepts → reward +0.12 (successful_tradeoff)

Round 5 ─ accept_counterparty_offer()
  → Deal closes. company_util=0.58, constraints clean
  → close_bonus +0.20
  → FINAL GRADE: 0.47 (geometric multiplier: 0.86)
```

## Design Principles

1. **Every action has an economic meaning.** Each of the 14 action types models a real procurement technique: MESO offers, bundled trades, BATNA-anchored walk-aways, explicit tradeoff requests.
2. **Hidden state forces inference, not memorization.** Counterparty utility weights are sampled per scenario and never exposed. Agents cannot pattern-match their way to high scores.
3. **Dense rewards for training, balanced grading for evaluation.** Per-step shaping supports gradient-based training, while the geometric-mean terminal grade resists single-dimension reward hacking.
4. **Calibrated, not punishing.** A 0.15 multiplier floor prevents catastrophic collapse from one mistake, so skill compounds across the trajectory.
5. **Spec-first and typed end-to-end.** All actions, observations, and state classes inherit from OpenEnv's typed base classes and expose a full `/schema`.

## Repository Layout

```
vendor_negotiation_gym/
├── server/
│   ├── app.py              # FastAPI app, OpenEnv endpoints
│   └── environment.py      # Core Environment class, grading, rewards
├── Dockerfile              # HuggingFace Spaces build
├── models.py               # Typed Pydantic Action / Observation / State
├── scenarios.py            # 52 hand-written scenarios
├── deal_engine.py          # Utility, constraint, and counterparty simulator
├── inference.py            # Baseline agent (LLM planner + rule-based controller)
├── client.py               # Thin HTTP client for local testing
├── tests/                  # Spec compliance and unit tests
├── openenv.yaml            # OpenEnv manifest (spec_version: 1)
└── pyproject.toml
```
