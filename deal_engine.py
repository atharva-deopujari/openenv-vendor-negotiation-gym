"""
Deterministic counterparty response engine.
Computes counterparty utility, generates responses and signals,
decides acceptance/rejection based on hidden preference model.
"""
from typing import Any, Dict, List, Optional, Tuple


class DealEngine:
    """Deterministic engine that simulates counterparty behavior."""

    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.cp_weights = scenario["cp_utility_weights"]
        self.cp_ideal = scenario["cp_ideal_terms"]
        self.cp_floor = scenario.get("cp_floor_terms", {})
        self.cp_threshold = scenario["cp_acceptance_threshold"]
        self.initial_terms = scenario["initial_terms"]

    def compute_counterparty_utility(self, terms: dict) -> float:
        """
        Compute how favorable the current terms are to the counterparty.
        Returns 0.0-1.0 where 1.0 = counterparty's ideal deal.

        Uses full range from company_target (worst for CP) to cp_ideal (best).
        This ensures the initial offer gives CP ~0.4-0.6 utility and deals
        are closeable through negotiation.
        """
        score = 0.0
        total_weight = 0.0
        company_targets = self.scenario.get("company_targets", {})

        for term, weight in self.cp_weights.items():
            if term not in terms or term not in self.cp_ideal:
                continue

            current = terms[term]
            ideal = self.cp_ideal[term]
            # Use company target as worst-case for CP (full negotiation range)
            worst = company_targets.get(term, self.initial_terms.get(term, ideal))

            # Handle support_tier as ordinal
            if term == "support_tier":
                tier_order = {"basic": 0, "standard": 1, "premium": 2, "enterprise": 3}
                current = tier_order.get(str(current), 1)
                ideal = tier_order.get(str(ideal), 1)
                worst = tier_order.get(str(worst), 1)

            # Normalize: how close is current to CP ideal vs company target (worst for CP)
            if ideal == worst:
                norm = 1.0
            else:
                norm = (current - worst) / (ideal - worst)
                norm = max(0.0, min(1.0, norm))

            # If weight is negative, counterparty dislikes higher values
            if weight < 0:
                norm = 1.0 - norm
                weight = abs(weight)

            score += norm * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5
        return round(score / total_weight, 4)

    def compute_company_utility(self, terms: dict, targets: dict) -> float:
        """
        Compute how favorable the current terms are to YOUR company.
        Returns 0.0-1.0 where 1.0 = perfect for your side.
        """
        score = 0.0
        count = 0

        for term, target in targets.items():
            if term not in terms:
                continue

            current = terms[term]
            initial = self.initial_terms.get(term, current)

            if term == "support_tier":
                tier_order = {"basic": 0, "standard": 1, "premium": 2, "enterprise": 3}
                current = tier_order.get(str(current), 1)
                target = tier_order.get(str(target), 1)
                initial = tier_order.get(str(initial), 1)

            if target == initial:
                norm = 1.0
            else:
                norm = (current - initial) / (target - initial)
                norm = max(0.0, min(1.0, norm))

            score += norm
            count += 1

        if count == 0:
            return 0.5
        return round(score / count, 4)

    def check_constraint_violations(self, terms: dict, constraints: dict) -> List[str]:
        """Check if current terms violate any hard constraints. Return list of violations."""
        violations = []
        for key, limit in constraints.items():
            # Parse constraint name: e.g. "price_max" -> term="price", direction="max"
            parts = key.rsplit("_", 1)
            if len(parts) != 2:
                continue
            term, direction = parts
            if term not in terms:
                continue
            value = terms[term]

            if term == "support_tier":
                tier_order = {"basic": 0, "standard": 1, "premium": 2, "enterprise": 3}
                value = tier_order.get(str(value), 1)
                limit = tier_order.get(str(limit), 1) if isinstance(limit, str) else limit

            if direction == "max" and value > limit:
                violations.append(f"{term} ({value}) exceeds max ({limit})")
            elif direction == "min" and value < limit:
                violations.append(f"{term} ({value}) below min ({limit})")
        return violations

    def should_accept(self, terms: dict) -> bool:
        """Determine if counterparty would accept these terms."""
        utility = self.compute_counterparty_utility(terms)
        return utility >= self.cp_threshold

    def should_reject(self, terms: dict) -> bool:
        """Determine if counterparty would walk away (terms too unfavorable)."""
        utility = self.compute_counterparty_utility(terms)
        return utility < 0.08  # only reject truly extreme proposals

    def generate_response(self, terms: dict, round_index: int, max_rounds: int) -> Tuple[str, List[str]]:
        """
        Generate counterparty message and signals based on current terms.
        Returns (message, signals).
        """
        utility = self.compute_counterparty_utility(terms)
        responses = self.scenario.get("responses", {})
        base_signals = self.scenario.get("signals", [])

        # Select response tone
        if utility >= 0.7:
            message = responses.get("favorable", "We can work with this.")
        elif utility >= 0.4:
            message = responses.get("neutral", "Let's review these terms.")
        elif utility >= 0.15:
            message = responses.get("unfavorable", "This needs significant adjustments.")
        else:
            message = responses.get("reject", "We cannot proceed under these terms.")

        # Generate contextual signals based on which terms matter most
        signals = []
        sorted_weights = sorted(self.cp_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        top_terms = [t for t, w in sorted_weights[:3]]

        if round_index == 0:
            signals = base_signals[:2]
        elif round_index < max_rounds - 1:
            # Give hints about what matters
            for term in top_terms[:2]:
                if term in terms:
                    current = terms[term]
                    ideal = self.cp_ideal.get(term)
                    if ideal and current != ideal:
                        weight = self.cp_weights.get(term, 0)
                        if weight > 0.15:
                            signals.append(f"We'd like to revisit the {term.replace('_', ' ')} terms.")
                        elif weight < -0.1:
                            signals.append(f"The {term.replace('_', ' ')} terms are reasonable.")
            if not signals:
                signals = base_signals[1:2]
        else:
            # Last round: more direct
            if utility < self.cp_threshold:
                signals.append("This is our final round. We need material improvement to proceed.")
            else:
                signals.append("We're close to agreement. Let's finalize.")

        # Time pressure signal
        remaining = max_rounds - round_index
        if remaining <= 2 and utility < 0.6:
            signals.append("Our team needs a decision soon.")

        return message, signals

    def generate_counteroffer(self, current_terms: dict) -> Dict[str, Any]:
        """
        Generate a counteroffer that moves terms toward counterparty's ideal.
        Used when agent's proposal isn't acceptable but not rejection-worthy.
        """
        counter = dict(current_terms)

        # Move the top 2-3 weighted terms toward counterparty ideal
        sorted_weights = sorted(self.cp_weights.items(), key=lambda x: abs(x[1]), reverse=True)

        for term, weight in sorted_weights[:3]:
            if term not in counter or term not in self.cp_ideal:
                continue

            current = counter[term]
            ideal = self.cp_ideal[term]

            if term == "support_tier":
                continue  # don't counter on tier

            if isinstance(current, (int, float)) and isinstance(ideal, (int, float)):
                # Move 30% toward counterparty ideal
                counter[term] = round(current + 0.3 * (ideal - current), 2)
                if isinstance(self.initial_terms.get(term), int):
                    counter[term] = int(round(counter[term]))

        return counter
