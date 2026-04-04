"""Tests for ScenarioDatabase."""
from scenarios import ScenarioDatabase


class TestScenarioDatabase:
    def test_has_at_least_50(self):
        db = ScenarioDatabase()
        assert len(db.scenarios) >= 50

    def test_all_types_covered(self):
        db = ScenarioDatabase()
        types = {s["counterparty_type"] for s in db.scenarios}
        expected = {"software_vendor", "cloud_provider", "logistics_vendor", "marketing_agency", "enterprise_buyer"}
        assert types == expected

    def test_required_fields(self):
        db = ScenarioDatabase()
        required = ["id", "title", "counterparty_type", "initial_terms", "company_targets",
                     "company_constraints", "cp_utility_weights", "cp_acceptance_threshold"]
        for s in db.scenarios:
            for f in required:
                assert f in s, f"Scenario {s.get('id')} missing {f}"

    def test_reproducible_random(self):
        db = ScenarioDatabase()
        s1 = db.get_random_scenario(seed=42)
        s2 = db.get_random_scenario(seed=42)
        assert s1["id"] == s2["id"]
