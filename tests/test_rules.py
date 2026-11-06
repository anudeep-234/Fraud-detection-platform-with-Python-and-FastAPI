from app.rules import check_rules, rule_decision


def test_amount_over_limit_blocks():
    triggered = check_rules(15000, "US", 0)
    assert "amount_over_limit" in triggered
    assert rule_decision(triggered) == "block"


def test_high_velocity_goes_to_review():
    triggered = check_rules(50, "US", 25)
    assert rule_decision(triggered) == "review"


def test_normal_transaction_passes_rules():
    assert check_rules(50, "US", 2) == []
    assert rule_decision([]) is None
