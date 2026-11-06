MAX_AMOUNT = 10_000.0          # any single transaction above this is blocked
MAX_TX_PER_HOUR = 20           # more than this in an hour goes to review
BLOCKED_COUNTRIES = {"XX"}     # placeholder code; replace with your real list


def check_rules(amount: float, country: str | None, tx_count_last_hour: int) -> list[str]:
    """Return the names of triggered rules. An empty list means no rule fired."""
    triggered = []
    if amount > MAX_AMOUNT:
        triggered.append("amount_over_limit")
    if country and country.upper() in BLOCKED_COUNTRIES:
        triggered.append("blocked_country")
    if tx_count_last_hour > MAX_TX_PER_HOUR:
        triggered.append("velocity_exceeded")
    return triggered


def rule_decision(triggered: list[str]) -> str | None:
    """Hard rules that block outright; None means the model decides."""
    hard_block = {"amount_over_limit", "blocked_country"}
    if any(rule in hard_block for rule in triggered):
        return "block"
    if triggered:
        return "review"
    return None
