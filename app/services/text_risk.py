import json
import re

import httpx

from app.config import settings

SCAM_PATTERNS = {
    "urgency": r"\b(urgent|immediately|right now|within \d+ (minutes|hours)|act fast)\b",
    "credential_request": r"\b(otp|one[- ]time (password|code)|pin|password|cvv|card number)\b",
    "impersonation": r"\b(bank (officer|staff|security)|fraud department|rbi|kyc)\b",
    "threat": r"\b(account (will be|is) (blocked|suspended|closed)|legal action|arrest)\b",
    "secrecy": r"\b(don'?t tell|do not tell|keep this secret|don'?t inform)\b",
}

def rule_based_score(text: str) -> dict:
    lowered = text.lower()
    hits = [name for name, pat in SCAM_PATTERNS.items() if re.search(pat, lowered)]
    score = min(1.0, len(hits) / 3)
    return {"method": "rules", "risk_score": round(score, 3), "indicators": hits}


def ai_score(text: str) -> dict | None:
    """Ask a free OpenRouter model to classify the transcript. Returns None if no key is set."""
    key = settings.openrouter_api_key
    if not key:
        return None

    prompt = (
        "You are a fraud analyst. Read this customer transcript and decide whether it "
        "shows signs of a scam or social-engineering attack against the customer or bank. "
        'Respond ONLY with JSON: {"risk_score": number between 0 and 1, '
        '"indicators": [strings], "explanation": string}.\n\nTranscript:\n' + text
    )
    resp = httpx.post(
        settings.openrouter_url,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        },
        timeout=90,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:]
    data = json.loads(raw)
    return {
        "method": "ai",
        "risk_score": round(float(data["risk_score"]), 3),
        "indicators": data.get("indicators", []),
        "explanation": data.get("explanation", ""),
    }


def analyze_transcript(text: str) -> dict:
    rules = rule_based_score(text)
    ai = None
    try:
        ai = ai_score(text)
    except Exception as exc:  # network, quota, or parse errors must not break the API
        rules["ai_error"] = str(exc)[:200]
    if ai is None:
        return rules
    combined = max(rules["risk_score"], ai["risk_score"])
    return {
        "risk_score": combined,
        "decision": "review" if combined >= 0.5 else "approve",
        "rules": rules,
        "ai": ai,
    }

