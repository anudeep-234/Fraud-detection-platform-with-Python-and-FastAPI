import json
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import models_db as m
from app.auth import AuthUser, create_token, hash_password, require_role, verify_password
from app.database import Base, SessionLocal, engine, get_db
from app.rules import check_rules, rule_decision
from app.schemas import PredictionOut, TransactionIn
from app.services.scoring import MODEL_VERSION, score_transaction
from app.services.profiles import get_profile, scoring_profile, update_profile
from app.services.velocity import record_transaction, velocity_features

SEVERITY = {"approve": 0, "review": 1, "block": 2}
SEEDED_USERS = (
    ("admin", "admin-pass", "admin"),
    ("analyst", "analyst-pass", "analyst"),
    ("viewer", "viewer-pass", "viewer"),
)


def seed_default_users() -> None:
    db = SessionLocal()
    try:
        for username, password, role in SEEDED_USERS:
            if db.query(AuthUser).filter_by(username=username).first() is None:
                db.add(
                    AuthUser(
                        username=username,
                        password_hash=hash_password(password),
                        role=role,
                    )
                )
        db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    seed_default_users()
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


class LoginIn(BaseModel):
    username: str
    password: str


class ReviewIn(BaseModel):
    label: str
    note: str | None = None


@app.post("/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.query(AuthUser).filter_by(username=body.username).first()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="wrong username or password")
    return {"access_token": create_token(user), "token_type": "bearer", "role": user.role}


@app.get("/users/{user_id}/profile")
def user_profile(
    user_id: int,
    db: Session = Depends(get_db),
    user: AuthUser = Depends(require_role("analyst")),
):
    return {"user_id": user_id, **get_profile(db, user_id)}


@app.post("/predict", response_model=PredictionOut)
def predict(
    tx: TransactionIn,
    db: Session = Depends(get_db),
    user: AuthUser = Depends(require_role("viewer")),
):
    if db.get(m.User, tx.user_id) is None:
        db.add(m.User(id=tx.user_id, name=f"user-{tx.user_id}"))
        db.flush()

    vel = velocity_features(tx.user_id)
    profile = get_profile(db, tx.user_id)

    triggered = check_rules(tx.amount, tx.country, vel["tx_count_last_hour"])
    rule_dec = rule_decision(triggered)
    rule_reasons = [{"feature": r, "contribution": 0.0} for r in triggered]

    if rule_dec == "block":
        decision = "block"
        risk = 1.0
        model_reasons = []
    else:
        result = score_transaction(
            tx.model_dump(),
            vel["time_since_last"],
            vel["tx_count_last_hour"],
            scoring_profile(profile, tx.country),
        )
        risk = result["risk_score"]
        decision = result["decision"]
        model_reasons = (
            [r for r in result["reasons"] if r["contribution"] > 0]
            if decision != "approve" else []
        )
        if rule_dec and SEVERITY[rule_dec] > SEVERITY[decision]:
            decision = rule_dec

    reasons = rule_reasons + model_reasons

    tx_row = m.Transaction(
        user_id=tx.user_id,
        amount=tx.amount,
        timestamp=datetime.utcnow(),
        merchant=tx.merchant,
        country=tx.country,
        device_id=tx.device_id,
        ip_address=tx.ip_address,
        account_last4=tx.account_last4,
        features=json.dumps({
            **tx.model_dump(include={"time", "amount", "v_features"}),
            "time_since_last": vel["time_since_last"],
            "tx_count_last_hour": vel["tx_count_last_hour"],
        }),
    )
    db.add(tx_row)
    db.flush()

    pred = m.Prediction(
        transaction_id=tx_row.id,
        model_version=MODEL_VERSION,
        risk_score=risk,
        decision=decision,
        reasons=json.dumps(reasons),
    )
    db.add(pred)
    db.flush()

    if decision != "approve":
        db.add(m.Alert(prediction_id=pred.id, status="open"))

    update_profile(
        db,
        tx.user_id,
        tx.amount,
        tx.country,
        tx.time,
        decision != "approve",
    )

    db.add(m.AuditLog(
        actor=user.username,
        action="predict",
        details=f"tx={tx_row.id} user={tx.user_id} decision={decision} risk={risk}",
    ))
    db.commit()

    record_transaction(tx.user_id, str(tx_row.id))

    return PredictionOut(
        transaction_id=tx_row.id,
        risk_score=risk,
        decision=decision,
        reasons=reasons,
        model_version=MODEL_VERSION,
    )


@app.get("/alerts")
def list_alerts(
    status: str = "open",
    db: Session = Depends(get_db),
    user: AuthUser = Depends(require_role("viewer")),
):
    rows = (
        db.query(m.Alert, m.Prediction, m.Transaction)
        .join(m.Prediction, m.Alert.prediction_id == m.Prediction.id)
        .join(m.Transaction, m.Prediction.transaction_id == m.Transaction.id)
        .filter(m.Alert.status == status)
        .order_by(m.Alert.created_at.desc())
        .limit(100)
        .all()
    )
    return [
        {
            "alert_id": a.id,
            "status": a.status,
            "transaction_id": t.id,
            "user_id": t.user_id,
            "amount": t.amount,
            "decision": p.decision,
            "risk_score": p.risk_score,
            "reasons": json.loads(p.reasons or "[]"),
            "created_at": a.created_at.isoformat(),
        }
        for a, p, t in rows
    ]


@app.post("/alerts/{alert_id}/review")
def review_alert(
    alert_id: int,
    body: ReviewIn,
    db: Session = Depends(get_db),
    user: AuthUser = Depends(require_role("analyst")),
):
    if body.label not in ("fraud", "legitimate"):
        raise HTTPException(status_code=400, detail="label must be 'fraud' or 'legitimate'")

    alert = db.get(m.Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="alert not found")
    if alert.status != "open":
        raise HTTPException(status_code=409, detail="alert already reviewed")

    review = m.Review(alert_id=alert.id, reviewer_id=None, label=body.label, note=body.note)
    db.add(review)
    alert.status = "closed"
    db.add(m.AuditLog(
        actor=user.username,
        action="review",
        details=f"alert={alert.id} label={body.label}",
    ))
    db.commit()
    return {"review_id": review.id, "alert_id": alert.id, "status": "closed", "reviewed_by": user.username}
from pathlib import Path
from fastapi.responses import FileResponse

UI_PATH = Path(__file__).resolve().parent.parent / "ui" / "index.html"


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(UI_PATH)

from pydantic import Field


class TranscriptIn(BaseModel):
    user_id: int | None = None
    text: str = Field(min_length=1, max_length=20000)


@app.post("/transcripts/analyze")
def analyze_transcript_route(
    body: TranscriptIn,
    db: Session = Depends(get_db),
    user: AuthUser = Depends(require_role("viewer")),
):
    from app.services.text_risk import analyze_transcript

    result = analyze_transcript(body.text)
    decision = result.get("decision") or ("review" if result["risk_score"] >= 0.5 else "approve")
    indicators = result["indicators"] if "indicators" in result else (
        result["rules"]["indicators"] + result["ai"]["indicators"]
    )

    row = m.TranscriptAnalysis(
        user_id=body.user_id,
        text=body.text,
        risk_score=result["risk_score"],
        decision=decision,
        details=json.dumps(result),
    )
    db.add(row)
    db.flush()
    db.add(m.AuditLog(
        actor=user.username,
        action="transcript_analyze",
        details=f"analysis={row.id} decision={decision} risk={result['risk_score']}",
    ))
    db.commit()

    return {
        "analysis_id": row.id,
        "risk_score": result["risk_score"],
        "decision": decision,
        "indicators": indicators,
    }


@app.get("/transcripts/flagged")
def list_flagged(
    db: Session = Depends(get_db),
    user: AuthUser = Depends(require_role("analyst")),
):
    rows = (
        db.query(m.TranscriptAnalysis)
        .filter(m.TranscriptAnalysis.decision == "review")
        .order_by(m.TranscriptAnalysis.created_at.desc())
        .limit(100)
        .all()
    )
    return [
        {
            "analysis_id": r.id,
            "user_id": r.user_id,
            "risk_score": r.risk_score,
            "text": r.text,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
