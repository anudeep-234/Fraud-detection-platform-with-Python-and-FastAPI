from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Index
from app.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)


class CustomerBehaviorProfile(Base):
    __tablename__ = "customer_behavior_profiles"
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    transaction_count = Column(Integer, nullable=False, default=0)
    average_amount = Column(Float, nullable=False, default=0.0)
    amount_stddev = Column(Float, nullable=False, default=0.0)
    usual_countries = Column(Text, nullable=False, default="[]")
    typical_hour = Column(Float, nullable=False, default=12.0)
    past_alert_count = Column(Integer, nullable=False, default=0)
    amount_sum = Column(Float, nullable=False, default=0.0)
    amount_sum_squares = Column(Float, nullable=False, default=0.0)
    hour_counts = Column(Text, nullable=False, default="{}")
    country_counts = Column(Text, nullable=False, default="{}")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    merchant = Column(String(100))
    country = Column(String(2))
    device_id = Column(String(64))
    ip_address = Column(String(45))
    # Masked account number: only the last 4 digits are stored
    account_last4 = Column(String(4))
    features = Column(Text)  # JSON of time, amount, V1-V28 for retraining

    __table_args__ = (
        Index("ix_transactions_user_time", "user_id", "timestamp"),
    )


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    model_version = Column(String(20), nullable=False)
    risk_score = Column(Float, nullable=False)
    decision = Column(String(10), nullable=False)
    reasons = Column(Text)
    is_shadow = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    status = Column(String(20), default="open", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False)
    reviewer_id = Column(Integer, ForeignKey("users.id"))
    label = Column(String(20), nullable=False)  # "fraud" or "legitimate"
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True)
    version = Column(String(20), unique=True, nullable=False)
    metrics = Column(Text)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    actor = Column(String(100))
    action = Column(String(50), nullable=False)
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class TranscriptAnalysis(Base):
    __tablename__ = "transcript_analyses"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    text = Column(Text, nullable=False)
    risk_score = Column(Float, nullable=False)
    decision = Column(String(10), nullable=False)
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
