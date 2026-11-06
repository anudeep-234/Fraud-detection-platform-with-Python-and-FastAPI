from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class TransactionIn(BaseModel):
    user_id: int
    amount: float = Field(gt=0)
    time: float = Field(ge=0, description="Seconds since the first transaction, as in the dataset")
    merchant: Optional[str] = None
    country: Optional[str] = Field(default=None, min_length=2, max_length=2)
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    account_last4: Optional[str] = Field(default=None, min_length=4, max_length=4)
    v_features: dict[str, float]

    @field_validator("v_features")
    @classmethod
    def check_v_features(cls, v):
        expected = {f"V{i}" for i in range(1, 29)}
        missing = expected - set(v)
        if missing:
            raise ValueError(f"missing features: {sorted(missing)}")
        return v


class Reason(BaseModel):
    feature: str
    contribution: float


class PredictionOut(BaseModel):
    transaction_id: int
    risk_score: float
    decision: Literal["approve", "review", "block"]
    reasons: List[Reason]
    model_version: str
