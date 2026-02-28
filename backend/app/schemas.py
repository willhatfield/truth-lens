from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from datetime import datetime, timezone

SCHEMA_VERSION = "1.0"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventEnvelope(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    analysis_id: str
    type: str
    ts: str = Field(default_factory=now_iso)
    payload: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    prompt: str


class AnalyzeResponse(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    analysis_id: str
    ws_url: Optional[str] = None
    sse_url: Optional[str] = None
