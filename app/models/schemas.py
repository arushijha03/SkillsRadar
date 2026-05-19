from typing import Literal, Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    role: str
    company: Optional[str] = None
    domain: Optional[str] = None
    horizon: Literal["6mo", "12mo"] = "6mo"


class SkillSignal(BaseModel):
    skill: str
    svs_score: float
    category: Literal["in_demand_now", "rising_6mo", "emerging_12mo"]
    sources: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    role: str
    skills: list[SkillSignal]
    report_markdown: str
    run_id: str
