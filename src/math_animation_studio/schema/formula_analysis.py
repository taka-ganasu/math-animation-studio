from __future__ import annotations

from pydantic import Field

from .storyboard import StrictModel


class SymbolRole(StrictModel):
    symbol: str
    normalized_symbol: str | None = None
    role: str
    meaning: str
    intuition: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class OperationAnalysis(StrictModel):
    operation: str
    meaning: str
    intuition: str | None = None
    visual_hint: str | None = None


class FormulaAnalysis(StrictModel):
    raw_formula: str
    normalized_formula_latex: str
    detected_name: str | None = None
    short_description: str
    symbols: list[SymbolRole]
    operations: list[OperationAnalysis]
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
