from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ChunkMetadata(BaseModel):
    """Normalized metadata captured for each chunk."""

    source_table: str = Field(..., description="Origin table or view name in Oracle.")
    snapshot_id: Optional[str] = Field(
        default=None, description="Identifier for the export run (batch id, filename, etc.)."
    )
    snapshot_ts: Optional[datetime] = Field(
        default=None, description="Timestamp when the source snapshot was taken."
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        description="Row-level last updated timestamp if provided for incremental loads.",
    )
    product_code: Optional[str] = None
    product_name: Optional[str] = None
    promo_tier: Optional[str] = None
    region: Optional[str] = None
    security_level: Optional[str] = Field(
        default=None, description="Access control label (Internal, Confidential, etc.)."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Lightweight labels used for pre-filtering (e.g., ['sales', 'promo']).",
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Carrier for additional metadata fields that we do not model yet.",
    )

    @validator("tags", pre=True, always=True)
    def _coerce_tags(cls, value: Optional[Any]) -> List[str]:  # noqa: D401,N805
        """Ensure tags is always a list of non-empty strings."""

        if value is None:
            return []
        if isinstance(value, str):
            candidate = [value]
        else:
            candidate = list(value)
        return [str(tag).strip() for tag in candidate if str(tag).strip()]


class OracleSentenceRecord(BaseModel):
    """Single sentence provided by the Oracle feed."""

    chunk_id: str = Field(..., description="Stable identifier for this chunk (promo ID hash, etc.).")
    sentence: str = Field(..., description="One-sentence natural language summary of the promo row.")
    metadata: ChunkMetadata = Field(
        default_factory=ChunkMetadata, description="Structured metadata for downstream filters."
    )

    @validator("sentence")
    def _trim_sentence(cls, value: str) -> str:  # noqa: D401,N805
        """Normalize whitespace so embeddings stay consistent."""

        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Sentence text cannot be empty")
        return cleaned


class OracleSentenceBatch(BaseModel):
    """Envelope describing a batch of Oracle sentences loaded from a file."""

    records: List[OracleSentenceRecord] = Field(
        default_factory=list, description="Validated sentence records ready for ingestion."
    )
    source_file: Optional[str] = Field(
        default=None, description="Original filename or URI for traceability."
    )
    record_count: int = Field(0, description="Number of records in the batch.")

    @validator("record_count", always=True)
    def _derive_count(cls, value: int, values: Dict[str, Any]) -> int:  # noqa: D401,N805
        """Keep record count aligned with the records payload."""

        if value:
            return value
        records = values.get("records") or []
        return len(records)
