from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import psycopg2
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer

from backend.logging_config import get_logger
from backend.parsers.models import OracleSentenceBatch, OracleSentenceRecord
from backend.settings import settings
from backend.utils import vector_to_list

logger = get_logger(__name__)


SUPPORTED_EXTENSIONS = {".jsonl", ".ndjson", ".json", ".csv"}


def _transform_flat_to_nested(flat_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform flat JSONL record to nested OracleSentenceRecord structure.
    
    Handles:
    - Flat structure where metadata fields are at top level
    - Field name mapping: snapshot_run_id -> snapshot_id
    - Extra fields go into metadata.extra
    """
    # Check if already nested (has 'metadata' key)
    if "metadata" in flat_record and isinstance(flat_record["metadata"], dict):
        return flat_record
    
    # Fields that belong at top level
    top_level_fields = {"chunk_id", "sentence"}
    
    # Metadata fields that are directly mapped
    metadata_fields = {
        "source_table",
        "product_code",
        "product_name",
        "promo_tier",
        "region",
        "security_level",
        "tags",
        "last_updated",
    }
    
    # Build nested structure
    nested: Dict[str, Any] = {
        "chunk_id": flat_record.get("chunk_id"),
        "sentence": flat_record.get("sentence"),
        "metadata": {},
    }
    
    # Map metadata fields
    for key in metadata_fields:
        if key in flat_record:
            nested["metadata"][key] = flat_record[key]
    
    # Map snapshot_run_id to snapshot_id
    if "snapshot_run_id" in flat_record:
        nested["metadata"]["snapshot_id"] = flat_record["snapshot_run_id"]
    
    # All other fields go into extra (except top-level fields)
    extra = {}
    for key, value in flat_record.items():
        if key not in top_level_fields and key not in metadata_fields and key != "snapshot_run_id":
            extra[key] = value
    
    if extra:
        nested["metadata"]["extra"] = extra
    
    # Ensure source_table is present (required field)
    if "source_table" not in nested["metadata"]:
        nested["metadata"]["source_table"] = flat_record.get("source_table", "UNKNOWN")
    
    return nested


def load_records(path: Path) -> OracleSentenceBatch:
    """Load and validate a batch of Oracle sentence records from disk."""

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    records: List[OracleSentenceRecord]
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = list(_load_json_lines(path))
    elif path.suffix.lower() == ".json":
        records = list(_load_json_array(path))
    else:
        records = list(_load_csv(path))

    batch = OracleSentenceBatch(records=records, source_file=str(path))
    logger.info("Loaded %s records from %s", batch.record_count, path)
    return batch


def _load_json_lines(path: Path) -> Iterable[OracleSentenceRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                # Transform flat structure to nested if needed
                payload = _transform_flat_to_nested(payload)
                yield OracleSentenceRecord.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.error("Invalid JSONL record at line %s in %s: %s", line_number, path, exc)
                raise


def _load_json_array(path: Path) -> Iterable[OracleSentenceRecord]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        raise

    if isinstance(data, dict):
        data = data.get("records", [])
    if not isinstance(data, Sequence):
        raise ValueError("JSON feed must be an array or contain a 'records' list")

    for index, entry in enumerate(data, start=1):
        try:
            # Transform flat structure to nested if needed
            entry = _transform_flat_to_nested(entry)
            yield OracleSentenceRecord.model_validate(entry)
        except ValidationError as exc:
            logger.error("Invalid record #%s in %s: %s", index, path, exc)
            raise


def _load_csv(path: Path) -> Iterable[OracleSentenceRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):  # header line is row 1
            try:
                metadata_fields = {
                    key[len("metadata.") :]: value
                    for key, value in row.items()
                    if key.startswith("metadata.") and value
                }
                payload = {
                    "chunk_id": row.get("chunk_id"),
                    "sentence": row.get("sentence"),
                    "metadata": metadata_fields,
                }
                yield OracleSentenceRecord.model_validate(payload)
            except ValidationError as exc:
                logger.error("Invalid CSV row #%s in %s: %s", index, path, exc)
                raise


def summarize_batch(batch: OracleSentenceBatch) -> str:
    """Return a short human-readable summary for logging or CLI output."""

    sample = batch.records[0] if batch.records else None
    sample_piece = f" Sample: {sample.chunk_id} -> {sample.sentence[:80]!r}." if sample else ""
    return f"Batch with {batch.record_count} records from {batch.source_file}.{sample_piece}"


class OracleFeedIngestor:
    """Ingest Oracle promo feed sentences into the vector store."""

    def __init__(self) -> None:
        self._embedding_model = SentenceTransformer(settings.embedding_model_name)

    def ingest(self, batch: OracleSentenceBatch) -> None:
        if not batch.records:
            logger.warning("Oracle feed ingest called with empty batch from %s", batch.source_file)
            return

        logger.info("Ingesting %s promo records from %s", batch.record_count, batch.source_file)

        connection = psycopg2.connect(
            host=settings.db_host,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password.get_secret_value(),
        )
        try:
            with connection:
                with connection.cursor() as cursor:
                    for record in batch.records:
                        chunk_text = self._compose_chunk_text(record, batch.source_file)
                        embedding = vector_to_list(self._embedding_model.encode(chunk_text))

                        cursor.execute("DELETE FROM vector_index WHERE doc_id = %s", (record.chunk_id,))
                        source_label = None
                        if record.metadata.extra:
                            source_label = record.metadata.extra.get("source_file")
                        if not source_label:
                            source_label = batch.source_file or record.metadata.source_table or "oracle_feed"

                        cursor.execute(
                            """
                            INSERT INTO vector_index (
                                doc_id,
                                source_path,
                                department,
                                security_level,
                                allowed_groups,
                                last_updated_by,
                                last_updated_at,
                                chunk_text,
                                embedding
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s)
                            """,
                                (
                                record.chunk_id,
                                    source_label,
                                "Sales",
                                (record.metadata.security_level or "Internal"),
                                ["sales"],
                                "oracle_feed",
                                chunk_text,
                                embedding,
                            ),
                        )
        finally:
            connection.close()

        logger.info("Completed ingest for %s promo records", batch.record_count)

    @staticmethod
    def _compose_chunk_text(record: OracleSentenceRecord, source_file: str | None) -> str:
        """Build a rich chunk body combining the summary sentence and key metadata."""
        parts: List[str] = [record.sentence]

        meta_lines: List[str] = []
        metadata = record.metadata

        if metadata.product_name or metadata.product_code:
            meta_lines.append(
                f"Product: {metadata.product_name or 'Unknown'}"
                + (f" (code {metadata.product_code})" if metadata.product_code else "")
            )

        if metadata.promo_tier:
            meta_lines.append(f"Promo Tier: {metadata.promo_tier}")
        if metadata.region:
            meta_lines.append(f"Region: {metadata.region}")
        if metadata.tags:
            meta_lines.append(f"Tags: {', '.join(metadata.tags)}")
        if metadata.snapshot_id:
            meta_lines.append(f"Snapshot: {metadata.snapshot_id}")
        if metadata.last_updated:
            meta_lines.append(f"Last Updated: {metadata.last_updated.isoformat()}")
        if source_file:
            meta_lines.append(f"Source File: {source_file}")

        if metadata.extra:
            extras = ", ".join(f"{k}={v}" for k, v in metadata.extra.items())
            meta_lines.append(f"Extra: {extras}")

        if meta_lines:
            parts.append("Metadata:\n" + "\n".join(meta_lines))

        return "\n\n".join(parts).strip()
