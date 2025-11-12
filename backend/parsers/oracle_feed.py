from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

from pydantic import ValidationError

from backend.parsers.models import OracleSentenceBatch, OracleSentenceRecord

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".jsonl", ".ndjson", ".json", ".csv"}


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
    """Placeholder ingestion class to be wired into the existing pipeline."""

    def ingest(self, batch: OracleSentenceBatch) -> None:
        logger.info("Prepared to ingest %s records (implementation pending).", batch.record_count)
        # TODO: integrate with embedding model and Postgres writes.
