from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backend.parsers.oracle_feed import (
    OracleFeedIngestor,
    load_records,
    summarize_batch,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and ingest Oracle-generated promo sentences into the RAG store.",
    )
    parser.add_argument(
        "feed_path",
        type=Path,
        help="Path to the JSONL/CSV export provided by the Oracle team.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize without writing to the database.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args()

    batch = load_records(args.feed_path)
    logger.info(summarize_batch(batch))

    if args.dry_run:
        logger.info("Dry run enabled; skipping ingestion step.")
        return

    ingestor = OracleFeedIngestor()
    ingestor.ingest(batch)


if __name__ == "__main__":
    main()
