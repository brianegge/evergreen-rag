#!/usr/bin/env python3
"""Download and load bulk MARC records into the Evergreen RAG database.

Downloads binary MARC21 files from the Internet Archive's contributed
MARC records collection and loads them into biblio.record_entry as MARCXML.

Usage:
    # Download ~100k records from Internet Archive and load into local DB
    python scripts/load_marc_bulk.py --source ia --limit 100000

    # Load from a local .mrc file
    python scripts/load_marc_bulk.py --file /path/to/records.mrc --limit 50000

    # Dry run — download and count records without loading
    python scripts/load_marc_bulk.py --source ia --limit 1000 --dry-run

Requirements:
    pip install pymarc psycopg httpx
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import os
import sys
import time
from pathlib import Path

import httpx
import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# Internet Archive MARC collection files — curated for quality and size.
# Each entry: (identifier, filename, approx record count, description)
# Internet Archive MARC collection files — verified and accessible.
# Each entry: (identifier, filename, approx record count, description)
IA_SOURCES = [
    (
        "marc_openlibraries_sanfranciscopubliclibrary",
        "sfpl_chq_2018_12_24_run01.mrc",
        200_000,
        "San Francisco Public Library run01 — ~200k records (207 MB)",
    ),
    (
        "marc_oregon_summit_records",
        "catalog_files/ohsu_ncnm_wscc_bibs.mrc",
        150_000,
        "Oregon Summit OHSU/NCNM/WSCC — ~150k records (156 MB)",
    ),
    (
        "marc_lendable_books",
        "all_meta.mrc",
        100_000,
        "Internet Archive lendable books — full set (129 MB)",
    ),
]

# Smaller sets for quick testing
IA_SMALL_SOURCES = [
    (
        "marc_lendable_books",
        "01_meta.mrc",
        10_000,
        "Internet Archive lendable books part 01 — ~10k records (12 MB)",
    ),
]

DATABASE_URL_DEFAULT = "postgresql://evergreen:evergreen@192.168.254.35:5432/evergreen_rag"


def marc_record_to_xml(record) -> str | None:
    """Convert a pymarc Record to MARCXML string."""
    try:
        from pymarc import record_to_xml
        xml_bytes = record_to_xml(record, namespace=True)
        if isinstance(xml_bytes, bytes):
            return xml_bytes.decode("utf-8")
        return str(xml_bytes)
    except Exception as e:
        logger.debug("Failed to convert record to XML: %s", e)
        return None


def extract_record_id(record) -> int | None:
    """Extract a numeric ID from MARC 001 field, or return None."""
    field_001 = record.get("001")
    if field_001:
        raw = field_001.data.strip() if hasattr(field_001, "data") else str(field_001).strip()
        # Try to extract a numeric ID
        digits = "".join(c for c in raw if c.isdigit())
        if digits:
            return int(digits[:15])  # cap length to avoid overflow
    return None


def _download_file(url: str, dest: Path) -> None:
    """Stream-download a URL to a local file with progress display."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=600.0, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"\r  Downloaded {downloaded / 1024 / 1024:.1f} MB "
                            f"/ {total / 1024 / 1024:.1f} MB ({pct:.0f}%)",
                            end="",
                            flush=True,
                        )
            print()
    size_mb = dest.stat().st_size / (1024 * 1024)
    logger.info("Downloaded %.1f MB to %s", size_mb, dest)


def download_ia_marc(identifier: str, filename: str, dest_dir: Path) -> Path:
    """Download a MARC file from Internet Archive if not already cached."""
    # Use the basename for local storage (flatten subdirectories)
    local_name = Path(filename).name
    dest_path = dest_dir / local_name

    if dest_path.exists() and dest_path.stat().st_size > 0:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        logger.info("Using cached file: %s (%.1f MB)", dest_path, size_mb)
        return dest_path

    url = f"https://archive.org/download/{identifier}/{filename}"
    logger.info("Downloading %s ...", url)
    _download_file(url, dest_path)
    return dest_path


def iter_marc_records(file_path: Path):
    """Yield pymarc Record objects from a .mrc or .mrc.gz file."""
    from pymarc import MARCReader

    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rb") as f:
        reader = MARCReader(f, to_unicode=True, force_utf8=True, utf8_handling="replace")
        for record in reader:
            if record is not None:
                yield record


def load_records(
    file_path: Path,
    database_url: str,
    limit: int,
    batch_size: int = 500,
    dry_run: bool = False,
    id_offset: int = 10000,
) -> dict:
    """Load MARC records from file into biblio.record_entry.

    Returns stats dict with counts.
    """
    stats = {"total_read": 0, "loaded": 0, "skipped": 0, "errors": 0}
    batch: list[tuple[int, str]] = []
    seen_ids: set[int] = set()
    auto_id = id_offset  # starting ID for records without a 001

    start_time = time.time()

    for record in iter_marc_records(file_path):
        stats["total_read"] += 1

        if stats["total_read"] > limit:
            break

        # Convert to XML
        xml = marc_record_to_xml(record)
        if not xml:
            stats["skipped"] += 1
            continue

        # Get or assign record ID
        record_id = extract_record_id(record)
        if record_id is None or record_id in seen_ids:
            auto_id += 1
            record_id = auto_id

        seen_ids.add(record_id)
        batch.append((record_id, xml))

        if len(batch) >= batch_size:
            if not dry_run:
                inserted = _insert_batch(database_url, batch)
                stats["loaded"] += inserted
                stats["errors"] += len(batch) - inserted
            else:
                stats["loaded"] += len(batch)
            batch.clear()

            elapsed = time.time() - start_time
            rate = stats["total_read"] / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d read, %d loaded, %d errors (%.0f records/sec)",
                stats["total_read"],
                stats["loaded"],
                stats["errors"],
                rate,
            )

    # Final batch
    if batch:
        if not dry_run:
            inserted = _insert_batch(database_url, batch)
            stats["loaded"] += inserted
            stats["errors"] += len(batch) - inserted
        else:
            stats["loaded"] += len(batch)

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["records_per_second"] = round(stats["total_read"] / elapsed, 1) if elapsed > 0 else 0
    return stats


def _insert_batch(database_url: str, batch: list[tuple[int, str]]) -> int:
    """Insert a batch of (id, marc_xml) into biblio.record_entry. Returns count inserted."""
    inserted = 0
    try:
        with psycopg.connect(database_url, autocommit=False) as conn:
            with conn.cursor() as cur:
                for record_id, xml in batch:
                    try:
                        with conn.transaction():
                            cur.execute(
                                """
                                INSERT INTO biblio.record_entry (id, marc)
                                VALUES (%s, %s)
                                ON CONFLICT (id) DO UPDATE SET
                                    marc = EXCLUDED.marc,
                                    edit_date = NOW()
                                """,
                                (record_id, xml),
                            )
                            inserted += 1
                    except Exception as e:
                        logger.debug("Failed to insert record %d: %s", record_id, e)
    except Exception as e:
        logger.error("Batch insert failed: %s", e)
    return inserted


def count_existing_records(database_url: str) -> int:
    """Count existing records in biblio.record_entry."""
    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM biblio.record_entry")
                return cur.fetchone()[0]
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download and load bulk MARC records into Evergreen RAG database",
    )
    parser.add_argument(
        "--source",
        choices=["ia", "ia-small"],
        default=None,
        help="Download source: 'ia' for Internet Archive large sets, 'ia-small' for smaller sets",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to a local .mrc or .mrc.gz file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="Maximum number of records to load (default: 100000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Database insert batch size (default: 500)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="PostgreSQL connection URL (default: from DATABASE_URL env or localhost)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/marc-cache"),
        help="Directory to cache downloaded MARC files (default: data/marc-cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count records without loading into database",
    )
    parser.add_argument(
        "--id-offset",
        type=int,
        default=10_000,
        help="Starting auto-ID for records without a 001 field (default: 10000)",
    )

    args = parser.parse_args()

    database_url = args.database_url or os.environ.get("DATABASE_URL", DATABASE_URL_DEFAULT)

    # Check pymarc is installed
    try:
        import pymarc  # noqa: F401
    except ImportError:
        logger.error("pymarc is required: pip install pymarc")
        sys.exit(1)

    if not args.dry_run:
        existing = count_existing_records(database_url)
        logger.info("Existing records in database: %d", existing)

    # Determine MARC file source
    if args.file:
        if not args.file.exists():
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        marc_path = args.file
    elif args.source:
        sources = IA_SOURCES if args.source == "ia" else IA_SMALL_SOURCES
        identifier, filename, _, description = sources[0]
        logger.info("Source: %s", description)
        marc_path = download_ia_marc(identifier, filename, args.cache_dir)
    else:
        logger.error("Specify --source or --file")
        sys.exit(1)

    logger.info("Loading up to %d records from %s", args.limit, marc_path)
    if args.dry_run:
        logger.info("DRY RUN — no database writes")

    stats = load_records(
        file_path=marc_path,
        database_url=database_url,
        limit=args.limit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        id_offset=args.id_offset,
    )

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("  Records read:   %d", stats["total_read"])
    logger.info("  Records loaded: %d", stats["loaded"])
    logger.info("  Skipped:        %d", stats["skipped"])
    logger.info("  Errors:         %d", stats["errors"])
    logger.info("  Elapsed:        %.1fs", stats["elapsed_seconds"])
    logger.info("  Rate:           %.0f records/sec", stats["records_per_second"])

    if not args.dry_run:
        final_count = count_existing_records(database_url)
        logger.info("  Total in DB:    %d", final_count)


if __name__ == "__main__":
    main()
