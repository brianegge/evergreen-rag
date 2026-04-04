"""MARC-XML text extraction."""

from evergreen_rag.extractor.marc_extractor import (
    extract_record,
    extract_records_from_collection,
)

__all__ = ["extract_record", "extract_records_from_collection"]
