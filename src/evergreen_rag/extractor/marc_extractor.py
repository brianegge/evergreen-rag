"""MARC-XML text extraction for bibliographic records."""

from __future__ import annotations

import logging
from typing import Any

from lxml import etree

from evergreen_rag.models.marc import ExtractedRecord

logger = logging.getLogger(__name__)

MARC_NS = "http://www.loc.gov/MARC21/slim"
NS = {"m": MARC_NS}


def extract_record(marc_xml: str | bytes, record_id: int | None = None) -> ExtractedRecord | None:
    """Parse a single MARC-XML ``<record>`` and return an ``ExtractedRecord``.

    If *record_id* is ``None``, it is read from controlfield 001.
    Returns ``None`` if the record cannot be parsed.
    """
    try:
        if isinstance(marc_xml, str):
            marc_xml = marc_xml.encode("utf-8")
        root = etree.fromstring(marc_xml)
        return _extract_from_element(root, record_id)
    except Exception:
        logger.exception("Failed to parse MARC-XML for record %s", record_id)
        return None


def extract_records_from_collection(xml_path: str) -> list[ExtractedRecord]:
    """Parse a MARC-XML collection file and return all extractable records."""
    try:
        tree = etree.parse(xml_path)  # noqa: S320
    except Exception:
        logger.exception("Failed to parse MARC-XML collection file: %s", xml_path)
        return []

    root = tree.getroot()
    records: list[ExtractedRecord] = []

    # Handle both namespaced and non-namespaced XML
    record_elements = root.findall("m:record", NS)
    if not record_elements:
        record_elements = root.findall("record")
    if not record_elements:
        record_elements = root.findall(f"{{{MARC_NS}}}record")

    for elem in record_elements:
        result = _extract_from_element(elem, record_id=None)
        if result is not None:
            records.append(result)

    logger.info("Extracted %d records from %s", len(records), xml_path)
    return records


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _extract_from_element(
    elem: Any, record_id: int | None
) -> ExtractedRecord | None:
    """Extract fields from an lxml ``<record>`` element."""
    try:
        if record_id is None:
            record_id = _get_control_field(elem, "001")
            if record_id is None:
                logger.warning("Record missing controlfield 001, skipping")
                return None
            record_id = int(record_id)

        return ExtractedRecord(
            record_id=record_id,
            title=_get_title(elem),
            authors=_get_authors(elem),
            subjects=_get_subjects(elem),
            summary=_get_field_text(elem, "520"),
            toc=_get_field_text(elem, "505"),
            notes=_get_all_field_texts(elem, "500"),
            series=_get_series(elem),
            edition=_get_field_text(elem, "250"),
            publisher=_get_publisher(elem),
            isbn=_get_isbns(elem),
        )
    except Exception:
        logger.exception("Failed to extract record %s", record_id)
        return None


def _get_control_field(elem: Any, tag: str) -> str | None:
    """Get text of a controlfield by tag."""
    for node in _find_all(elem, "controlfield"):
        if node.get("tag") == tag:
            return (node.text or "").strip()
    return None


def _get_title(elem: Any) -> str:
    """Extract title from field 245, subfields a, b, c."""
    parts: list[str] = []
    for df in _find_datafields(elem, "245"):
        for code in ("a", "b"):
            val = _get_subfield(df, code)
            if val:
                parts.append(val)
    return " ".join(parts).strip().rstrip("/").strip()


def _get_authors(elem: Any) -> list[str]:
    """Extract authors from fields 100, 110, 700, 710."""
    authors: list[str] = []
    for tag in ("100", "110", "700", "710"):
        for df in _find_datafields(elem, tag):
            name = _get_subfield(df, "a")
            if name:
                # Remove trailing commas/periods
                name = name.strip().rstrip(",").rstrip(".")
                if name:
                    authors.append(name)
    return authors


def _get_subjects(elem: Any) -> list[str]:
    """Extract subjects from all 6XX fields."""
    subjects: list[str] = []
    for df in _find_all(elem, "datafield"):
        tag = df.get("tag", "")
        if tag.startswith("6"):
            parts: list[str] = []
            for sf in _find_all(df, "subfield"):
                code = sf.get("code", "")
                text = (sf.text or "").strip()
                if text and code not in ("0", "2", "4"):
                    parts.append(text)
            if parts:
                subject = " -- ".join(parts)
                subjects.append(subject)
    return subjects


def _get_series(elem: Any) -> list[str]:
    """Extract series from fields 490 and 830."""
    series: list[str] = []
    for tag in ("490", "830"):
        for df in _find_datafields(elem, tag):
            parts: list[str] = []
            for code in ("a", "v"):
                val = _get_subfield(df, code)
                if val:
                    parts.append(val)
            if parts:
                series.append(" ".join(parts))
    return series


def _get_publisher(elem: Any) -> str:
    """Extract publisher from field 264 (preferred) or 260."""
    for tag in ("264", "260"):
        for df in _find_datafields(elem, tag):
            parts: list[str] = []
            for code in ("a", "b", "c"):
                val = _get_subfield(df, code)
                if val:
                    parts.append(val)
            if parts:
                return " ".join(parts).strip()
    return ""


def _get_isbns(elem: Any) -> list[str]:
    """Extract ISBNs from field 020."""
    isbns: list[str] = []
    for df in _find_datafields(elem, "020"):
        val = _get_subfield(df, "a")
        if val:
            # Take just the ISBN portion (before any parenthetical qualifier)
            isbn = val.split("(")[0].strip()
            if isbn:
                isbns.append(isbn)
    return isbns


def _get_field_text(elem: Any, tag: str) -> str:
    """Get concatenated subfield text for the first occurrence of a datafield."""
    for df in _find_datafields(elem, tag):
        val = _get_subfield(df, "a")
        if val:
            return val.strip()
    return ""


def _get_all_field_texts(elem: Any, tag: str) -> list[str]:
    """Get subfield $a text for all occurrences of a datafield."""
    texts: list[str] = []
    for df in _find_datafields(elem, tag):
        val = _get_subfield(df, "a")
        if val:
            texts.append(val.strip())
    return texts


def _find_datafields(elem: Any, tag: str) -> list[Any]:
    """Find all datafield elements with the given tag."""
    results: list[Any] = []
    for df in _find_all(elem, "datafield"):
        if df.get("tag") == tag:
            results.append(df)
    return results


def _get_subfield(datafield: Any, code: str) -> str | None:
    """Get text of a subfield by code."""
    for sf in _find_all(datafield, "subfield"):
        if sf.get("code") == code:
            text = (sf.text or "").strip()
            return text if text else None
    return None


def _find_all(elem: Any, local_name: str) -> list[Any]:
    """Find child elements by local name, handling namespaces."""
    # Try namespaced first
    results = elem.findall(f"m:{local_name}", NS)
    if results:
        return results
    # Try non-namespaced
    results = elem.findall(local_name)
    if results:
        return results
    # Try full namespace URI
    results = elem.findall(f"{{{MARC_NS}}}{local_name}")
    return results


# ------------------------------------------------------------------
# Language detection
# ------------------------------------------------------------------


def detect_language(marc_xml: str | bytes) -> str:
    """Detect the language code from a MARC-XML record.

    Checks MARC 041$a first (explicit language code), then falls back to
    controlfield 008 positions 35-37. Returns a 3-letter MARC language
    code (e.g. ``"eng"``, ``"spa"``, ``"fre"``), or ``"und"``
    (undetermined) if no language can be detected.
    """
    try:
        if isinstance(marc_xml, str):
            marc_xml = marc_xml.encode("utf-8")
        root = etree.fromstring(marc_xml)
        return _detect_language_from_element(root)
    except Exception:
        logger.debug("Failed to detect language from MARC-XML", exc_info=True)
        return "und"


def _detect_language_from_element(elem: Any) -> str:
    """Detect language from a parsed MARC element.

    Priority: 041$a > 008 positions 35-37.
    """
    # Try 041$a (explicit language code)
    for df in _find_datafields(elem, "041"):
        code = _get_subfield(df, "a")
        if code and len(code) == 3 and code.isalpha():
            return code.lower()

    # Fall back to 008 positions 35-37
    cf008 = _get_control_field(elem, "008")
    if cf008 and len(cf008) >= 38:
        lang = cf008[35:38].strip()
        if len(lang) == 3 and lang.isalpha():
            return lang.lower()

    return "und"
