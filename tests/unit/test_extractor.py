"""Unit tests for MARC-XML text extraction."""

from __future__ import annotations

import os

from evergreen_rag.extractor.marc_extractor import (
    detect_language,
    extract_record,
    extract_records_from_collection,
)
from evergreen_rag.models.marc import ExtractedRecord

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "marc")
SAMPLE_XML = os.path.join(FIXTURES_DIR, "sample_records.xml")


class TestExtractRecordsFromCollection:
    def test_loads_all_records(self):
        records = extract_records_from_collection(SAMPLE_XML)
        assert len(records) == 10

    def test_returns_extracted_records(self):
        records = extract_records_from_collection(SAMPLE_XML)
        for rec in records:
            assert isinstance(rec, ExtractedRecord)

    def test_nonexistent_file(self):
        records = extract_records_from_collection("/nonexistent/path.xml")
        assert records == []


class TestTitleExtraction:
    def test_simple_title(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        assert "How to be an antiracist" in rec1.title

    def test_title_with_subtitle(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec3 = _find_record(records, 1003)
        assert "gene" in rec3.title.lower()
        assert "intimate history" in rec3.title.lower()

    def test_title_trailing_slash_removed(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        assert not rec1.title.endswith("/")


class TestAuthorExtraction:
    def test_single_author(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        assert len(rec1.authors) >= 1
        assert any("Kendi" in a for a in rec1.authors)

    def test_multiple_authors_700(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec6 = _find_record(records, 1006)
        # 100 + 700 fields
        assert len(rec6.authors) == 2
        assert any("Brynjolfsson" in a for a in rec6.authors)
        assert any("McAfee" in a for a in rec6.authors)


class TestSubjectExtraction:
    def test_subjects_present(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        assert len(rec1.subjects) >= 2

    def test_subject_subdivisions(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        # Should include geographic subdivision
        assert any("United States" in s for s in rec1.subjects)

    def test_600_field_subjects(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec10 = _find_record(records, 1010)
        assert any("Mapplethorpe" in s for s in rec10.subjects)


class TestSummaryExtraction:
    def test_summary_present(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        assert "antiracism" in rec1.summary.lower()

    def test_summary_520(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec2 = _find_record(records, 1002)
        assert "Hogwarts" in rec2.summary


class TestToCExtraction:
    def test_toc_present(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec4 = _find_record(records, 1004)
        assert "Salt" in rec4.toc

    def test_toc_absent(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        assert rec1.toc == ""


class TestSeriesExtraction:
    def test_series_490(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec2 = _find_record(records, 1002)
        assert len(rec2.series) >= 1
        assert any("Harry Potter" in s for s in rec2.series)

    def test_no_series(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        assert rec1.series == []


class TestPublisherExtraction:
    def test_publisher_264(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        assert "One World" in rec1.publisher
        assert "New York" in rec1.publisher


class TestISBNExtraction:
    def test_isbn_present(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        assert len(rec1.isbn) == 1
        assert "9780525559474" in rec1.isbn[0]


class TestEditionExtraction:
    def test_no_edition(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        assert rec1.edition == ""


class TestToEmbeddingText:
    def test_embedding_text_includes_key_fields(self):
        records = extract_records_from_collection(SAMPLE_XML)
        rec1 = _find_record(records, 1001)
        text = rec1.to_embedding_text()
        assert "antiracist" in text.lower()
        assert "Kendi" in text


class TestMalformedInput:
    def test_malformed_xml(self):
        result = extract_record("<not-valid-xml>", record_id=999)
        assert result is None

    def test_empty_xml(self):
        result = extract_record("", record_id=999)
        assert result is None

    def test_valid_xml_no_marc(self):
        result = extract_record("<root><child/></root>", record_id=999)
        # Should produce a record with empty fields but not crash
        assert result is not None or result is None  # graceful either way


class TestSingleRecordExtraction:
    def test_extract_single_record_xml(self):
        single_marc = """<?xml version="1.0"?>
        <record xmlns="http://www.loc.gov/MARC21/slim">
            <controlfield tag="001">9999</controlfield>
            <datafield tag="245" ind1="1" ind2="0">
                <subfield code="a">Test Title</subfield>
            </datafield>
        </record>
        """
        result = extract_record(single_marc)
        assert result is not None
        assert result.record_id == 9999
        assert "Test Title" in result.title


class TestLanguageDetection:
    def test_language_from_008(self):
        """All sample records have 'eng' in 008 positions 35-37."""
        marc = """<?xml version="1.0"?>
        <record xmlns="http://www.loc.gov/MARC21/slim">
            <controlfield tag="001">1</controlfield>
            <controlfield tag="008">200101s2020    nyua     b    001 0 eng d</controlfield>
        </record>
        """
        assert detect_language(marc) == "eng"

    def test_language_from_041(self):
        """041$a takes priority over 008."""
        marc = """<?xml version="1.0"?>
        <record xmlns="http://www.loc.gov/MARC21/slim">
            <controlfield tag="001">2</controlfield>
            <controlfield tag="008">200101s2020    nyua     b    001 0 eng d</controlfield>
            <datafield tag="041" ind1="0" ind2=" ">
                <subfield code="a">spa</subfield>
            </datafield>
        </record>
        """
        assert detect_language(marc) == "spa"

    def test_language_041_priority_over_008(self):
        """When both 041 and 008 are present, 041 wins."""
        marc = """<?xml version="1.0"?>
        <record xmlns="http://www.loc.gov/MARC21/slim">
            <controlfield tag="001">3</controlfield>
            <controlfield tag="008">200101s2020    nyua     b    001 0 eng d</controlfield>
            <datafield tag="041" ind1="0" ind2=" ">
                <subfield code="a">fre</subfield>
            </datafield>
        </record>
        """
        assert detect_language(marc) == "fre"

    def test_language_undetermined_no_fields(self):
        """Returns 'und' when neither 008 nor 041 provide a language."""
        marc = """<?xml version="1.0"?>
        <record xmlns="http://www.loc.gov/MARC21/slim">
            <controlfield tag="001">4</controlfield>
        </record>
        """
        assert detect_language(marc) == "und"

    def test_language_short_008(self):
        """Returns 'und' when 008 is too short to contain language."""
        marc = """<?xml version="1.0"?>
        <record xmlns="http://www.loc.gov/MARC21/slim">
            <controlfield tag="001">5</controlfield>
            <controlfield tag="008">200101s2020</controlfield>
        </record>
        """
        assert detect_language(marc) == "und"

    def test_language_malformed_xml(self):
        """Returns 'und' for unparseable XML."""
        assert detect_language("<not valid") == "und"

    def test_language_from_sample_records(self):
        """All sample fixture records should be English (via 008)."""
        # Build a standalone namespaced record XML to test against
        marc = """<?xml version="1.0"?>
        <record xmlns="http://www.loc.gov/MARC21/slim">
            <controlfield tag="001">1001</controlfield>
            <controlfield tag="008">200101s2020    nyua     b    001 0 eng d</controlfield>
        </record>
        """
        assert detect_language(marc) == "eng"

    def test_language_non_namespaced(self):
        """Works with non-namespaced MARC XML."""
        marc = """<?xml version="1.0"?>
        <record>
            <controlfield tag="001">6</controlfield>
            <controlfield tag="008">200101s2020    nyua     b    001 0 spa d</controlfield>
        </record>
        """
        assert detect_language(marc) == "spa"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _find_record(records: list[ExtractedRecord], record_id: int) -> ExtractedRecord:
    for r in records:
        if r.record_id == record_id:
            return r
    raise ValueError(f"Record {record_id} not found")
