"""Models for MARC record text extraction."""

from pydantic import BaseModel


class ExtractedRecord(BaseModel):
    """Text extracted from a MARC bibliographic record, ready for embedding."""

    record_id: int
    title: str
    authors: list[str] = []
    subjects: list[str] = []
    summary: str = ""
    toc: str = ""
    notes: list[str] = []
    series: list[str] = []
    edition: str = ""
    publisher: str = ""
    isbn: list[str] = []
    format: str = ""

    def to_embedding_text(self) -> str:
        """Concatenate fields into a single string for embedding."""
        parts = [self.title]
        if self.authors:
            parts.append("by " + ", ".join(self.authors))
        if self.subjects:
            parts.append("Subjects: " + "; ".join(self.subjects))
        if self.summary:
            parts.append(self.summary)
        if self.toc:
            parts.append("Contents: " + self.toc)
        if self.notes:
            parts.append(" ".join(self.notes))
        if self.series:
            parts.append("Series: " + ", ".join(self.series))
        if self.edition:
            parts.append(self.edition)
        if self.publisher:
            parts.append(self.publisher)
        return "\n".join(parts)
