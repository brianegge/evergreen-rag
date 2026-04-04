"""Generate synthetic MARC-XML records for benchmarking.

Usage:
    python -m tests.benchmarks.generate_marc --count 1000 --output /tmp/synthetic.xml
"""

from __future__ import annotations

import argparse
import random
import sys

TITLES = [
    "The Art of Programming",
    "Introduction to Library Science",
    "Modern Database Systems",
    "Principles of Information Retrieval",
    "A History of Cataloging",
    "Digital Libraries and Archives",
    "Foundations of Metadata",
    "Search Engine Design",
    "Natural Language Processing",
    "Machine Learning in Practice",
    "Data Structures and Algorithms",
    "The Web of Knowledge",
    "Understanding Classification Systems",
    "Bibliographic Control in the Digital Age",
    "Open Source Software Development",
    "Network Architecture Fundamentals",
    "Cloud Computing Patterns",
    "Distributed Systems Concepts",
    "Statistical Methods for Data Analysis",
    "Applied Cryptography",
]

AUTHORS = [
    "Smith, John",
    "Johnson, Maria",
    "Williams, Robert",
    "Brown, Patricia",
    "Jones, Michael",
    "Garcia, Ana",
    "Miller, David",
    "Davis, Sarah",
    "Rodriguez, Carlos",
    "Martinez, Elena",
    "Anderson, James",
    "Taylor, Lisa",
    "Thomas, William",
    "Hernandez, Jose",
    "Moore, Jennifer",
]

SUBJECTS = [
    "Computer science",
    "Library science",
    "Information retrieval",
    "Database management",
    "Software engineering",
    "Artificial intelligence",
    "Data mining",
    "Machine learning",
    "Natural language processing",
    "Cataloging",
    "Metadata",
    "Digital preservation",
    "Knowledge management",
    "Information systems",
    "Web technologies",
]

PUBLISHERS = [
    "Academic Press",
    "University Press",
    "Technical Publishing",
    "Open Knowledge Press",
    "Digital Editions",
    "Scholarly Works",
    "Research Publications",
]

CITIES = [
    "New York",
    "London",
    "San Francisco",
    "Boston",
    "Chicago",
    "Toronto",
    "Berlin",
]

SUMMARIES = [
    "A comprehensive guide covering fundamental concepts and advanced topics.",
    "This text provides an in-depth exploration of theory and practice.",
    "An essential reference for students and professionals in the field.",
    "Covers the latest developments and emerging trends.",
    "A practical handbook with real-world examples and case studies.",
    "Explores the intersection of technology and information management.",
    "A thorough examination of methods, tools, and best practices.",
]


def _marc_record_xml(record_id: int) -> str:
    """Generate a single MARC-XML <record> with realistic content."""
    title = random.choice(TITLES)
    subtitle = random.choice(["", " : a practical guide", " : theory and practice", ""])
    author = random.choice(AUTHORS)
    subject1 = random.choice(SUBJECTS)
    subject2 = random.choice(SUBJECTS)
    publisher = random.choice(PUBLISHERS)
    city = random.choice(CITIES)
    year = str(random.randint(1990, 2025))
    summary = random.choice(SUMMARIES)
    isbn = f"978{random.randint(1000000000, 9999999999)}"

    return f"""  <record xmlns="http://www.loc.gov/MARC21/slim">
    <leader>00000nam a2200000 a 4500</leader>
    <controlfield tag="001">{record_id}</controlfield>
    <controlfield tag="005">{year}0101120000.0</controlfield>
    <controlfield tag="008">{year[2:]}0101s{year}    nyu           000 0 eng d</controlfield>
    <datafield tag="020" ind1=" " ind2=" ">
      <subfield code="a">{isbn}</subfield>
    </datafield>
    <datafield tag="100" ind1="1" ind2=" ">
      <subfield code="a">{author}</subfield>
    </datafield>
    <datafield tag="245" ind1="1" ind2="0">
      <subfield code="a">{title}{subtitle}</subfield>
    </datafield>
    <datafield tag="260" ind1=" " ind2=" ">
      <subfield code="a">{city} :</subfield>
      <subfield code="b">{publisher},</subfield>
      <subfield code="c">{year}.</subfield>
    </datafield>
    <datafield tag="520" ind1=" " ind2=" ">
      <subfield code="a">{summary}</subfield>
    </datafield>
    <datafield tag="650" ind1=" " ind2="0">
      <subfield code="a">{subject1}</subfield>
    </datafield>
    <datafield tag="650" ind1=" " ind2="0">
      <subfield code="a">{subject2}</subfield>
    </datafield>
  </record>"""


def generate_collection(count: int) -> str:
    """Generate a MARC-XML collection with *count* records."""
    records = [_marc_record_xml(i + 1) for i in range(count)]
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<collection xmlns="http://www.loc.gov/MARC21/slim">\n'
        + "\n".join(records)
        + "\n</collection>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MARC-XML records")
    parser.add_argument(
        "--count", type=int, default=1000, help="Number of records (default: 1000)"
    )
    parser.add_argument(
        "--output", type=str, default="/tmp/synthetic.xml", help="Output file path"
    )
    args = parser.parse_args()

    xml = generate_collection(args.count)
    with open(args.output, "w") as f:
        f.write(xml)
    print(f"Generated {args.count} MARC-XML records -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
