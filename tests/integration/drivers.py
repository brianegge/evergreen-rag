"""Custom testplan drivers for evergreen-rag integration tests."""

import json
import subprocess
import time
from pathlib import Path

from testplan.testing.multitest.driver import Driver

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


class PostgresDriver(Driver):
    """Driver that manages a PostgreSQL+pgvector instance via docker compose."""

    def __init__(self, name: str = "postgres", **kwargs):
        super().__init__(name=name, **kwargs)
        self._host = "localhost"
        self._port = 5432

    @property
    def connection_string(self) -> str:
        return f"postgresql://evergreen:evergreen@{self._host}:{self._port}/evergreen"

    def starting(self):
        super().starting()
        subprocess.run(
            ["podman-compose", "up", "-d", "db"],
            check=True,
            capture_output=True,
        )
        self._wait_for_ready()
        self._init_schema()
        self._load_sample_marc()

    def stopping(self):
        self._cleanup_test_data()
        subprocess.run(
            ["podman-compose", "stop", "db"],
            check=True,
            capture_output=True,
        )
        super().stopping()

    def _wait_for_ready(self, timeout: int = 30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = subprocess.run(
                ["podman", "exec", "evergreen-rag_db_1", "pg_isready", "-U", "evergreen"],
                capture_output=True,
            )
            if result.returncode == 0:
                return
            time.sleep(1)
        raise TimeoutError("PostgreSQL did not become ready")

    def _init_schema(self):
        """Run init-db.sql to create schemas and tables."""
        init_sql = SCRIPTS_DIR / "init-db.sql"
        subprocess.run(
            [
                "podman", "exec", "-i", "evergreen-rag_db_1",
                "psql", "-U", "evergreen", "-d", "evergreen",
            ],
            input=init_sql.read_bytes(),
            check=True,
            capture_output=True,
        )

    def _load_sample_marc(self):
        """Load sample MARC XML records into biblio.record_entry."""
        from lxml import etree

        marc_path = FIXTURES_DIR / "marc" / "sample_records.xml"
        tree = etree.parse(str(marc_path))  # noqa: S320
        root = tree.getroot()
        ns = {"m": "http://www.loc.gov/MARC21/slim"}

        records = root.findall("m:record", ns)
        if not records:
            records = root.findall("{http://www.loc.gov/MARC21/slim}record")

        rows = []
        for elem in records:
            # Get record ID from controlfield 001
            cf = elem.find("m:controlfield[@tag='001']", ns)
            if cf is None:
                cf = elem.find("{http://www.loc.gov/MARC21/slim}controlfield[@tag='001']")
            if cf is None:
                continue
            record_id = int(cf.text.strip())
            marc_xml = etree.tostring(elem, encoding="unicode")
            rows.append((record_id, marc_xml))

        import psycopg

        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Clear existing test data
                cur.execute("DELETE FROM rag.biblio_embedding")
                cur.execute("DELETE FROM rag.ingest_log")
                cur.execute("DELETE FROM biblio.record_entry")
                for record_id, marc_xml in rows:
                    cur.execute(
                        "INSERT INTO biblio.record_entry (id, marc) VALUES (%s, %s) "
                        "ON CONFLICT (id) DO UPDATE SET marc = EXCLUDED.marc",
                        (record_id, marc_xml),
                    )
            conn.commit()

    def _cleanup_test_data(self):
        """Remove test data so runs are idempotent."""
        try:
            import psycopg

            with psycopg.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM rag.biblio_embedding")
                    cur.execute("DELETE FROM rag.ingest_log")
                    cur.execute("DELETE FROM biblio.record_entry")
                conn.commit()
        except Exception:
            pass  # Best-effort cleanup


class EmbeddingServiceDriver(Driver):
    """Driver that manages an Ollama instance and ensures the embedding model is loaded."""

    def __init__(
        self,
        name: str = "embedding",
        model: str = "nomic-embed-text",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._model = model
        self._host = "localhost"
        self._port = 11434

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def model(self) -> str:
        return self._model

    def starting(self):
        super().starting()
        subprocess.run(
            ["podman-compose", "up", "-d", "ollama"],
            check=True,
            capture_output=True,
        )
        self._wait_for_ready()
        self._pull_model()

    def stopping(self):
        subprocess.run(
            ["podman-compose", "stop", "ollama"],
            check=True,
            capture_output=True,
        )
        super().stopping()

    def _wait_for_ready(self, timeout: int = 60):
        import httpx

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            time.sleep(2)
        raise TimeoutError("Ollama did not become ready")

    def _pull_model(self):
        import httpx

        resp = httpx.get(f"{self.base_url}/api/tags", timeout=10)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(self._model in m for m in models):
            subprocess.run(
                ["podman", "exec", "evergreen-rag_ollama_1", "ollama", "pull", self._model],
                check=True,
                timeout=300,
            )


class GenerationServiceDriver(Driver):
    """Driver that ensures a small LLM model is available in Ollama for generation tests."""

    def __init__(
        self,
        name: str = "generation",
        model: str = "tinyllama",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._model = model
        self._host = "localhost"
        self._port = 11434

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def model(self) -> str:
        return self._model

    def starting(self):
        super().starting()
        # Ollama should already be running from EmbeddingServiceDriver
        self._wait_for_ready()
        self._pull_model()

    def stopping(self):
        super().stopping()

    def _wait_for_ready(self, timeout: int = 60):
        import httpx

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            time.sleep(2)
        raise TimeoutError("Ollama did not become ready for generation")

    def _pull_model(self):
        import httpx

        resp = httpx.get(f"{self.base_url}/api/tags", timeout=10)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(self._model in m for m in models):
            subprocess.run(
                [
                    "podman", "exec", "evergreen-rag_ollama_1",
                    "ollama", "pull", self._model,
                ],
                check=True,
                timeout=600,
            )


def load_quality_queries() -> list[dict]:
    """Load quality query test cases from fixtures."""
    path = FIXTURES_DIR / "quality_queries.json"
    with open(path) as f:
        return json.load(f)
