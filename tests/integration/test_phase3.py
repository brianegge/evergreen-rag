"""Integration tests for Phase 3 features: LLM generation, graceful degradation, benchmarks."""

from testplan import test_plan
from testplan.testing.multitest import MultiTest, testcase, testsuite

from evergreen_rag.embedding.service import EmbeddingService
from evergreen_rag.generation.service import GenerationService
from evergreen_rag.ingest.pipeline import IngestPipeline
from evergreen_rag.models.search import SearchQuery
from evergreen_rag.search.vector_search import VectorSearch
from tests.integration.drivers import (
    EmbeddingServiceDriver,
    GenerationServiceDriver,
    PostgresDriver,
)


def _make_test_client(db_url, ollama_url, embed_model, gen_model=None):
    """Create a FastAPI TestClient with configured services."""
    from starlette.testclient import TestClient

    from evergreen_rag.api.main import create_app

    app = create_app()

    embedding_svc = EmbeddingService(ollama_url=ollama_url, model=embed_model)
    vector_search = VectorSearch(db_url=db_url)
    vector_search.open()

    app.state.embedding_service = embedding_svc
    app.state.vector_search = vector_search

    if gen_model:
        gen_svc = GenerationService(ollama_url=ollama_url, model=gen_model)
        if gen_svc.health_check():
            app.state.generation_service = gen_svc
        else:
            app.state.generation_service = None
    else:
        app.state.generation_service = None

    client = TestClient(app, raise_server_exceptions=False)
    return client, vector_search


@testsuite
class SetupSuite:
    """Ensure embeddings exist before running Phase 3 tests."""

    @testcase
    def test_ingest_sample_records(self, env, result):
        """Run ingest so that search and generation have data to work with."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        model = env.embedding.model

        embedding_svc = EmbeddingService(ollama_url=ollama_url, model=model)
        pipeline = IngestPipeline(
            db_url=db_url,
            embedding_service=embedding_svc,
            batch_size=10,
        )

        stats = pipeline.run(full=True)
        result.log(
            f"Setup ingest: total={stats.total} embedded={stats.embedded} "
            f"failed={stats.failed}"
        )
        result.gt(stats.embedded, 0, "Must have embeddings for Phase 3 tests")


@testsuite
class GenerationSuite:
    """Tests for LLM generation endpoints using a real Ollama model."""

    @testcase
    def test_search_with_generate(self, env, result):
        """POST /search with generate=true returns generated_text in response."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        embed_model = env.embedding.model
        gen_model = env.generation.model

        client, vs = _make_test_client(
            db_url, ollama_url, embed_model, gen_model
        )
        try:
            resp = client.post(
                "/search",
                json={
                    "query": "books about racism and antiracism",
                    "limit": 3,
                    "generate": True,
                },
            )

            result.equal(resp.status_code, 200, "Search with generate should return 200")
            data = resp.json()
            result.gt(data["total"], 0, "Should have search results")
            result.true(
                data["generated_text"] is not None,
                "generated_text should be present when generate=true",
            )
            result.true(
                len(data["generated_text"]) > 10,
                "Generated text should be non-trivial",
            )
            result.log(f"Generated summary length: {len(data['generated_text'])} chars")
            result.log(f"Generated text preview: {data['generated_text'][:200]}...")
        finally:
            vs.close()

    @testcase
    def test_recommend_endpoint(self, env, result):
        """POST /recommend returns reading recommendations text."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        embed_model = env.embedding.model
        gen_model = env.generation.model

        # First get some real search results
        embedding_svc = EmbeddingService(ollama_url=ollama_url, model=embed_model)
        search = VectorSearch(db_url=db_url)

        query_text = "children's fantasy novel"
        query_embedding = embedding_svc.embed_text(query_text)
        search_query = SearchQuery(query=query_text, limit=3)
        sem_response = search.similarity_search(query_embedding, search_query)

        results_payload = [
            {
                "record_id": r.record_id,
                "similarity": r.similarity,
                "chunk_text": r.chunk_text,
            }
            for r in sem_response.results
        ]

        client, vs = _make_test_client(
            db_url, ollama_url, embed_model, gen_model
        )
        try:
            resp = client.post(
                "/recommend",
                json={
                    "query": query_text,
                    "results": results_payload,
                },
            )

            result.equal(resp.status_code, 200, "Recommend should return 200")
            data = resp.json()
            result.true(
                data["recommendations"] is not None,
                "Should return recommendations text",
            )
            result.true(
                len(data["recommendations"]) > 10,
                "Recommendations should be non-trivial",
            )
            result.log(f"Recommendations length: {len(data['recommendations'])} chars")
            result.log(f"Recommendations preview: {data['recommendations'][:200]}...")
        finally:
            vs.close()

    @testcase
    def test_refine_endpoint(self, env, result):
        """POST /refine returns a list of refined query suggestions."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        embed_model = env.embedding.model
        gen_model = env.generation.model

        # Get some real search results
        embedding_svc = EmbeddingService(ollama_url=ollama_url, model=embed_model)
        search = VectorSearch(db_url=db_url)

        query_text = "programming books"
        query_embedding = embedding_svc.embed_text(query_text)
        search_query = SearchQuery(query=query_text, limit=3)
        sem_response = search.similarity_search(query_embedding, search_query)

        results_payload = [
            {
                "record_id": r.record_id,
                "similarity": r.similarity,
                "chunk_text": r.chunk_text,
            }
            for r in sem_response.results
        ]

        client, vs = _make_test_client(
            db_url, ollama_url, embed_model, gen_model
        )
        try:
            resp = client.post(
                "/refine",
                json={
                    "query": query_text,
                    "results": results_payload,
                },
            )

            result.equal(resp.status_code, 200, "Refine should return 200")
            data = resp.json()
            result.true(
                len(data["suggestions"]) > 0,
                "Should return at least one refined query suggestion",
            )
            result.log(f"Refined queries: {data['suggestions']}")
        finally:
            vs.close()

    @testcase
    def test_health_includes_generation(self, env, result):
        """GET /health reports generation_service status."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        embed_model = env.embedding.model
        gen_model = env.generation.model

        client, vs = _make_test_client(
            db_url, ollama_url, embed_model, gen_model
        )
        try:
            resp = client.get("/health")
            result.equal(resp.status_code, 200, "Health should return 200")
            data = resp.json()
            result.true(
                "generation_service" in data["checks"],
                "Health checks should include generation_service",
            )
            result.log(f"Health checks: {data['checks']}")
        finally:
            vs.close()


@testsuite
class GracefulDegradationSuite:
    """Tests that generation features degrade gracefully when unavailable."""

    @testcase
    def test_search_generate_without_generation_service(self, env, result):
        """POST /search with generate=true still returns results when no LLM."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        embed_model = env.embedding.model

        # Explicitly create client without generation service
        client, vs = _make_test_client(
            db_url, ollama_url, embed_model, gen_model=None
        )
        try:
            resp = client.post(
                "/search",
                json={
                    "query": "books about racism",
                    "limit": 3,
                    "generate": True,
                },
            )

            result.equal(
                resp.status_code, 200,
                "Search should still return 200 without generation service",
            )
            data = resp.json()
            result.gt(data["total"], 0, "Should still return search results")
            result.true(
                data["generated_text"] is None,
                "generated_text should be null when generation is unavailable",
            )
            result.log(f"Search returned {data['total']} results without generation")
        finally:
            vs.close()

    @testcase
    def test_recommend_without_generation_service(self, env, result):
        """POST /recommend returns null gracefully when no LLM."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        embed_model = env.embedding.model

        client, vs = _make_test_client(
            db_url, ollama_url, embed_model, gen_model=None
        )
        try:
            resp = client.post(
                "/recommend",
                json={
                    "query": "test",
                    "results": [
                        {
                            "record_id": 1001,
                            "similarity": 0.9,
                            "chunk_text": "test book",
                        },
                    ],
                },
            )

            result.equal(resp.status_code, 200, "Recommend should return 200")
            data = resp.json()
            result.true(
                data["recommendations"] is None,
                "Recommendations should be null without generation service",
            )
        finally:
            vs.close()

    @testcase
    def test_refine_without_generation_service(self, env, result):
        """POST /refine returns empty suggestions when no LLM."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        embed_model = env.embedding.model

        client, vs = _make_test_client(
            db_url, ollama_url, embed_model, gen_model=None
        )
        try:
            resp = client.post(
                "/refine",
                json={
                    "query": "test",
                    "results": [
                        {
                            "record_id": 1001,
                            "similarity": 0.9,
                            "chunk_text": "test book",
                        },
                    ],
                },
            )

            result.equal(resp.status_code, 200, "Refine should return 200")
            data = resp.json()
            result.equal(
                data["suggestions"], [],
                "Suggestions should be empty without generation service",
            )
        finally:
            vs.close()

    @testcase
    def test_search_without_generate_flag(self, env, result):
        """POST /search without generate flag returns results with null generated_text."""
        db_url = env.db.connection_string
        ollama_url = env.embedding.base_url
        embed_model = env.embedding.model
        gen_model = env.generation.model

        client, vs = _make_test_client(
            db_url, ollama_url, embed_model, gen_model
        )
        try:
            resp = client.post(
                "/search",
                json={"query": "books about racism", "limit": 3},
            )

            result.equal(resp.status_code, 200, "Search should return 200")
            data = resp.json()
            result.true(
                data["generated_text"] is None,
                "generated_text should be null when generate flag is not set",
            )
            result.gt(data["total"], 0, "Should still return search results")
        finally:
            vs.close()


@testsuite
class BenchmarkSmokeSuite:
    """Smoke tests verifying benchmark scripts are importable and functional."""

    @testcase
    def test_generate_marc_produces_valid_xml(self, env, result):
        """generate_marc.generate_collection() produces valid MARC-XML."""
        from lxml import etree

        from tests.benchmarks.generate_marc import generate_collection

        xml_str = generate_collection(5)
        result.true(len(xml_str) > 0, "Generated XML should not be empty")

        # Parse as XML to verify validity
        root = etree.fromstring(xml_str.encode("utf-8"))  # noqa: S320
        ns = {"m": "http://www.loc.gov/MARC21/slim"}
        records = root.findall("m:record", ns)

        result.equal(
            len(records), 5,
            "Should generate exactly 5 MARC records",
        )

        # Verify each record has expected structure
        for i, rec in enumerate(records):
            cf001 = rec.find("m:controlfield[@tag='001']", ns)
            result.true(
                cf001 is not None,
                f"Record {i} should have a controlfield 001",
            )
            title = rec.find("m:datafield[@tag='245']", ns)
            result.true(
                title is not None,
                f"Record {i} should have a title field 245",
            )

        result.log(f"Generated {len(records)} valid MARC-XML records")

    @testcase
    def test_benchmark_modules_importable(self, env, result):
        """Benchmark modules can be imported without error."""
        import_errors = []

        try:
            import tests.benchmarks.generate_marc  # noqa: F401
        except ImportError as e:
            import_errors.append(f"generate_marc: {e}")

        try:
            import tests.benchmarks.run_all  # noqa: F401
        except ImportError as e:
            import_errors.append(f"run_all: {e}")

        try:
            import tests.benchmarks.bench_embedding  # noqa: F401
        except ImportError as e:
            import_errors.append(f"bench_embedding: {e}")

        try:
            import tests.benchmarks.bench_ingest  # noqa: F401
        except ImportError as e:
            import_errors.append(f"bench_ingest: {e}")

        try:
            import tests.benchmarks.bench_search  # noqa: F401
        except ImportError as e:
            import_errors.append(f"bench_search: {e}")

        result.equal(
            len(import_errors), 0,
            f"All benchmark modules should import cleanly: {import_errors}",
        )
        result.log("All benchmark modules imported successfully")


def make_multitest():
    return MultiTest(
        name="Phase 3 Integration",
        suites=[
            SetupSuite(),
            GenerationSuite(),
            GracefulDegradationSuite(),
            BenchmarkSmokeSuite(),
        ],
        environment=[
            PostgresDriver(name="db"),
            EmbeddingServiceDriver(name="embedding"),
            GenerationServiceDriver(name="generation"),
        ],
    )


@test_plan(name="Evergreen RAG Phase 3 Integration Tests")
def main(plan):
    plan.add(make_multitest())


if __name__ == "__main__":
    main()
