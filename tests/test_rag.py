
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.rag.ingest import chunk_text, ingest_pdfs
from backend.rag.retriever import search_guidance


class TestChunkText(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n  "), [])

    def test_short_text_returns_one_chunk(self):
        self.assertEqual(chunk_text("Hello world"), ["Hello world"])

    def test_long_text_is_split(self):
        text = "A" * 2500
        chunks = chunk_text(text, chunk_size=1000, overlap=100)

        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk) <= 1000 for chunk in chunks))

    def test_chunks_have_expected_overlap(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_text(text, chunk_size=10, overlap=3)

        self.assertEqual(chunks[0][-3:], chunks[1][:3])
        self.assertEqual(chunks[1][-3:], chunks[2][:3])

    def test_whitespace_is_normalized(self):
        self.assertEqual(
            chunk_text("  Purchased   electricity \n emissions  "),
            ["Purchased electricity emissions"],
        )

    def test_invalid_chunk_size_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=0)

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, overlap=10)

        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, overlap=-1)


class TestIngestion(unittest.TestCase):
    def test_missing_guidance_directory_raises_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_directory = Path(temporary_directory) / "missing"

            with self.assertRaises(FileNotFoundError):
                ingest_pdfs(
                    guidance_dir=missing_directory,
                    database_dir=Path(temporary_directory) / "db",
                )

    def test_empty_guidance_directory_raises_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            guidance_directory = Path(temporary_directory) / "guidance"
            guidance_directory.mkdir()

            with self.assertRaises(FileNotFoundError):
                ingest_pdfs(
                    guidance_dir=guidance_directory,
                    database_dir=Path(temporary_directory) / "db",
                )


class TestGuidanceRetrieval(unittest.TestCase):
    
    def test_negative_max_distance_is_rejected(self):
        with self.assertRaises(ValueError):
            search_guidance("electricity", max_distance=-0.1)

    @patch("backend.rag.retriever.create_collection")
    def test_max_distance_filters_weak_matches(self, mock_create_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "documents": [["Relevant passage", "Weak passage"]],
            "metadatas": [[
                {
                    "source_filename": "Scope 2 Guidance.pdf",
                    "page_number": 12,
                    "chunk_index": 0,
                },
                {
                    "source_filename": "Scope 2 Guidance.pdf",
                    "page_number": 13,
                    "chunk_index": 0,
                },
            ]],
            "distances": [[0.2, 0.9]],
        }
        mock_create_collection.return_value = mock_collection

        results = search_guidance("purchased electricity", max_distance=0.5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Relevant passage")

    @patch("backend.rag.retriever.create_collection")
    def test_duplicate_passages_are_removed(self, mock_create_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "documents": [["Same passage", "Same   passage"]],
            "metadatas": [[
                {
                    "source_filename": "standard.pdf",
                    "page_number": 4,
                    "chunk_index": 0,
                },
                {
                    "source_filename": "standard.pdf",
                    "page_number": 4,
                    "chunk_index": 1,
                },
            ]],
            "distances": [[0.2, 0.3]],
        }
        mock_create_collection.return_value = mock_collection

        results = search_guidance("emissions", n_results=2)

        self.assertEqual(len(results), 1)

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            search_guidance("")

        with self.assertRaises(ValueError):
            search_guidance("   ")

    def test_non_positive_result_count_is_rejected(self):
        with self.assertRaises(ValueError):
            search_guidance("electricity", n_results=0)

        with self.assertRaises(ValueError):
            search_guidance("electricity", n_results=-1)

    @patch("backend.rag.retriever.create_collection")
    def test_empty_collection_returns_no_results(self, mock_create_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_create_collection.return_value = mock_collection

        results = search_guidance("electricity emissions")

        self.assertEqual(results, [])
        mock_collection.query.assert_not_called()

    @patch("backend.rag.retriever.create_collection")
    def test_results_preserve_source_metadata(self, mock_create_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "documents": [["Purchased electricity emissions guidance."]],
            "metadatas": [[
                {
                    "source_filename": "Scope 2 Guidance.pdf",
                    "page_number": 12,
                    "chunk_index": 2,
                }
            ]],
            "distances": [[0.25]],
        }
        mock_create_collection.return_value = mock_collection

        results = search_guidance("purchased electricity")

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["source_filename"], "Scope 2 Guidance.pdf"
        )
        self.assertEqual(results[0]["page_number"], 12)
        self.assertEqual(results[0]["chunk_index"], 2)
        self.assertEqual(results[0]["distance"], 0.25)
        self.assertIn("Purchased electricity", results[0]["text"])

    @patch("backend.rag.retriever.create_collection")
    def test_result_count_is_limited_to_collection_size(
        self, mock_create_collection
    ):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "documents": [["Passage one", "Passage two"]],
            "metadatas": [
                [
                    {
                        "source_filename": "standard.pdf",
                        "page_number": 1,
                        "chunk_index": 0,
                    },
                    {
                        "source_filename": "standard.pdf",
                        "page_number": 2,
                        "chunk_index": 0,
                    },
                ]
            ],
            "distances": [[0.1, 0.2]],
        }
        mock_create_collection.return_value = mock_collection

        results = search_guidance("emissions", n_results=10)

        self.assertEqual(len(results), 2)
        self.assertEqual(
            mock_collection.query.call_args.kwargs["n_results"], 2
        )


if __name__ == "__main__":
    unittest.main()
