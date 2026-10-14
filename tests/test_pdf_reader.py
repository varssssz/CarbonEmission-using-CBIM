
import tempfile
import unittest
from pathlib import Path

import pymupdf

from backend.processing.pdf_reader import read_pdf


class PDFReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_extracts_text_and_page_number(self):
        path = self.directory / "report.pdf"

        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text(
                (72, 72),
                "Supplier: Example Energy Ltd\n"
                "Activity: Purchased electricity\n"
                "Quantity: 1200 kWh",
            )
            document.save(path)

        result = read_pdf(path)

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].page_number, 1)
        self.assertEqual(result.records[0].source_filename, "report.pdf")
        self.assertIn(
            "Example Energy Ltd",
            result.records[0].raw_fields["page_text"],
        )

    def test_missing_file(self):
        result = read_pdf(self.directory / "missing.pdf")

        self.assertTrue(result.errors)
        self.assertEqual(result.records, [])

    def test_wrong_extension(self):
        path = self.directory / "report.txt"
        path.write_text("Not a PDF", encoding="utf-8")

        result = read_pdf(path)

        self.assertTrue(result.errors)
        self.assertEqual(result.records, [])


if __name__ == "__main__":
    unittest.main()
