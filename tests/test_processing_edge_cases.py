import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pymupdf

from backend.processing.pdf_reader import read_pdf
from backend.processing.readers import (
    dataframe_to_result,
    optional_number,
    optional_text,
    read_csv,
    read_excel,
)


class ProcessingEdgeCaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_empty_csv(self):
        path = self.directory / "empty.csv"
        path.write_text("", encoding="utf-8")

        result = read_csv(path)

        self.assertEqual(len(result.records), 0)
        self.assertTrue(result.errors)

    def test_csv_with_headers_but_no_rows(self):
        path = self.directory / "headers.csv"
        path.write_text(
            "Supplier,Activity,Quantity,Unit\n",
            encoding="utf-8",
        )

        result = read_csv(path)

        self.assertEqual(len(result.records), 0)
        self.assertTrue(result.errors)

    def test_corrupted_csv(self):
        path = self.directory / "corrupted.csv"
        path.write_bytes(b"\x00\x00\x00\x00")

        result = read_csv(path)

        self.assertTrue(result.errors)
        self.assertEqual(len(result.records), 0)

    def test_empty_excel_file(self):
        path = self.directory / "empty.xlsx"
        path.write_bytes(b"")

        result = read_excel(path)

        self.assertTrue(result.errors)
        self.assertEqual(len(result.records), 0)

    def test_corrupted_excel_file(self):
        path = self.directory / "corrupted.xlsx"
        path.write_text("This is not an Excel workbook", encoding="utf-8")

        result = read_excel(path)

        self.assertTrue(result.errors)
        self.assertEqual(len(result.records), 0)

    def test_corrupted_pdf_file(self):
        path = self.directory / "corrupted.pdf"
        path.write_text("This is not a PDF", encoding="utf-8")

        result = read_pdf(path)

        self.assertTrue(result.errors)
        self.assertEqual(len(result.records), 0)

    def test_empty_pdf_document(self):
        path = self.directory / "empty.pdf"

        with pymupdf.open() as document:
            document.new_page()
            document.save(path)

        result = read_pdf(path)

        self.assertTrue(result.errors)
        self.assertEqual(len(result.records), 0)
    def test_pdf_without_extractable_text(self):
        path = self.directory / "image_only.pdf"

        with pymupdf.open() as document:
            document.new_page()
            document.save(path)

        result = read_pdf(path)

        self.assertTrue(result.errors)
        self.assertEqual(len(result.records), 0)
        self.assertIn("OCR", result.errors[0])

    def test_invalid_numeric_values(self):
        invalid_values = [
            "NaN",
            "Infinity",
            "-Infinity",
            "abc",
            "",
            "N/A",
            True,
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                self.assertIsNone(optional_number(value))

    def test_valid_numeric_formats(self):
        cases = {
            "1,200": 1200.0,
            "(250)": -250.0,
            "12.5": 12.5,
            "-10": -10.0,
            "1.5e2": 150.0,
        }

        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(optional_number(value), expected)

    def test_missing_text_values(self):
        for value in [
            None,
            "",
            " ",
            "N/A",
            "not available",
            "null",
            "-",
        ]:
            with self.subTest(value=value):
                self.assertIsNone(optional_text(value))

    def test_blank_rows_are_skipped(self):
        dataframe = pd.DataFrame({
            "Supplier": ["Example Energy", None],
            "Activity": ["Electricity", None],
            "Quantity": [100, None],
            "Unit": ["kWh", None],
        })

        result = dataframe_to_result(
            dataframe,
            source_filename="sample.csv",
            source_type="csv",
        )

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].supplier, "Example Energy")

    def test_missing_file_errors(self):
        csv_result = read_csv(self.directory / "missing.csv")
        excel_result = read_excel(self.directory / "missing.xlsx")
        pdf_result = read_pdf(self.directory / "missing.pdf")

        self.assertTrue(csv_result.errors)
        self.assertTrue(excel_result.errors)
        self.assertTrue(pdf_result.errors)

    @patch("pytesseract.image_to_string", side_effect=RuntimeError("OCR failed"))
    def test_ocr_failure_is_mockable(self, mock_ocr):
        with self.assertRaises(RuntimeError):
            import pytesseract

            pytesseract.image_to_string(object())

        mock_ocr.assert_called_once()


if __name__ == "__main__":
    unittest.main()

