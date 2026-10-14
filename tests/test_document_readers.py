
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.processing.readers import read_csv, read_excel


class DocumentReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

        self.dataframe = pd.DataFrame({
            "Supplier Name": ["Example Energy Ltd", "Example Transport"],
            "Reporting Year": [2025, 2025],
            "Activity Type": ["Purchased electricity", "Business travel"],
            "Quantity": [1200, 450],
            "Unit": ["kWh", "km"],
            "CO2e Emissions": [540, 90],
        })

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_csv_reader(self):
        path = self.directory / "activities.csv"
        self.dataframe.to_csv(path, index=False)

        result = read_csv(path)

        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.records[0].supplier, "Example Energy Ltd")
        self.assertEqual(result.records[0].quantity, 1200.0)
        self.assertEqual(result.records[0].unit, "kWh")
        self.assertEqual(result.records[0].emissions_value, 540.0)
        self.assertEqual(result.records[0].row_number, 2)
        self.assertEqual(
            result.records[0].source_filename,
            "activities.csv",
        )

    def test_excel_reader(self):
        path = self.directory / "activities.xlsx"

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.dataframe.to_excel(
                writer,
                sheet_name="Electricity",
                index=False,
            )

        result = read_excel(path)

        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.records[0].sheet_name, "Electricity")
        self.assertEqual(result.records[0].row_number, 2)
        self.assertEqual(result.records[0].activity, "Purchased electricity")

    def test_missing_file(self):
        result = read_csv(self.directory / "missing.csv")

        self.assertEqual(len(result.records), 0)
        self.assertTrue(result.errors)

    def test_invalid_file_extension(self):
        path = self.directory / "activities.txt"
        path.write_text("not a CSV", encoding="utf-8")

        result = read_csv(path)

        self.assertEqual(len(result.records), 0)
        self.assertTrue(result.errors)

    def test_missing_optional_fields(self):
        path = self.directory / "partial.csv"

        pd.DataFrame({
            "Activity": ["Natural gas"],
            "Quantity": [100],
            "Unit": ["m3"],
        }).to_csv(path, index=False)

        result = read_csv(path)

        self.assertEqual(len(result.records), 1)
        self.assertIsNone(result.records[0].supplier)
        self.assertIsNone(result.records[0].emissions_value)
        self.assertEqual(result.records[0].quantity, 100.0)


if __name__ == "__main__":
    unittest.main()
