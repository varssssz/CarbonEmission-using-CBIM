
import json
import unittest
from pathlib import Path

from backend.processing.readers import read_csv


ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT_DIRECTORY / "data" / "sample" / "emissions_activity.csv"
EXPECTED_FILE = ROOT_DIRECTORY / "data" / "expected" / "emissions_activity_expected.json"


class ProcessingIntegrationTests(unittest.TestCase):
    def test_sample_csv_matches_expected_records(self):
        result = read_csv(SAMPLE_FILE)

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.records), 3)

        with EXPECTED_FILE.open("r", encoding="utf-8") as file:
            expected_records = json.load(file)

        actual_records = [
            record.model_dump(
                include={
                    "supplier",
                    "reporting_period",
                    "activity",
                    "quantity",
                    "unit",
                    "emissions_value",
                }
            )
            for record in result.records
        ]

        self.assertEqual(actual_records, expected_records)

    def test_sample_records_preserve_source_information(self):
        result = read_csv(SAMPLE_FILE)

        self.assertEqual(len(result.records), 3)

        for record in result.records:
            self.assertEqual(record.source_filename, SAMPLE_FILE.name)
            self.assertEqual(record.source_type, "csv")
            self.assertIsNotNone(record.row_number)
            self.assertTrue(record.raw_fields)


if __name__ == "__main__":
    unittest.main()
