"""Dataset checks that run without fitting a model."""
import csv
import unittest
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "spam.csv"


class SpamDataTests(unittest.TestCase):
    def test_label_and_message_columns(self):
        with DATA.open(newline="", encoding="latin-1") as f:
            reader = csv.DictReader(f)
            self.assertTrue({"v1", "v2"}.issubset(reader.fieldnames or []))
            rows = list(reader)
        self.assertGreater(len(rows), 0)
        labels = {r["v1"] for r in rows}
        self.assertIn("ham", labels)
        self.assertIn("spam", labels)
        self.assertTrue(all(r["v2"].strip() for r in rows))


if __name__ == "__main__":
    unittest.main()
