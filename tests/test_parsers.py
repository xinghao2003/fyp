import unittest
import os
import pandas as pd
from src.data_ingestion.parsers import parse_csv_generic, parse_alpha_vantage_csv

test_csv_content = """date,open,high,low,close,volume
2024-01-01,100,110,90,105,10000
2024-01-02,105,115,95,110,15000
"""

test_alpha_vantage_content = """date,1. open,2. high,3. low,4. close,5. volume
2025-05-16,212.36,212.57,209.77,211.26,54737850.0
2025-05-15,210.95,212.96,209.54,211.45,45029473.0
"""

test_csv_path = "test_generic.csv"
test_alpha_vantage_path = "test_alpha_vantage.csv"


class TestParsers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(test_csv_path, 'w') as f:
            f.write(test_csv_content)
        with open(test_alpha_vantage_path, 'w') as f:
            f.write(test_alpha_vantage_content)

    @classmethod
    def tearDownClass(cls):
        os.remove(test_csv_path)
        os.remove(test_alpha_vantage_path)

    def test_parse_csv_generic(self):
        df = parse_csv_generic(
            test_csv_path, date_col='date', index_col='date')
        self.assertEqual(list(df.columns), [
                         'open', 'high', 'low', 'close', 'volume'])
        self.assertEqual(len(df), 2)
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))

    def test_parse_alpha_vantage_csv(self):
        df = parse_alpha_vantage_csv(test_alpha_vantage_path)
        self.assertEqual(list(df.columns), [
                         'open', 'high', 'low', 'close', 'volume'])
        self.assertEqual(len(df), 2)
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))


if __name__ == '__main__':
    unittest.main()
