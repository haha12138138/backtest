import unittest
import pymysql
import pandas as pd
from SQL_STOCKDATABASE.FinancialDatabase import FinancialDatabase


class TestFinancialDatabaseMethods(unittest.TestCase):

    def setUp(self):
        """
        Setup before each test. Initializes a test database and table structure.
        """
        self.host = "localhost"
        self.user = "root"
        self.password = "314159"
        self.port = 3306
        self.database_name = "test_financial_db_methods"

        # Initialize the database
        self.db = FinancialDatabase(
            host=self.host,
            user=self.user,
            password=self.password,
            database_name=self.database_name,
            port=self.port
        )

        # Set up table structure
        with self.db.connection.cursor() as cursor:
            # Create companies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(10) UNIQUE NOT NULL
                );
            """)

            # Insert a test company
            cursor.execute("INSERT IGNORE INTO companies (ticker) VALUES ('TEST');")

            # Create prices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    company_id INT,
                    date DATE NOT NULL,
                    field VARCHAR(50) NOT NULL,
                    value DOUBLE,
                    UNIQUE (company_id, date, field),
                    FOREIGN KEY (company_id) REFERENCES companies(id)
                );
            """)

            # Create balance_sheets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS balance_sheets (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    company_id INT,
                    period DATE NOT NULL,
                    field VARCHAR(50) NOT NULL,
                    value DECIMAL(20, 6),
                    frequency ENUM('annual', 'quarterly') NOT NULL,
                    UNIQUE (company_id, period, field),
                    FOREIGN KEY (company_id) REFERENCES companies(id)
                );
            """)
        self.db.connection.commit()

    def tearDown(self):
        """
        Cleanup after each test. Removes the test database.
        """
        with self.db.connection.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS {self.database_name};")
        self.db.connection.close()

    def test_get_company_id(self):
        """
        Test retrieving company ID by ticker.
        """
        company_id = self.db.get_company_id("TEST")
        self.assertIsNotNone(company_id, "Company ID should not be None.")
        self.assertGreater(company_id, 0, "Company ID should be a positive integer.")

    def test_upsert_data_prices(self):
        """
        Test inserting and updating price data.
        """
        data = [
            {"date": "2023-12-31", "field": "close", "value": 150.25},
            {"date": "2023-12-31", "field": "open", "value": 148.50}
        ]
        self.db.upsert_data("TEST", "prices", data, is_price_data=True)

        # Verify insertion
        with self.db.connection.cursor() as cursor:
            cursor.execute("SELECT date, field, value FROM prices;")
            result = cursor.fetchall()

        normalized_result = [
            (row[0].strftime("%Y-%m-%d"), row[1], float(row[2])) for row in result
        ]
        self.assertEqual(len(normalized_result), 2, "Two records should be inserted.")
        self.assertIn(("2023-12-31", "close", 150.25), normalized_result)
        self.assertIn(("2023-12-31", "open", 148.50), normalized_result)

    def test_upsert_data_balance_sheets(self):
        """
        Test inserting and updating balance sheet data.
        """
        data = [
            {"period": "2023-12-31", "field": "total_assets", "value": 5100000.00, "frequency": "annual"},
            {"period": "2023-12-31", "field": "total_liabilities", "value": 2000000.00, "frequency": "annual"}
        ]
        self.db.upsert_data("TEST", "balance_sheets", data)

        # Verify insertion
        with self.db.connection.cursor() as cursor:
            cursor.execute("SELECT period, field, value, frequency FROM balance_sheets;")
            result = cursor.fetchall()
        normalized_result = [
            (row[0].strftime("%Y-%m-%d"), row[1], float(row[2]), row[3]) for row in result
        ]
        self.assertEqual(len(normalized_result), 2, "Two records should be inserted.")
        self.assertIn(("2023-12-31", "total_assets", 5100000.00, "annual"), normalized_result)
        self.assertIn(("2023-12-31", "total_liabilities", 2000000.00, "annual"), normalized_result)

    def test_get_data_prices(self):
        """
        Test retrieving price data.
        """
        # Insert test data
        data = [
            {"date": "2023-12-31", "field": "close", "value": 150.25},
            {"date": "2023-12-30", "field": "close", "value": 148.75}
        ]
        self.db.upsert_data("TEST", "prices", data, is_price_data=True)

        # Retrieve data
        df = self.db.get_data("TEST", "prices", "close", start_date="2023-12-30", end_date="2023-12-31")
        self.assertIsInstance(df, pd.DataFrame, "Result should be a Pandas DataFrame.")
        self.assertEqual(len(df), 2, "Two records should be retrieved.")
        self.assertListEqual(list(df["value"]), [148.75, 150.25], "Values should match the inserted data.")




def suite():
    """
    Custom test suite to define the execution order of tests.
    """
    suite = unittest.TestSuite()

    # Add tests in the desired order
    suite.addTest(TestFinancialDatabaseMethods('test_get_company_id'))
    suite.addTest(TestFinancialDatabaseMethods('test_upsert_data_prices'))
    suite.addTest(TestFinancialDatabaseMethods('test_upsert_data_balance_sheets'))
    suite.addTest(TestFinancialDatabaseMethods('test_get_data_prices'))

    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

