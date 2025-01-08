import unittest
import pymysql
from SQL_STOCKDATABASE.FinancialDatabase import FinancialDatabase


class TestFinancialDatabase(unittest.TestCase):

    def setUp(self):
        """
        Set up a clean environment for testing.
        """
        self.host = "localhost"
        self.user = "root"
        self.password = "314159"
        self.port = 3306
        self.database_name = "test_financial_db"

        # Initialize the database
        self.db = FinancialDatabase(
            host=self.host,
            user=self.user,
            password=self.password,
            database_name=self.database_name
        )

    def tearDown(self):
        """
        Clean up by dropping the test database after each test.
        """
        with self.db.connection.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS {self.database_name};")
        self.db.close()

    def test_database_creation(self):
        """
        Test that the database is created successfully.
        """
        with self.db.connection.cursor() as cursor:
            cursor.execute(f"SHOW DATABASES LIKE '{self.database_name}';")
            result = cursor.fetchone()
        self.assertIsNotNone(result, "The database should exist.")

    def test_table_creation(self):
        """
        Test that all required tables are created successfully.
        """
        expected_tables = {"companies", "prices", "balance_sheets"}
        with self.db.connection.cursor() as cursor:
            cursor.execute("SHOW TABLES;")
            tables = {row[0] for row in cursor.fetchall()}
        self.assertTrue(expected_tables.issubset(tables), "All required tables should be created.")

    def test_get_company_id(self):
        """
        Test that company IDs are retrieved correctly.
        """
        # Insert a test company
        with self.db.connection.cursor() as cursor:
            cursor.execute("INSERT INTO companies (ticker) VALUES ('TEST');")
        self.db.connection.commit()

        # Retrieve company ID
        company_id = self.db.get_company_id("TEST")
        self.assertIsNotNone(company_id, "The company ID should be retrieved.")
        self.assertGreater(company_id, 0, "The company ID should be a positive integer.")

    def test_upsert_data_prices(self):
        """
        Test inserting and updating price data.
        """
        # Insert a test company
        with self.db.connection.cursor() as cursor:
            cursor.execute("INSERT INTO companies (ticker) VALUES ('TEST');")
        self.db.connection.commit()

        # Upsert price data
        data = [
            {"date": "2023-12-31", "field": "close", "value": 150.25},
            {"date": "2023-12-31", "field": "open", "value": 148.50}
        ]
        self.db.upsert_data("TEST", "prices", data, is_price_data=True)

        # Verify insertion
        with self.db.connection.cursor() as cursor:
            cursor.execute("SELECT date, field, value FROM prices;")
            result = cursor.fetchall()
            result = [
                (row[0].strftime("%Y-%m-%d"), row[1], float(row[2])) for row in result
            ]
        self.assertEqual(len(result), 2, "Two price records should be inserted.")
        self.assertIn(("2023-12-31", "close", 150.25), result)
        self.assertIn(("2023-12-31", "open", 148.50), result)

    def test_upsert_data_balance_sheets(self):
        """
        Test inserting and updating balance sheet data.
        """
        # Insert a test company
        with self.db.connection.cursor() as cursor:
            cursor.execute("INSERT INTO companies (ticker) VALUES ('TEST');")
        self.db.connection.commit()

        # Upsert balance sheet data
        data = [
            {"period": "2023-12-31", "field": "total_assets", "value": 5100000.00, "frequency": "annual"},
            {"period": "2023-12-31", "field": "total_liabilities", "value": 2000000.00, "frequency": "annual"}
        ]
        self.db.upsert_data("TEST", "balance_sheets", data)

        # Verify insertion
        with self.db.connection.cursor() as cursor:
            cursor.execute("SELECT period, field, value, frequency FROM balance_sheets;")
            result = cursor.fetchall()
            result = [
                (row[0].strftime("%Y-%m-%d"), row[1], float(row[2]), row[3]) for row in result
            ]
        self.assertEqual(len(result), 2, "Two balance sheet records should be inserted.")
        self.assertIn(("2023-12-31", "total_assets", 5100000.00, "annual"), result)
        self.assertIn(("2023-12-31", "total_liabilities", 2000000.00, "annual"), result)

    def test_get_data(self):
        """
        Test retrieving data from tables.
        """
        # Insert a test company
        with self.db.connection.cursor() as cursor:
            cursor.execute("INSERT INTO companies (ticker) VALUES ('TEST');")
        self.db.connection.commit()

        # Insert data into prices table
        data = [
            {"date": "2023-12-31", "field": "close", "value": 150.25},
            {"date": "2023-12-30", "field": "close", "value": 148.75}
        ]
        self.db.upsert_data("TEST", "prices", data, is_price_data=True)

        # Retrieve data
        df = self.db.get_data("TEST", "prices", "close", start_date="2023-12-30", end_date="2023-12-31")
        self.assertEqual(len(df), 2, "Two price records should be retrieved.")
        self.assertEqual(df["value"].iloc[0], 148.75, "The first value should match.")
        self.assertEqual(df["value"].iloc[1], 150.25, "The second value should match.")

    def test_check_database_updates(self):
        """
        Test the check_database_updates method for identifying specific tables/frequencies needing updates.
        """
        # Insert test data
        with self.db.connection.cursor() as cursor:
            # Insert a test company
            cursor.execute("INSERT INTO companies (ticker) VALUES ('TEST');")
            company_id = cursor.lastrowid

            # Insert daily data
            cursor.execute("INSERT INTO prices (company_id, date, field, value) VALUES (%s, %s, %s, %s);",
                           (company_id, "2023-12-29", "close", 150.25))  # Latest date is Friday

            # Insert quarterly data
            cursor.execute("""
                        INSERT INTO balance_sheets (company_id, period, field, value, frequency)
                        VALUES (%s, %s, %s, %s, %s);
                    """, (company_id, "2023-09-30", "total_assets", 5100000.00, "quarterly"))

            # Insert annual data
            cursor.execute("""
                        INSERT INTO balance_sheets (company_id, period, field, value, frequency)
                        VALUES (%s, %s, %s, %s, %s);
                    """, (company_id, "2022-12-31", "total_assets", 10000000.00, "annual"))

        self.db.connection.commit()

        tickers = ["TEST"]
        start_date = "2023-01-01"
        end_date = "2024-01-01"  # Sunday, adjusted to the latest working day (Friday, 2023-12-29)

        updates = self.db.check_database_updates(tickers, start_date, end_date)

        # Verify results
        self.assertIn(("prices", "daily"), updates["TEST"], "Daily prices should need updates.")
        self.assertIn(("balance_sheets", "quarterly"), updates["TEST"], "Quarterly balance sheets should need updates.")
        self.assertIn(("balance_sheets", "annual"), updates["TEST"], "Annual balance sheets should need updates.")


def suite():
    """
    Custom test suite to define the execution order of tests.
    """
    suite = unittest.TestSuite()

    # Add tests in the desired order
    suite.addTest(TestFinancialDatabase('test_database_creation'))
    suite.addTest(TestFinancialDatabase('test_table_creation'))
    suite.addTest(TestFinancialDatabase('test_get_company_id'))
    suite.addTest(TestFinancialDatabase('test_upsert_data_prices'))
    suite.addTest(TestFinancialDatabase('test_upsert_data_balance_sheets'))
    suite.addTest(TestFinancialDatabase('test_get_data_prices'))
    suite.addTest(TestFinancialDatabase('test_check_database_updates'))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
