import unittest
import pymysql
from SQL_STOCKDATABASE.FinancialDatabase import FinancialDatabase


class TestFinancialDatabaseInit(unittest.TestCase):

    def setUp(self):
        """
        Setup before each test. Connects to the MySQL server to prepare for testing.
        """
        self.host = "localhost"
        self.user = "root"
        self.password = "314159"
        self.port = 3306

    def tearDown(self):
        """
        Cleanup after each test. Removes test databases if they exist.
        """
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            port=self.port
        )
        with connection.cursor() as cursor:
            # Drop all test databases
            cursor.execute("DROP DATABASE IF EXISTS test_financial_db1;")
            cursor.execute("DROP DATABASE IF EXISTS test_financial_db2;")
            cursor.execute("DROP DATABASE IF EXISTS test_financial_db3;")
        connection.close()

    def test_create_new_database(self):
        """
        Test that a new database is created successfully.
        """
        db_name = "test_financial_db1"
        db = FinancialDatabase(
            host=self.host,
            user=self.user,
            password=self.password,
            database_name=db_name,
            port=self.port
        )
        with db.connection.cursor() as cursor:
            cursor.execute(f"SHOW DATABASES LIKE '{db_name}';")
            result = cursor.fetchone()
        self.assertIsNotNone(result, f"Database '{db_name}' should be created.")
        db.close()

    def test_connect_to_existing_database(self):
        """
        Test that the class connects to an existing database.
        """
        db_name = "test_financial_db2"

        # Manually create the database
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            port=self.port
        )
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {db_name};")
        connection.close()

        # Initialize the class and connect
        db = FinancialDatabase(
            host=self.host,
            user=self.user,
            password=self.password,
            database_name=db_name,
            port=self.port
        )
        with db.connection.cursor() as cursor:
            cursor.execute(f"SELECT DATABASE();")
            result = cursor.fetchone()
        self.assertEqual(result[0], db_name, f"Should connect to existing database '{db_name}'.")
        db.close()

    def test_invalid_database_name(self):
        """
        Test that the class raises an error for invalid database names.
        """
        db_name = "Invalid_DB"  # Contains capital letters, which are not allowed.
        with self.assertRaises(ValueError) as context:
            FinancialDatabase(
                host=self.host,
                user=self.user,
                password=self.password,
                database_name=db_name,
                port=self.port
            )
        self.assertIn("must not contain any capital letter", str(context.exception))


if __name__ == "__main__":
    unittest.main()
