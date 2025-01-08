import re
from datetime import timedelta, datetime

import pymysql
import pandas as pd
from sqlalchemy import create_engine


class FinancialDatabase:
    # Helper variables for SQL table creation
    TABLES = {
        "companies": """
                CREATE TABLE IF NOT EXISTS companies (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(10) UNIQUE NOT NULL
                );
            """,
        "prices": """
                CREATE TABLE IF NOT EXISTS prices (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    company_id INT,
                    date DATE NOT NULL,
                    field VARCHAR(50) NOT NULL,
                    value DOUBLE,
                    UNIQUE (company_id, date, field),
                    FOREIGN KEY (company_id) REFERENCES companies(id)
                );
            """,
        "balance_sheets": """
                CREATE TABLE IF NOT EXISTS balance_sheets (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    company_id INT,
                    period DATE NOT NULL,
                    field VARCHAR(50) NOT NULL,
                    value DOUBLE,
                    frequency ENUM('annual', 'quarterly') NOT NULL,
                    UNIQUE (company_id, period, field),
                    FOREIGN KEY (company_id) REFERENCES companies(id)
                );
            """
    }
    def _create_tables(self):
        """
        Creates necessary tables (companies, prices, balance_sheets) if they do not exist.
        """
        with self.connection.cursor() as cursor:
            for table_name, sql in self.TABLES.items():
                cursor.execute(sql)
                print(f"Table '{table_name}' checked/created successfully.")
            self.connection.commit()
    def __init__(self, host="localhost",port=3306, user="root", password="314159",database_name = "financial_database"):
        """
        Initializes the database connection.
        """
        self.connection = None
        if re.search('[A-Z]', database_name):
            raise ValueError("database_name must not contain any capital letter")

        try:
            # Connect to MySQL without specifying a database
            connection = pymysql.connect(
                host=host,
                user=user,
                password=password,
                port=port
            )
            self._db_create(connection,database_name)
            self._create_tables(connection)
            self.connection = connection
            self.engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database_name}")
        except pymysql.MySQLError as e:
            print(f"Error: {e}")
            raise

    def _db_create(self,connection,database_name):
        with connection.cursor() as cursor:
            # Check if the database exists
            cursor.execute(f"SHOW DATABASES LIKE '{database_name}';")
            result = cursor.fetchone()
            if not result:
                # Create the database if it doesn't exist
                cursor.execute(f"CREATE DATABASE {database_name};")
                print(f"Database '{database_name}' created successfully.")
            else:
                print(f"Database '{database_name}' already exists.")
            cursor.execute(f"USE {database_name};")
            cursor.execute("SELECT DATABASE();")
            print("Current Database:", cursor.fetchone()[0])

    def _table_exists(self, connection, table_name):
        """
        Checks if a table exists in the database.
        """
        with connection.cursor() as cursor:
            cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
            return cursor.fetchone() is not None

    def _create_tables(self,connection):
        """
        Creates necessary tables (companies, prices, balance_sheets) if they do not exist.
        """
        with connection.cursor() as cursor:
            for table_name, sql in self.TABLES.items():
                if not self._table_exists(connection, table_name):
                    cursor.execute(sql)
                    print(f"Table '{table_name}' created successfully.")
                else:
                    print(f"Table '{table_name}' already exists.")
            connection.commit()

    def get_company_id(self, ticker):
        """
        Fetches the company_id for the given ticker.
        """
        query = "SELECT id FROM companies WHERE ticker = %s;"
        with self.connection.cursor() as cursor:
            cursor.execute(query, (ticker,))
            result = cursor.fetchone()
        if result:
            return result[0]
        else:
            raise ValueError(f"Company with ticker '{ticker}' not found.")

    def upsert_data(self, ticker, table, data, is_price_data=False):
        """
        Upserts data into the specified table (e.g., prices, balance_sheets).

        Args:
            ticker (str): Company ticker symbol.
            table (str): Table name to update (e.g., 'prices', 'balance_sheets').
            data (list): List of dictionaries containing the data to upsert.
            is_price_data (bool): Set to True if the data is for price fields (e.g., OHLC).
        """
        company_id = self.get_company_id(ticker)

        # Determine fields and query
        if is_price_data:
            query = f"""
                INSERT INTO {table} (company_id, date, field, value)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    value = VALUES(value);
            """
            values = [
                (company_id, record['date'], record['field'], record['value'])
                for record in data
            ]
        else:
            query = f"""
                INSERT INTO {table} (company_id, period, field, value, frequency)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    value = VALUES(value),
                    frequency = VALUES(frequency);
            """
            values = [
                (company_id, record['period'], record['field'], record['value'], record['frequency'])
                for record in data
            ]

        with self.connection.cursor() as cursor:
            cursor.executemany(query, values)
            self.connection.commit()

    def get_data(self, ticker, table, field, start_date=None, end_date=None, count=None, frequency=None):
        """
        Retrieves data from the specified table (e.g., prices, balance_sheets).

        Args:
            ticker (str): Company ticker symbol.
            table (str): Table name (e.g., 'prices', 'balance_sheets').
            field (str): Field to retrieve (e.g., 'close', 'total_assets').
            start_date (str): Start date for filtering (optional).
            end_date (str): End date for filtering (optional).
            count (int): Number of recent records to retrieve (optional).
            frequency (str): Frequency filter for fundamental data (e.g., 'annual', 'quarterly').

        Returns:
            pd.DataFrame: Data retrieved as a Pandas DataFrame.
        """
        company_id = self.get_company_id(ticker)

        if count and end_date:
            # Retrieve the most recent `count` records up to `end_date`
            query = f"""
                WITH latest_periods AS (
                    SELECT period
                    FROM {table}
                    WHERE company_id = %s
                      AND field = %s
                      {"AND frequency = %s" if frequency else ""}
                      AND period <= %s
                    ORDER BY period DESC
                    LIMIT %s
                )
                SELECT t.period AS date, t.value
                FROM {table} t
                JOIN latest_periods lp ON t.period = lp.period
                WHERE t.company_id = %s AND t.field = %s
                      {"AND t.frequency = %s" if frequency else ""}
                ORDER BY t.period;
            """
            params = (company_id, field, frequency, end_date, count, company_id, field, frequency) if frequency else (
                company_id, field, end_date, count, company_id, field)
        else:
            # Retrieve data within a date range
            query = f"""
                SELECT t.{('date' if table == 'prices' else 'period')} AS date, t.value
                FROM {table} t
                WHERE t.company_id = %s
                  AND t.field = %s
                  AND t.{('date' if table == 'prices' else 'period')} BETWEEN %s AND %s
                  {"AND t.frequency = %s" if frequency else ""}
                ORDER BY t.{('date' if table == 'prices' else 'period')};
            """
            params = (company_id, field, start_date, end_date, frequency) if frequency else (
            company_id, field, start_date, end_date)

        data = pd.read_sql(query, self.engine, params=params)
        return data

    def adjust_for_weekends(self, date):
        """
        Adjusts the given date to the most recent working day if it falls on a weekend.
        """
        # If date is Saturday, move back to Friday
        if date.weekday() == 5:  # Saturday
            return date - timedelta(days=1)
        # If date is Sunday, move back to Friday
        elif date.weekday() == 6:  # Sunday
            return date - timedelta(days=2)
        # Otherwise, return the date unchanged
        return date

    def check_database_updates(self, tickers, start_date, end_date):
        """
        Checks if the database needs updates for the given tickers and date range,
        considering weekends for adjustments, and tracks which tables/frequencies need updates.

        Args:
            tickers (list[str]): List of tickers to check.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            dict[str, list[tuple]]: Dictionary where each key is a ticker, and each value
                                    is a list of tuples specifying (table, frequency) needing updates.
        """
        results = {ticker: [] for ticker in tickers}

        # Adjust end_date for weekends
        end_date = self.adjust_for_weekends(datetime.strptime(end_date, "%Y-%m-%d"))

        tables_with_frequencies = {
            "prices": ["daily"],
            "balance_sheets": ["quarterly", "annual"],
            "income_statements": ["quarterly", "annual"],
            "cash_flows": ["quarterly", "annual"]
        }
        for ticker in tickers:
            for table in ["prices", "balance_sheets"]:
                for frequency in tables_with_frequencies[table]:
                    # Define table and time adjustments
                    if frequency == "daily":
                        time_adjustment = timedelta(days=0)  # No adjustment for daily data
                    elif frequency == "quarterly":
                        time_adjustment = timedelta(days=90)
                    elif frequency == "annual":
                        time_adjustment = timedelta(days=365)
                    else:
                        raise ValueError("Invalid frequency specified.")

                    # Adjust the end date based on frequency
                    adjusted_end_date = end_date - time_adjustment

                    # Query the latest date for this ticker and frequency
                    query = f"""
                        SELECT MAX({"date" if table == "prices" else "period"}) AS latest_date
                        FROM {table} t
                        WHERE t.company_id = %s
                        {"AND t.frequency = %s" if frequency != "daily" else ""}
                    """
                    params = []
                    company_id = self.get_company_id(ticker)
                    params.append(company_id)
                    if frequency != "daily":
                        params.append(frequency)

                    # Execute query
                    with self.connection.cursor() as cursor:
                        cursor.execute(query, tuple(params))
                        d = cursor.fetchone()

                    latest_date = d[0] if d and d[0] else None

                    # Determine if the table/frequency needs an update
                    if not latest_date or latest_date < adjusted_end_date.date():
                        results[ticker].append((table, frequency))


        return results

    def close(self):
        """
        Closes the database connection.
        """
        self.connection.close()
