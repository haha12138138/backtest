import datetime
import logging
import warnings
from enum import Enum
from typing import Optional, Union

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

from common_util import merge_data_by_date

# some basic global variables
API_KEY: str = "mZDI2HhRfp4QDHQ5KxGG3vvzwUpVUQRm"
BASE_URL_v3: str = "https://financialmodelingprep.com/api/v3/"
BASE_URL_v4: str = "https://financialmodelingprep.com/api/v4/"
INDUSTRY_VALUES: list = [
    "Entertainment",
    "Oil & Gas Midstream",
    "Semiconductors",
    "Specialty Industrial Machinery",
    "Banks Diversified",
    "Consumer Electronics",
    "Software Infrastructure",
    "Broadcasting",
    "Computer Hardware",
    "Building Materials",
    "Resorts & Casinos",
    "Auto Manufacturers",
    "Internet Content & Information",
    "Insurance Diversified",
    "Telecom Services",
    "Metals & Mining",
    "Capital Markets",
    "Steel",
    "Footwear & Accessories",
    "Household & Personal Products",
    "Other Industrial Metals & Mining",
    "Oil & Gas E&P",
    "Banks Regional",
    "Drug Manufacturers General",
    "Internet Retail",
    "Communication Equipment",
    "Semiconductor Equipment & Materials",
    "Oil & Gas Services",
    "Chemicals",
    "Electronic Gaming & Multimedia",
    "Oil & Gas Integrated",
    "Credit Services",
    "Online Media",
    "Business Services",
    "Biotechnology",
    "Grocery Stores",
    "Oil & Gas Equipment & Services",
    "REITs",
    "Copper",
    "Software Application",
    "Home Improvement Retail",
    "Pharmaceutical Retailers",
    "Communication Services",
    "Oil & Gas Drilling",
    "Electronic Components",
    "Packaged Foods",
    "Information Technology Services",
    "Leisure",
    "Specialty Retail",
    "Oil & Gas Refining & Marketing",
    "Tobacco",
    "Financial Data & Stock Exchanges",
    "Insurance Specialty",
    "Beverages Non-Alcoholic",
    "Asset Management",
    "REIT Diversified",
    "Residential Construction",
    "Travel & Leisure",
    "Gold",
    "Discount Stores",
    "Confectioners",
    "Medical Devices",
    "Banks",
    "Independent Oil & Gas",
    "Airlines",
    "Travel Services",
    "Aerospace & Defense",
    "Retail Apparel & Specialty",
    "Diagnostics & Research",
    "Trucking",
    "Insurance Property & Casualty",
    "Health Care Plans",
    "Consulting Services",
    "Aluminum",
    "Beverages Brewers",
    "REIT Residential",
    "Education & Training Services",
    "Apparel Retail",
    "Railroads",
    "Apparel Manufacturing",
    "Staffing & Employment Services",
    "Utilities Diversified",
    "Agricultural Inputs",
    "Restaurants",
    "Drug Manufacturers General Specialty & Generic",
    "Financial Conglomerates",
    "Personal Services",
    "Thermal Coal",
    "REIT Office",
    "Advertising Agencies",
    "Farm & Heavy Construction Machinery",
    "Consumer Packaged Goods",
    "Publishing",
    "Specialty Chemicals",
    "Engineering & Construction",
    "Utilities Independent Power Producers",
    "Utilities Regulated Electric",
    "Medical Instruments & Supplies",
    "Building Products & Equipment",
    "Packaging & Containers",
    "REIT Mortgage",
    "Department Stores",
    "Insurance Life",
    "Luxury Goods",
    "Auto Parts",
    "Autos",
    "REIT Specialty",
    "Integrated Freight & Logistics",
    "Security & Protection Services",
    "Utilities Regulated Gas",
    "Airports & Air Services",
    "Farm Products",
    "REIT Healthcare Facilities",
    "REIT Industrial",
    "Metal Fabrication",
    "Scientific & Technical Instruments",
    "Solar",
    "REIT Hotel & Motel",
    "Medical Distribution",
    "Medical Care Facilities",
    "Agriculture",
    "Food Distribution",
    "Health Information Services",
    "Industrial Products",
    "REIT Retail",
    "Conglomerates",
    "Health Care Providers",
    "Waste Management",
    "Beverages Wineries & Distilleries",
    "Marine Shipping",
    "Real Estate Services",
    "Tools & Accessories",
    "Auto & Truck Dealerships",
    "Industrial Distribution",
    "Uranium",
    "Lodging",
    "Electrical Equipment & Parts",
    "Gambling",
    "Specialty Business Services",
    "Recreational Vehicles",
    "Furnishings",
    "Fixtures & Appliances",
    "Forest Products",
    "Silver",
    "Business Equipment & Supplies",
    "Medical Instruments & Equipment",
    "Utilities Regulated",
    "Coking Coal",
    "Insurance Brokers",
    "Rental & Leasing Services",
    "Lumber & Wood Production",
    "Medical Diagnostics & Research",
    "Pollution & Treatment Controls",
    "Transportation & Logistics",
    "Other Precious Metals & Mining",
    "Brokers & Exchanges",
    "Beverages Alcoholic",
    "Mortgage Finance",
    "Utilities Regulated Water",
    "Manufacturing Apparel & Furniture",
    "Retail Defensive",
    "Real Estate Development",
    "Paper & Paper Products",
    "Insurance Reinsurance",
    "Homebuilding & Construction",
    "Coal",
    "Electronics & Computer Distribution",
    "Health Care Equipment & Services",
    "Education",
    "Employment Services",
    "Textile Manufacturing",
    "Real Estate Diversified",
    "Consulting & Outsourcing",
    "Utilities Renewable",
    "Tobacco Products",
    "Farm & Construction Machinery",
    "Shell Companies",
    "N/A",
    "Advertising & Marketing Services",
    "Capital Goods",
    "Insurance",
    "Industrial Electrical Equipment",
    "Utilities",
    "Pharmaceuticals",
    "Biotechnology & Life Sciences",
    "Infrastructure Operations",
    "Energy",
    "NULL",
    "Property Management",
    "Auto Dealerships",
    "Apparel Stores",
    "Mortgage Investment",
    "Software & Services",
    "Industrial Metals & Minerals",
    "Media & Entertainment",
    "Diversified Financials",
    "Consumer Services",
    "Commercial  & Professional Services",
    "Electronics Wholesale",
    "Retailing",
    "Automobiles & Components",
    "Materials",
    "Real Estate",
    "Food",
    "Beverage & Tobacco",
    "Closed-End Fund Debt",
    "Transportation",
    "Food & Staples Retailing",
    "Consumer Durables & Apparel",
    "Technology Hardware & Equipment",
    "Telecommunication Services",
    "Semiconductors & Semiconductor Equipment",
]
SECTOR_VALUES: list = [
    "Communication Services",
    "Energy",
    "Technology",
    "Industrials",
    "Financial Services",
    "Basic Materials",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Healthcare",
    "Real Estate",
    "Utilities",
    "Financial",
    "Building",
    "Industrial Goods",
    "Pharmaceuticals",
    "Services",
    "Conglomerates",
    "Media",
    "Banking",
    "Airlines",
    "Retail",
    "Metals & Mining",
    "Textiles",
    "Apparel & Luxury Goods",
    "Chemicals",
    "Biotechnology",
    "Electrical Equipment",
    "Aerospace & Defense",
    "Telecommunication",
    "Machinery",
    "Food Products",
    "Insurance",
    "Logistics & Transportation",
    "Health Care",
    "Beverages",
    "Consumer products",
    "Semiconductors",
    "Automobiles",
    "Trading Companies & Distributors",
    "Commercial Services & Supplies",
    "Construction",
    "Auto Components",
    "Hotels",
    "Restaurants & Leisure",
    "Life Sciences Tools & Services",
    "Communications",
    "Industrial Conglomerates",
    "Professional Services",
    "Road & Rail",
    "Tobacco",
    "Paper & Forest",
    "Packaging",
    "Leisure Products",
    "Transportation Infrastructure",
    "Distributors",
    "Marine",
    "Diversified Consumer Services",
]
PERIOD_VALUES: list = [
    "annual",
    "quarter",
]
TIME_DELTA_VALUES: list = [
    "1min",
    "5min",
    "15min",
    "30min",
    "1hour",
    "4hour",
]
TECHNICAL_INDICATORS_TIME_DELTA_VALUES: list = [
    "1min",
    "5min",
    "15min",
    "30min",
    "1hour",
    "4hour",
    "daily",
]
SERIES_TYPE_VALUES: list = [
    "line",
]
STATISTICS_TYPE_VALUES: list = [
    "sma",
    "ema",
    "wma",
    "dema",
    "tema",
    "williams",
    "rsa",
    "adx",
    "standardDeviation",
]


class Mkt_index(Enum):
    sp500 = "sp500_constituent"
    dowj = "dowjones_constituent"
    nasdaq = "nasdaq_constituent"

class Forcast_entry(Enum):
    date = "date"
    low_forecast_rev = "estimatedRevenueLow"
    high_forecast_rev = "estimatedRevenueHigh"
    mid_forecast_rev = "estimatedRevenueAvg"
    low_forecast_ebitda = "estimatedEbitdaLow"
    high_forecast_ebitda = "estimatedEbitdaHigh"
    mid_forecast_ebitda = "estimatedEbitdaAvg"
    low_forecast_ebit = "estimatedEbitLow"
    high_forecast_ebit = "estimatedEbitHigh"
    mid_forecast_ebit = "estimatedEbitAvg"
    low_forecast_net_income = "estimatedNetIncomeLow"
    high_forecast_net_income = "estimatedNetIncomeHigh"
    mid_forecast_net_income = "estimatedNetIncomeAvg"
    low_forecast_sga_expense = "estimatedSgaExpenseLow"
    high_forecast_sga_expense = "estimatedSgaExpenseHigh"
    mid_forecast_sga_expense = "estimatedSgaExpenseAvg"
    mid_forecast_eps = "estimatedEpsAvg"
    high_forecast_eps = "estimatedEpsHigh"
    low_forecast_eps = "estimatedEpsLow"
    num_analyst_estimated_revenue = "numberAnalystEstimatedRevenue"
    num_analyst_estimated_eps = "numberAnalystsEstimatedEps"

class Financial_ratio_entry(Enum):
    date = "date"
    current_ratio = "currentRatio"
    quick_ratio = "quickRatio"
    cash_ratio = "cashRatio"
    days_of_sales_outstanding = "daysOfSalesOutstanding"
    days_of_inventory_outstanding = "daysOfInventoryOutstanding"
    operating_cycle = "operatingCycle"
    days_of_payables_outstanding = "daysOfPayablesOutstanding"
    cash_conversion_cycle = "cashConversionCycle"
    gross_profit_margin = "grossProfitMargin"
    operating_profit_margin = "operatingProfitMargin"
    net_profit_margin = "netProfitMargin"
    tax_rate = "effectiveTaxRate"
    ROA = "returnOnAssets"
    ROE = "returnOnEquity"
    ROCE = "returnOnCapitalEmployed"
    debt_over_asset = "debtRatio"
    debt_over_equity = "debtEquityRatio"
    long_term_debt_to_cap = "longTermDebtToCapitalization"  # long-term debt by total available capital (long-term
    # debt, preferred stock, and common stock).
    debt_to_cap = "totalDebtToCapitalization"
    interest_coverage = "interestCoverage"
    cashflow_over_debt = "cashFlowToDebtRatio"
    equity_multiplier = "companyEquityMultiplier"
    receivables_turnover = "receivablesTurnover"
    payables_turnover = "payablesTurnover"
    inventory_turnover = "inventoryTurnover"
    fixed_asset_turnover = "fixedAssetTurnover"
    asset_turnover = "assetTurnover"
    operating_cashflow_per_share = "operatingCashFlowPerShare"
    FCF_per_share = "freeCashFlowPerShare"
    cash_per_share = "cashPerShare"
    payout_ratio = "payoutRatio"
    operating_cashflow_over_revenue = "operatingCashFlowSalesRatio"
    FCF_over_operating_cashflow = "freeCashFlowOperatingCashFlowRatio"
    cashflow_coverage = "cashFlowCoverageRatios"  # operating cash flow / total debt
    short_term_coverage = "shortTermCoverageRatios"
    capex_coverage = "capitalExpenditureCoverageRatio"  # operating cash flow / capex
    dividend_payout = "dividendPayoutRatio"  # dividend / net_income
    PB = "priceToBookRatio"
    PS = "priceToSalesRatio"
    PE = "priceEarningsRatio"
    PFCF = "priceToFreeCashFlowsRatio"
    POCF = "priceToOperatingCashFlowsRatio"
    PCF = "priceCashFlowRatio"
    PEG = "priceEarningsToGrowthRatio"
    dividend_yield = "dividendYield"
    EV_multiple = "enterpriseValueMultiple"
    price_fair_value = "priceFairValue"


class Income_statement_entry(Enum):
    date = "date"
    currency = "reportedCurrency"
    accept_date = "acceptedDate"
    revenue = "revenue"
    cost_of_revenue = "costOfRevenue"
    gross_profit = "grossProfit"
    gross_profit_ratio = "grossProfitRatio"
    R_and_D_expense = "researchAndDevelopmentExpenses"
    admin_expense = "generalAndAdministrativeExpenses"
    sales_expense = "sellingAndMarketingExpenses"
    sales_admin_expense = "sellingGeneralAndAdministrativeExpenses"
    other_expense = "otherExpenses"
    operating_expense = "operatingExpenses"
    cost_and_expense = "costAndExpenses"
    interest_income = "interestIncome"
    interest_expense = "interestExpense"
    depreciation_and_amortization = "depreciationAndAmortization"
    EBITDA = "ebitda"
    EBITDA_ratio = "ebitdaratio"
    operating_income = "operatingIncome"
    operating_income_ratio = "operatingIncomeRatio"
    income_before_tax = "incomeBeforeTax"
    income_tax_expense = "incomeTaxExpense"
    net_profit = "netIncome"
    net_profit_margin = "netIncomeRatio"
    EPS = "eps"
    EPS_diluted = "epsdiluted"
    outstanding_shares_weighted_avg = "weightedAverageShsOut"
    outstanding_diluted_shares_weighted_avg = "weightedAverageShsOutDil"


class Balance_statement_entry(Enum):
    date = "date"
    currency = "reportedCurrency"
    accept_date = "acceptedDate"
    cash = "cashAndCashEquivalents"
    short_term_investment = "shortTermInvestments"
    cash_and_short_investment = "cashAndShortTermInvestments"
    net_receivables = "netReceivables"
    inventory = "inventory"
    other_current_assets = "otherCurrentAssets"
    current_assets = "totalCurrentAssets"
    PPE = "propertyPlantEquipmentNet"
    good_will = "goodwill"
    intangible_assets = "intangibleAssets"
    long_term_investment = "longTermInvestments"
    tax_assets = "taxAssets"
    other_non_current_assets = "otherNonCurrentAssets"
    non_current_assets = "totalNonCurrentAssets"
    other_assets = "otherAssets"
    total_assets = "totalAssets"
    account_payables = "accountPayables"
    short_term_debt = "shortTermDebt"
    tax_payables = "taxPayables"
    defferred_revenue = "deferredRevenue"
    other_current_liabilities = "otherCurrentLiabilities"
    current_liabilities = "totalCurrentLiabilities"
    long_term_debt = "longTermDebt"
    deferred_non_current_revenue = "deferredRevenueNonCurrent"
    deferred_non_current_tax = "deferredTaxLiabilitiesNonCurrent"
    other_non_current_liabilities = "otherNonCurrentLiabilities"
    total_non_current_liabilities = "totalNonCurrentLiabilities"
    other_liabilities = "otherLiabilities"
    capital_lease_oblg = "capitalLeaseObligations"
    total_liabilities = "totalLiabilities"
    preferred_stock = "preferredStock"
    common_stock = "commonStock"
    retained_earning = "retainedEarnings"
    other_comprehensive_income_loss = "accumulatedOtherComprehensiveIncomeLoss"
    other_total_stock = "othertotalStockholdersEquity"
    stockholder_equity = "totalStockholdersEquity"
    total_equity = "totalEquity"
    total_liabilities_and_stockholder_equity = "totalLiabilitiesAndStockholdersEquity"
    minority_interest = "minorityInterest"
    total_liabilities_and_equity = "totalLiabilitiesAndTotalEquity"
    total_investment = "totalInvestments"
    total_debt = "totalDebt"
    net_debt = "netDebt"


class Cashflow_statement_entry(Enum):
    date = "date"
    currency = "reportedCurrency"
    accept_date = "acceptedDate"
    net_profit = "netIncome"
    depreciation_and_amortization = "depreciationAndAmortization"
    deferred_income_tax = "deferredIncomeTax"
    stock_compensation = "stockBasedCompensation"
    working_capital_chg = "changeInWorkingCapital"
    account_receivables = "accountsReceivables"
    inventory = "inventory"
    account_payables = "accountsPayables"
    other_working_capitals = "otherWorkingCapital"
    other_non_cash_items = "otherNonCashItems"
    net_cash_from_operating_activities = "netCashProvidedByOperatingActivities"
    investment_in_PPE = "investmentsInPropertyPlantAndEquipment"
    net_aquisition = "acquisitionsNet"
    purchase_of_investments = "purchasesOfInvestments"
    sales_maturities_of_investments = "salesMaturitiesOfInvestments"
    other_investing_activities = "otherInvestingActivites"
    net_cash_from_investing_activities = "netCashUsedForInvestingActivites"
    debt_repayment = "debtRepayment"
    issued_common_stock = "commonStockIssued"
    purchased_common_stock = "commonStockRepurchased"
    dividends = "dividendsPaid"
    other_financing_activities = "otherFinancingActivites"
    net_cash_from_financing_activities = "netCashUsedProvidedByFinancingActivities"
    forex_chg_on_cash = "effectOfForexChangesOnCash"
    net_cash_chg = "netChangeInCash"
    cash_at_end_of_period = "cashAtEndOfPeriod"
    cash_at_start_of_period = "cashAtBeginningOfPeriod"
    operating_cashflow = "operatingCashFlow"
    capex = "capitalExpenditure"
    FCF = "freeCashFlow"


class Price_entry(Enum):
    date = "date"
    open = "open"
    high = "high"
    low = "low"
    close = "close"
    adjClose = "adjClose"
    volume = "volume"
    change = "change"
    change_percent = "changePercent"
    vwap = "vwap"


class Datapoint_period(Enum):
    day = 0
    quarter = 1
    annual = 2


class Financial_statement_period(Enum):
    annual = "annual"
    quarter = "quarter"


# some basic url functions
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 30
import time


def rate_limits(limit_per_min):
    def decorator(func):
        last_call = 0

        def wrapper(*args, **kwargs):
            nonlocal last_call
            # Calculate time elapsed since last reset
            now = time.time()
            elapsed = now - last_call
            if elapsed < 60 / limit_per_min:
                time.sleep(60 / limit_per_min - elapsed)
            last_call = now
            # Call the original function
            return func(*args, **kwargs)

        return wrapper

    return decorator


@rate_limits(300)
def __api_access(url):
    not_finish = True
    return_var = []
    retry = 0
    while not_finish:
        try:
            response = requests.get(
                url=url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            if len(response.content) > 0:
                return_var = response.json()

            if len(response.content) == 0 or (
                    isinstance(return_var, dict) and len(return_var.keys()) == 0
            ):
                logging.warning("Response appears to have no data.  Returning empty List.")
            not_finish = False
        except requests.Timeout:
            time.sleep(0.01)
            retry += 1
            if retry > 5:
                not_finish = False
                logging.error(f"Connection to {url} timed out.")
            else:
                not_finish = True
        except requests.ConnectionError:
            time.sleep(0.01)
            retry += 1
            if retry > 5:
                not_finish = False
                logging.error(
                    f"Connection to {url} failed:  DNS failure, refused connection or some other connection related "
                    f"issue."
                )
            else:
                not_finish = True
        except requests.TooManyRedirects:
            logging.error(
                f"Request to {url} exceeds the maximum number of predefined redirections."
            )
            not_finish = False
        except Exception as e:
            logging.error(
                f"A requests exception has occurred that we have not yet detailed an 'except' clause for.  "
                f"Error: {e}"
            )
            not_finish = False

    return return_var


# def fmp_check_ticker_exists(ticker: str, apikey:str):
#     url = f"{BASE_URL_v3}/search-ticker?query={ticker}&limit=1&apikey={apikey}"
#     res = __api_access(url)
#     return len(res) != 0

def fmp_check_ticker_exists(ticker: str, apikey: str, only_active_trading=False):
    url = f"{BASE_URL_v3}/profile/{ticker}?apikey={apikey}"
    res = __api_access(url)
    if len(res) == 0:
        return False
    elif (res[0]["isActivelyTrading"] == False) and (only_active_trading == True):
        warnings.warn(f"This ticker: {ticker} is not actively traded. only_active_trading flag is set to True")
        return False
    else:
        return True


def fmp_get_industry(apikey: str):
    url = f"{BASE_URL_v4}/standard_industrial_classification/all?apikey={apikey}"
    res = __api_access(url)
    df = pd.DataFrame(res)
    df = df.set_index("symbol")
    return df


def get_mkt_constituents(index: Mkt_index, apikey) -> tuple[str, list]:
    url = f"{BASE_URL_v3}/{index.value}?apikey={apikey}"
    res = __api_access(url)
    df = pd.DataFrame(res)
    # we only need, symbol, sector and subsector
    tickers = df["symbol"].values.tolist()
    return str(datetime.date.today()), tickers


def construct_mkt_historical_constituents(index: Mkt_index, constituents, apikey):
    url_hist = f"{BASE_URL_v3}/historical/{index.value}?apikey={apikey}"
    res = __api_access(url_hist)
    last_change_date = constituents[0]
    all_historical_constituents = []
    next_constituent_values = dict()
    for change in res:
        change_date = change["date"]
        change_symbol = change["symbol"]
        if change_date != last_change_date:
            if len(all_historical_constituents) == 0:
                all_historical_constituents = [constituents]
                next_constituent_values = {i: None for i in constituents[1]}
                last_change_date = change_date
            else:
                all_historical_constituents.append(((datetime.datetime.strptime(last_change_date,
                                                                                "%Y-%m-%d") - datetime.timedelta(
                    days=1)).strftime("%Y-%m-%d")
                                                    , list(next_constituent_values.keys())))
                last_change_date = change_date
        if change["removedTicker"] == "":
            # added into the index on this date. we need to remove it from the constituents
            del next_constituent_values[change_symbol]
        else:
            next_constituent_values[change_symbol] = None
    return all_historical_constituents


def fmp_get_etf_holdings(ticker: str, apikey: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                         highest_N=10):
    url = f"{BASE_URL_v4}etf-holdings/portfolio-date?symbol={ticker}&apikey={apikey}"
    res = __api_access(url)
    list_of_dates = [pd.to_datetime(i["date"], format="%Y-%m-%d") for i in res]
    start_date = pd.to_datetime(start_date, format="%Y-%m-%d")
    end_date = pd.to_datetime(end_date, format="%Y-%m-%d")
    list_of_date_in_date_range = list(filter(lambda x: (x >= start_date) and (x <= end_date), list_of_dates))
    df_holding = pd.DataFrame(columns=["date", "holdings"])
    verified_list = []
    failed_list =[]
    if list_of_date_in_date_range != []:
        for date in list_of_date_in_date_range:
            url = f"{BASE_URL_v4}etf-holdings?date={date.strftime('%Y-%m-%d')}&symbol={ticker}&apikey={apikey}"
            res = __api_access(url)
            df_res = pd.DataFrame(res)[["symbol", "pctVal"]].sort_values(by="pctVal", ascending=False).iloc[0:highest_N]
            holding_list = df_res["symbol"].values.tolist()
            holding_list = list(filter(lambda x: x is not None, holding_list))
            verified_set = set(verified_list)
            failed_set = set(failed_list)
            holding_list_temp = []

            for holding in holding_list:
                if holding in verified_set:
                    holding_list_temp.append(holding)
                elif holding not in failed_set:
                    if fmp_check_ticker_exists(holding, apikey):
                        verified_set.add(holding)
                        holding_list_temp.append(holding)
                    else:
                        failed_set.add(holding)

            # Update the lists with the sets for future iterations
            verified_list = list(verified_set)
            failed_list = list(failed_set)


            df_holding.loc[len(df_holding)] = {"date": date, "holdings": holding_list_temp}
    if start_date < min(list_of_date_in_date_range):
        df_oldest_holdings = df_holding.loc[df_holding["date"] == min(list_of_date_in_date_range)]
        oldest_holdings = df_oldest_holdings["holdings"].tolist()[0]
        holding_list_temp = []
        for holding in oldest_holdings:
            if fmp_check_ticker_exists(holding, apikey) == True:
                holding_list_temp.append(holding)

        df_holding.loc[len(df_holding)] = {"date": start_date, "holdings": holding_list_temp}
    return df_holding.set_index("date").sort_index(ascending=True)


def fmp_get_mktcap_of_a_stock(ticker: str
                              , apikey: str
                              , start: Optional[str] = None
                              , end: Optional[str] = None):
    """
    This is a function that gets daily mktcap of a stock.
    mktcap is calculated by multiplying the current share price by the number of outstanding shares.
    :param ticker:
    :param apikey:
    :param start: data start_date date. (if both start_date and end_date are None, fetch the latest mktcap data)
    :param end: data end_date date. (if both start_date and end_date are None, fetch the latest mktcap data)
    :return: dataframe that use ticker as column name and date as index. date is formatted as "YYYY-mm-dd". date is sorted
    in ascending order (oldest date first)
    """
    if (start is None) and (end is None):
        # get only the latest data.
        url = f"{BASE_URL_v3}/market-capitalization/{ticker}?apikey={apikey}"
        res = __api_access(url)
        df = pd.DataFrame(pd.DataFrame(res).set_index("date")["marketCap"]).rename({"marketCap": ticker},
                                                                                   axis="columns")
        return df
    elif (start is not None) and (end is not None):
        df_list = []
        url_base = f"{BASE_URL_v3}/historical-market-capitalization/{ticker}?"
        start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
        temp_start_date = end_date - relativedelta(years=5)
        while temp_start_date > start_date:
            temp_start = temp_start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")
            url = f"{url_base}from={temp_start}&to={end}&apikey={apikey}"
            res = __api_access(url)
            if len(res) != 0:
                df_list.append(pd.DataFrame(res).set_index("date")["marketCap"])
            end_date = temp_start_date - relativedelta(days=1)
            temp_start_date = end_date - relativedelta(years=5)
        # temp_start_date <= data_start_date
        end = end_date.strftime("%Y-%m-%d")
        url = f"{url_base}from={start}&to={end}&apikey={apikey}"
        res = __api_access(url)
        if len(res) != 0:
            df_list.append(pd.DataFrame(res).set_index("date")["marketCap"])

        if len(df_list) == 0:
            df_temp = pd.DataFrame(columns=[ticker])
            df_temp.index.name = "date"
            df_list.append(df_temp)
        return pd.DataFrame(pd.concat(df_list)).sort_index(ascending=True).rename({"marketCap": ticker}, axis="columns")


def fmp_get_price_of_a_stock(ticker: str
                             , fields: list[Price_entry]
                             , apikey: str
                             , start: Optional[str] = None
                             , end: Optional[str] = None
                             ) -> pd.DataFrame:
    """
    :param fields:
    :param ticker: a string that contains the ticker of a stock
    :param start: start_date date of the data (if that day is holiday or weekend, it will be the first day after)
    :param end: end_date date of the data (if that day is holiday or weekend, it will be the first day before)
    :return: a multi-index dataframe.
            the 1st level col is ticker
            , the 2nd level contains fields selected in arg field (eg: open/close/high/low/volumn)
    """
    url = f"{BASE_URL_v3}/historical-price-full/{ticker}?apikey={apikey}"
    if start is not None:
        url += f"&from={start}"
    if end is not None:
        url += f"&to={end}"
    res = __api_access(url)
    field_names = list(map(lambda x: x.name, fields))
    if len(res) != 0:
        df = pd.DataFrame(res["historical"])
        df['date'] = pd.to_datetime(df['date'])
        df.rename(columns={x.value: x.name for x in fields}, inplace=True)

        df = df[field_names + ['date']].set_index('date')
        df.columns = pd.MultiIndex.from_product([[res['symbol']], df.columns])
    else:
        df = pd.DataFrame(data=None, columns=field_names)
        df.index.name = 'date'
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    return df


def __fmp_get_statement(ticker: str
                        , type: str
                        , fields: Union[
            list[Income_statement_entry], list[Balance_statement_entry], list[Cashflow_statement_entry], list[
                Financial_ratio_entry]]
                        , apikey: str
                        , start: Optional[str] = None
                        , end: Optional[str] = None
                        , count: Optional[int] = None
                        , period: Financial_statement_period = Financial_statement_period.annual
                        ) -> pd.DataFrame:
    """
    This is a helper function of fmp_get_fundamentals_of_a_stock. It can only get 1 type of statement.
    """
    print(fields)
    # check the type of field.
    if type == "I":
        url = f"{BASE_URL_v3}/income-statement/{ticker}?period={period.value}&apikey={apikey}"
    elif type == "B":
        url = f"{BASE_URL_v3}/balance-sheet-statement/{ticker}?period={period.value}&apikey={apikey}"
    elif type == "C":
        url = f"{BASE_URL_v3}/cash-flow-statement/{ticker}?period={period.value}&apikey={apikey}"
    elif type == "R":
        url = f"{BASE_URL_v3}/ratios/{ticker}?period={period.value}&apikey={apikey}"
    else:
        raise ValueError(
            "field can only be chosen among income_statement_entry balance_statement_entry cashflow_statement_entry")
    field_name = list(map(lambda x: x.value, fields))
    res = __api_access(url)
    if len(res) != 0:
        df = pd.DataFrame(res)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.sort_index(ascending=False, inplace=True)
        if start is None:
            df = df[df.index <= end][:count]
        elif end is None:
            df = df[df.index >= start][-(count):]
        else:
            df = df[(df.index >= start) & (df.index <= end)]
        # df = get_filtered_date_data(df
        #                             , data_start_date=start_date, data_end_date=end_date, count=count
        #                             , freq="3M" if period == Financial_statement_period.quarter else "12M")
        # fetch necessary fields
        df = df[field_name]
        # convert the name to enum name
        df.rename(columns={x.value: x.name for x in fields}, inplace=True)
        df.columns = pd.MultiIndex.from_product([[res[0]['symbol']], df.columns])
    else:
        df = pd.DataFrame(data=None, columns=field_name)
        df.index.name = 'date'
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    return df


def fmp_get_fundamentals_of_a_stock(ticker: str
                                    , fields: list[
            Union[Income_statement_entry, Balance_statement_entry, Cashflow_statement_entry, Financial_ratio_entry]]
                                    , apikey: str
                                    , start: Optional[str] = None
                                    , end: Optional[str] = None
                                    , count: Optional[int] = None
                                    , period: Financial_statement_period = Financial_statement_period.annual
                                    ) -> pd.DataFrame:
    """
    This function is to retrieve financial information. this includes, balance sheet, cashflow sheet, income sheet
    along with other useful financial ratios.

    :param ticker: stock name list
    :param fields: a list of financial information
    :param apikey:
    :param start: start_date date
    :param end: end_date date
    :param count: the length of data measured by "period" arg. So it is possible that 2 results are given even though count == 1
    :param period: whether you are retrieve quarterly result or annual result
    :return: a multi-level index dataframe.
             index is the date (timestamp, datetime ...)
             1st level of column is ticker, 2nd level constains the name of financial information you searched for

    """
    # group field by their type
    field_type: dict = {"I": [], "B": [], "C": [], "R": []}
    for field in fields:
        if field.name in Income_statement_entry.__members__.keys():
            field_type['I'].append(field)
        elif field.name in Balance_statement_entry.__members__.keys():
            field_type['B'].append(field)
        elif field.name in Cashflow_statement_entry.__members__.keys():
            field_type['C'].append(field)
        elif field.name in Financial_ratio_entry.__members__.keys():
            field_type['R'].append(field)
        else:
            raise ValueError("unrecognized field type: " + str(type(field)))
    dfs = []
    for k, v in field_type.items():
        if v == []:
            continue
        dfs.append(__fmp_get_statement(ticker, type=k, fields=v, apikey=apikey
                                       , start=start, end=end, count=count
                                       , period=period))
    return merge_data_by_date(dfs)

def fmp_get_forecast(ticker: str
                     , fields: list[Forcast_entry]
                     , apikey:str
                     , start: Optional[str] = None
                     , count: Optional[int] = None
                     , period: Financial_statement_period = Financial_statement_period.annual) -> pd.DataFrame:
    url = f"{BASE_URL_v3}/analyst-estimates/{ticker}?period={period.value}&apikey={apikey}"
    field_name = list(map(lambda x: x.value, fields))
    res = __api_access(url)
    if len(res) != 0:
        df = pd.DataFrame(res)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.sort_index(ascending=False, inplace=True)
        df = df[df.index >= start][-(count):]
        df = df[field_name]
        df.rename(columns={x.value: x.name for x in fields}, inplace=True)
        df.columns = pd.MultiIndex.from_product([[res[0]['symbol']], df.columns])
    else:
        df = pd.DataFrame(data=None, columns=field_name)
        df.index.name = 'date'
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    return df
