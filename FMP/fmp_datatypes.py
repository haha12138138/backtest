from enum import Enum

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
