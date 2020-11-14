import numpy as np
import pandas as pd
# from datetime import datetime
import seaborn as sns

sns.set_style('white')
from dateutil.relativedelta import relativedelta
import make_farsi_text as farsi
from sqlalchemy import Column, DateTime, String, Float, Integer, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine, MetaData, Table
import scipy.optimize as sco

"""
Required functions for portfolio optimization:

    portfolio_annualised_performance: measures portfolio performance and returns 'return' and 'std' 

    random_portfolios: producing 'num_portfolios' random portfolios and returns 'weights' and 'results' ('results' 
    includes portfolio_std_dev, portfolio_return and (portfolio_return - risk_free_rate) / portfolio_std_dev)

    display_simulated_ef_with_random: displays simulated results (only for test and not used in final processing)

    neg_sharpe_ratio: handles negative sharpe_ratio

    max_sharpe_ratio: minimizing using 'sco.minimize' for maximum of sharpe_ratio

    portfolio_volatility: calculates volatility of portfolios by calling 'portfolio_annualised_performance' function

    min_variance: minimizing using 'sco.minimize' for minimize of variance

    efficient_return: returns 'efficient_return' as result using 'sco.minimize'

    efficient_frontier: some calculations for efficient frontier (not used in final processing) 

    display_calculated_ef_with_random: displays calculated results (is used in final processing) 

"""

""" Essential functions """

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, tickers_num):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(tickers_num)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, tickers_num)
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=table.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=table.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets * [1. / num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets * [1. / num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    return result


def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets * [1. / num_assets, ], args=args, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, _ = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, tickers_num)

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index=table.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x, index=table.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)

    return results, max_sharpe_allocation, min_vol_allocation, sdp, rp, sdp_min, rp_min


"""
Monte_Carlo (MC) simulations
"""

def MC(data):
    num_simulations = 1000
    predicted_days = 252

    price = data[-252:]
    returns = price.pct_change().dropna()  # dropna is not necessary because of method='ffill'

    last_price = price[-1]

    simulation_df = pd.DataFrame()

    for x in range(num_simulations):
        count = 0
        daily_vol = returns.std()
        price_series = []
        price = last_price * (1 + np.random.normal(0, daily_vol))
        price_series.append(price)
        for y in range(predicted_days):
            if count == 251:
                break
            price = price_series[count] * (1 + np.random.normal(0, daily_vol))
            price_series.append(price)
            count += 1
        simulation_df[x] = price_series

    ind1 = np.arange(0, 21)
    ser1 = simulation_df.iloc[ind1, :]
    ind2 = np.arange(41, 252, 21)
    ser2 = simulation_df.iloc[ind2, :]
    ser = pd.concat([ser1, ser2])

    return ser

"""
Defining farsi fonts to call namads and also use in plots (namad:= name of ticker)
"""

font_title = {'family': 'B Mitra',
              'color': 'black',
              'weight': 'bold',
              'size': 14}
font_labels = {'family': 'B Nazanin',
               'color': 'black',
               'weight': 'normal',
               'size': 14}

""" read sectors' names from database and save them and their IDs in 'groups' and 'GroupsID' """
cnx = mysql.connector.connect(user='root', password='',
                              host='',
                              database='database_name')

cur = cnx.cursor()
sql = "SELECT * FROM namad_sector"
cur.execute(sql)
groups = cur.fetchall()
groups_data = pd.DataFrame(groups)
groups = pd.DataFrame(groups_data.iloc[:, 3])  # read sector names
GroupsID = pd.DataFrame(groups_data.iloc[:, 0])

WholeTickers = pd.DataFrame()  # define a dataframe to hold all namads (tickers)
CompletedTable = pd.DataFrame()  # define a dataframe to hold close price of all tickers
WholeID = pd.DataFrame()  # define a dataframe to hold all IDs for tickers
cnx.close()

""" 
---> Main Loop including all processes for extracting features, portfolio optimization (using all tickers and 
also tickers every single sector)

---> this loop iterates for all groups

"""

for group in range(len(groups)):

    cnx = mysql.connector.connect(user='root', password='',
                                  host='',
                                  database='database_name')
    cur = cnx.cursor()

    sql = "SELECT * FROM namad n join namad_sector nc on n.sector_id = nc.id where nc.name = " + "\'" + \
          groups.iloc[group, 0] + "\'"
    cur.execute(sql)
    data = cur.fetchall()  # Load tickers for group and save them in tickers dataframe
    data = pd.DataFrame(data)
    tickers = pd.DataFrame(data.iloc[:, 9:11].values)
    if len(tickers) == 0:
        continue

    MarketValue = pd.DataFrame(data.iloc[:, 3])  # Load MarketValue for tickers
    sectorTotalValue = np.sum(MarketValue)  # Sum of MarketValue

    # if SectorTotalValue = 0, MarketValue will be substituted by transactions' volume in that sector, this
    # dose not affect on results
    if SectorTotalValue.values == 0:
        MarketValue = pd.DataFrame(data.iloc[:, 7])
        SectorTotalValue = np.sum(MarketValue)

    WeightsIndex = []  # required weights for calculating sector index (group index)
    for i in range(len(MarketValue)):
        WeightsIndex.append(MarketValue.iloc[i, 0] / SectorTotalValue.values)

    WeightsIndex = pd.DataFrame(WeightsIndex)

    """
    Load history of data for all tickers in sector, pre-process them and convert to a applicable format
    """
    namad_len = np.zeros(shape=(len(tickers), 2))
    for namad in range(len(tickers)):
        sql = "SELECT n.namad, n.name, nh.date, nh.close, nh.volume FROM namad_history nh join namad n on nh.namad_id=n.id where n.namad = " + "\'" + \
              tickers.iloc[namad, 0] + "\'" + " ORDER BY date ASC"
        cur.execute(sql)
        crude_data = cur.fetchall()
        if namad == 0:
            data = pd.DataFrame(crude_data)
            str_data = data.iloc[:, 0:2]
            num_data = data.drop(data.columns[0:2], 1)
            namad_len[namad, 0] = 0
            namad_len[namad, 1] = num_data.shape[0] - 1
        else:
            df = pd.DataFrame(crude_data)
            str_df = df.iloc[:, 0:2]
            num_df = df.drop(df.columns[0:2], 1)
            str_data = pd.concat([str_data, str_df], ignore_index=True)
            num_data = pd.concat([num_data, num_df], ignore_index=True)
            namad_len[namad, 0] = namad_len[namad - 1, 1] + 1
            namad_len[namad, 1] = namad_len[namad, 0] + num_df.shape[0] - 1

    data = pd.concat([str_data, num_data], axis=1)
    data.columns = ['Namad', 'Name', 'Date', 'Close', 'Volume']

    d = data['Date'].values
    dd = np.unique(d)
    Date_sorted = np.sort(dd)

    CloseTick = pd.DataFrame(index=range(len(Date_sorted)), columns=range(len(tickers)))
    VolumeTick = pd.DataFrame(index=range(len(Date_sorted)), columns=range(len(tickers)))
    for t in range(len(tickers)):
        DateTicker = data[data['Namad'] == tickers.iloc[t, 0]]['Date'].values
        for i in range(len(Date_sorted)):
            k = namad_len[t][0]
            for j in range(len(DateTicker)):
                if Date_sorted[i] == DateTicker[j]:
                    CloseTick[t][i] = data['Close'][k]
                    VolumeTick[t][i] = data['Volume'][k]
                k = k + 1

    CloseTick.fillna(method='ffill', inplace=True)
    index = []
    for i in range(CloseTick.shape[0]):
        index.append(np.sum(CloseTick.iloc[i, :] * WeightsIndex.iloc[:, 0]))

    index = pd.DataFrame(index)

    nan_index = []
    for i in range(len(tickers)):
        if CloseTick.iloc[:, i].isnull().sum() > 0.9 * CloseTick.shape[0]:
            nan_index.append(i)
    CloseTick = CloseTick.drop(CloseTick.columns[nan_index], 1)
    VolumeTick = VolumeTick.drop(VolumeTick.columns[nan_index], 1)
    tickers = tickers.drop(tickers.index[[nan_index]])
    tickers = tickers.reset_index(drop=True)

    if len(tickers) == 0:
        continue

    WholeTickers = pd.concat([WholeTickers, tickers], ignore_index=True)
    ID = pd.DataFrame()
    for namad in range(len(tickers)):
        sql_id = "SELECT n.id as nid from namad n join namad_history nh on nh.namad_id=n.id where n.namad = " + "\'" + \
                 tickers.iloc[namad, 0] + "\'" + "limit 1"
        cur.execute(sql_id)
        CrudeID = cur.fetchall()
        CrudeID = pd.DataFrame(CrudeID)
        ID = pd.concat([ID, CrudeID], axis=1)

    WholeID = pd.concat([WholeID, ID], axis=1)

    CloseTick = pd.concat([CloseTick, index], axis=1)

    TickersDict = {}
    for ind in range(len(tickers) + 1):
        if ind < len(tickers):
            TickersDict[ind] = tickers.iloc[ind, 0]
        else:
            TickersDict[ind] = 'sector index'

    CloseTick.columns = TickersDict.values()

    VolumeTick.columns = tickers.iloc[:, 0]

    d_sorted = pd.DataFrame(Date_sorted)
    d_sorted.columns = ['Date']
    close = pd.concat([d_sorted, CloseTick], axis=1)
    volume = pd.concat([d_sorted, VolumeTick], axis=1)
    close.set_index(['Date'], inplace=True)
    close.index = pd.to_datetime(close.index)
    volume.set_index(['Date'], inplace=True)
    volume.index = pd.to_datetime(volume.index)

    close = close.astype(float)
    volume = volume.astype(float)

    namad_close = close.iloc[:, 0:len(tickers)]  # make a copy of namad's close prices
    # calculate correlation matrix
    corr = namad_close.corr()

    SimpleReturn = namad_close.copy()  # Make a copy of close df

    # First not-nan valid values in close's columns   # this should be modified after applying "method = ffill"
    FirstValidIndices = np.zeros(shape=(1, len(tickers)))
    # Last date available in data set
    LastDate = close.index[-1]
    # Duration years available in each columns in data set
    DurationYears = np.zeros(shape=(1, len(tickers)))

    # Calculating return over all-days and Duration years available in each columns (at the same time)
    for ind in range(len(tickers)):
        FirstValidIndices[0, ind] = close.loc[close.iloc[:, ind].FirstValidIndex()][ind]
        SimpleReturn.iloc[:, ind] = ((SimpleReturn.iloc[:, ind] - FirstValidIndices[0, ind]) /
                                     FirstValidIndices[0, ind]) * 100
        DurationYears[0, ind] = relativedelta(LastDate, close.iloc[:, ind].FirstValidIndex()).years

    # calculate annualized return
    AnnualReturn = np.zeros(shape=(len(tickers), 1))
    for t in range(len(tickers)):
        if DurationYears[0, t] > 0:
            AnnualReturn[t, 0] = ((SimpleReturn.iloc[:, t][-1:] / 100 + 1) ** (
                    1 / DurationYears[0, t]) - 1) * 100

    AnnualReturn = pd.DataFrame(AnnualReturn)
    AnnualReturn.columns = [farsi.make_farsi_text('annual return')]

    # daily returns of these stocks
    CloseCopy = namad_close.copy()
    DailyReturn = CloseCopy.pct_change().dropna()

    DailyReturn_mean = DailyReturn.mean()
    DailyReturn_std = DailyReturn.std()

    # calculate sharpe ratio
    AnnualRF = 0.02  # annual rf := 0.02 (annual risk-free rate of return is chosen randomly here, the exact value
    # could be determined for every market)
    DailyRFReturn = (1 + AnnualRF) ** (1 / 252) - 1
    DailyReturn_adj = DailyReturn - DailyRFReturn
    sharpe = np.zeros(shape=(len(tickers), 1))
    for t in range(len(tickers)):
        if DailyReturn_adj.iloc[:, t].std() == 0:
            sharpe[t, 0] = DailyReturn_adj.iloc[:, t].mean() / 0.01
        else:
            sharpe[t, 0] = DailyReturn_adj.iloc[:, t].mean() / DailyReturn_adj.iloc[:, t].std()

    sharpe = pd.DataFrame(sharpe)
    sharpe.columns = ['نسبت شارپ']      # Sharp ratio

    # calculating beta
    ref = close.iloc[:, -1].pct_change().dropna().tail(len(DailyReturn))  # index's daily return
    beta = np.zeros(shape=(len(tickers), 1))
    for t in range(len(tickers)):
        beta[t, 0] = np.cov(DailyReturn.iloc[:, t], ref)[0][1] / np.var(ref)

    beta = pd.DataFrame(beta)
    beta.columns = ['بتا']      # Beta

    # calculating alpha
    FirstValidIndexIndex = close.loc[close.iloc[:, -1].FirstValidIndex()][-1]
    SimpleReturnIndex = ((close.iloc[:, -1] - FirstValidIndexIndex) / FirstValidIndexIndex) * 100
    DurationYearsIndex = relativedelta(LastDate, close.iloc[:, -1].FirstValidIndex()).years
    AnnualReturn_index = ((SimpleReturnIndex[-1:] / 100 + 1) ** (1 / DurationYearsIndex) - 1) * 100

    if np.any(np.isinf(AnnualReturnIndex) == 1):
        continue

    alpha = np.zeros(shape=(len(tickers), 1))
    for k, v in enumerate(beta.values):
        alpha[k, 0] = AnnualReturn.iloc[k, 0] / 100 - AnnualRF - v * (AnnualReturnIndex / 100 - AnnualRF)

    #############################################
    """
    Feature's ranking 
    """
    RankedClose = close

    namad_close = RankedClose.iloc[:, :-1]  # make a copy of namad's close prices

    SimpleReturn = namad_close.copy()  # Make a copy of close df

    # First not-nan valid values in close's columns
    FirstValidIndices = np.zeros(shape=(1, len(namad_close.columns)))
    # Last date available in data set
    LastDate = RankedClose.index[-1]
    # Duration years available in each columns in data set
    DurationYears = np.zeros(shape=(1, len(namad_close.columns)))
    TotalReturn = np.zeros(shape=(len(namad_close.columns), 1))

    # Calculating return over all-days and Duration years available in each columns (at the same time)
    # and total return for each namad
    for ind in range(len(namad_close.columns)):
        FirstValidIndices[0, ind] = namad_close.loc[namad_close.iloc[:, ind].FirstValidIndex()][ind]
        SimpleReturn.iloc[:, ind] = ((SimpleReturn.iloc[:, ind] - FirstValidIndices[0, ind]) /
                                     FirstValidIndices[0, ind]) * 100
        TotalReturn[ind, 0] = ((SimpleReturn.iloc[-1, ind] - FirstValidIndices[0, ind]) / FirstValidIndices[
            0, ind]) * 100
        DurationYears[0, ind] = relativedelta(LastDate, namad_close.iloc[:, ind].FirstValidIndex()).years

    TotalReturn = pd.DataFrame(TotalReturn)
    TotalReturn.columns = [farsi.make_farsi_text('بازدهی کل')]      # Total return

    AnnualReturn = np.zeros(shape=(len(namad_close.columns), 1))
    for t in range(len(namad_close.columns)):
        if DurationYears[0, t] > 0:
            AnnualReturn[t, 0] = ((SimpleReturn.iloc[:, t][-1:] / 100 + 1) ** (
                    1 / DurationYears[0, t]) - 1) * 100

    AnnualReturn = pd.DataFrame(AnnualReturn)
    AnnualReturn.columns = [farsi.make_farsi_text('بازدهی سالانه')]     # Annual return
    # daily returns for these stocks
    CloseCopy = namad_close.copy()
    DailyReturn = CloseCopy.pct_change().dropna()

    # calculate sharpe ratio
    AnnualRF = 0.23  # annual rf := 0.23 (It can be changed based on financial people)
    DailyRFReturn = (1 + AnnualRF) ** (1 / 252) - 1
    DailyReturn_adj = DailyReturn - DailyRFReturn
    sharpe = np.zeros(shape=(len(namad_close.columns), 1))
    for t in range(len(namad_close.columns)):
        if DailyReturn_adj.iloc[:, t].std() == 0:
            sharpe[t, 0] = DailyReturn_adj.iloc[:, t].mean() / 0.01
        else:
            sharpe[t, 0] = DailyReturn_adj.iloc[:, t].mean() / DailyReturn_adj.iloc[:, t].std()

    sharpe = pd.DataFrame(sharpe)
    sharpe.columns = [farsi.make_farsi_text('نسبت شارپ')]       # Sharp ratio
    # calculating beta
    ref = RankedClose.iloc[:, -1].pct_change().dropna().tail(len(DailyReturn))  # index's daily return
    beta = np.zeros(shape=(len(namad_close.columns), 1))
    for t in range(len(namad_close.columns)):
        beta[t, 0] = np.cov(DailyReturn.iloc[:, t], ref)[0][1] / np.var(ref)

    beta = pd.DataFrame(beta)
    beta.columns = [farsi.make_farsi_text('بتا')]       # Beta

    # calculating alpha
    FirstValidIndexIndex = RankedClose.loc[RankedClose.iloc[:, -1].FirstValidIndex()][-1]
    SimpleReturnIndex = ((RankedClose.iloc[:, -1] - FirstValidIndexIndex) / FirstValidIndexIndex) * 100
    DurationYearsIndex = relativedelta(LastDate, RankedClose.iloc[:, -1].FirstValidIndex()).years
    AnnualReturnIndex = ((SimpleReturnIndex[-1:] / 100 + 1) ** (1 / DurationYearsIndex) - 1) * 100

    alpha = np.zeros(shape=(len(namad_close.columns), 1))
    for k, v in enumerate(beta.values):
        alpha[k, 0] = AnnualReturn.iloc[k, 0] / 100 - AnnualRF - v * (AnnualReturnIndex / 100 - AnnualRF)

    alpha = pd.DataFrame(alpha)
    alpha.columns = [farsi.make_farsi_text('آلفا')]         # Alpha

    volatility = []
    AverageDailyReturn = []
    for t in range(len(namad_close.columns)):
        volatility.append(DailyReturn.iloc[:, t].std())
        AverageDailyReturn.append(DailyReturn.iloc[:, t].mean())

    volatility = pd.DataFrame(volatility)
    if volatility.shape[1] == 0:
        continue
    volatility.columns = [farsi.make_farsi_text('نوسان قیمت روزانه')]   # Volatility for daily price

    AverageDailyReturn = pd.DataFrame(AverageDailyReturn)
    if AverageDailyReturn.shape[1] == 0:
        continue
    AverageDailyReturn.columns = [farsi.make_farsi_text('میانگین بازدهی روزانه')]   # Average of daily return

    results = pd.concat([TotalReturn, AnnualReturn, volatility, AverageDailyReturn, alpha, beta, sharpe],
                        axis=1)
    results[farsi.make_farsi_text('نماد')] = namad_close.columns
    results.set_index([farsi.make_farsi_text('نماد')], inplace=True)

    results.columns = ['بازدهی کل', 'بازدهی سالانه', 'انحراف معیار', 'میانگین بازدهی روزانه', 'آلفا', 'بتا',
                       'نسبت شارپ']

    cnx.close()

    engine = create_engine('mysql+mysqlconnector://root:password@host/database_name', echo=False)

    if group == 0:

        check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                               " 'database_name' AND table_name = 'proc_namad_result')")
        checkin = check.fetchall()
        if np.sum(checkin) != 0:
            engine.execute("DROP TABLE proc_namad_result")

        check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                               " 'database_name' AND table_name = 'proc_correlation')")
        checkin = check.fetchall()
        if np.sum(checkin) != 0:
            engine.execute("DROP TABLE proc_correlation")

        check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                               " 'database_name' AND table_name = 'proc_daily_return')")
        checkin = check.fetchall()
        if np.sum(checkin) != 0:
            engine.execute("DROP TABLE proc_daily_return")

        check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                               " 'database_name' AND table_name = 'proc_simple_return')")
        checkin = check.fetchall()
        if np.sum(checkin) != 0:
            engine.execute("DROP TABLE proc_simple_return")

        check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                               " 'database_name' AND table_name = 'proc_close')")
        checkin = check.fetchall()
        if np.sum(checkin) != 0:
            engine.execute("DROP TABLE proc_close")

        tablemeta = MetaData(engine)
        namad = Table('namad', tablemeta, autoload=True)
        namad_sector = Table('namad_sector', tablemeta, autoload=True)
        Results = Table('proc_namad_result', tablemeta,
                        Column('id', Integer, primary_key=True),
                        Column('sharpe_ratio', Float),
                        Column('alpha', Float),
                        Column('beta', Float),
                        Column('total_return', Float),
                        Column('AnnualReturn', Float),
                        Column('std', Float),
                        Column('daily_return_average', Float),
                        Column('proc_namad', String(64)),
                        Column('proc_name', String(64)),
                        Column('namad_id', ForeignKey('namad.id')),
                        Column('proc_sector', String(64)),
                        Column('namad_sector_id', ForeignKey('namad_sector.id'))
                        )

        Correlation = Table('proc_correlation', tablemeta,
                            Column('id', Integer, primary_key=True),
                            Column('correlation', Float),
                            Column('namad_1', String(64)),
                            Column('namad_id', ForeignKey('namad.id')),
                            Column('namad_2', String(64)),
                            Column('corr_to_namad', ForeignKey('namad.id')),
                            Column('proc_sector', String(64)),
                            Column('namad_sector_id', ForeignKey('namad_sector.id'))
                            )

        Daily = Table('proc_daily_return', tablemeta,
                      Column('id', Integer, primary_key=True),
                      Column('date', DateTime),
                      Column('daily_re', Float),
                      Column('proc_namad', String(64)),
                      Column('proc_name', String(64)),
                      Column('namad_id', ForeignKey('namad.id')),
                      Column('proc_sector', String(64)),
                      Column('namad_sector_id', ForeignKey('namad_sector.id'))
                      )

        Simple = Table('proc_simple_return', tablemeta,
                       Column('id', Integer, primary_key=True),
                       Column('date', DateTime),
                       Column('simple_re', Float),
                       Column('proc_namad', String(64)),
                       Column('proc_name', String(64)),
                       Column('namad_id', ForeignKey('namad.id')),
                       Column('proc_sector', String(64)),
                       Column('namad_sector_id', ForeignKey('namad_sector.id'))
                       )

        Close = Table('proc_close', tablemeta,
                      Column('id', Integer, primary_key=True),
                      Column('date', DateTime),
                      Column('close_price', Float),
                      Column('vol', Float),
                      Column('proc_namad', String(64)),
                      Column('proc_name', String(64)),
                      Column('namad_id', ForeignKey('namad.id')),
                      Column('proc_sector', String(64)),
                      Column('namad_sector_id', ForeignKey('namad_sector.id'))
                      )

        Base = automap_base(metadata=tablemeta)
        Base.prepare()
        session = sessionmaker()
        session.configure(bind=engine)
        Base.metadata.create_all(engine)
    s = session()

    SimpleReturn.fillna(method='ffill', inplace=True)
    SimpleReturn = SimpleReturn.astype(object).where(pd.notnull(SimpleReturn), None)

    close.fillna(method='ffill', inplace=True)
    close = close.astype(object).where(pd.notnull(close), None)

    volume.fillna(method='ffill', inplace=True)
    volume = volume.astype(object).where(pd.notnull(volume), None)

    DailyReturn = DailyReturn.astype(object).where(pd.notnull(DailyReturn), None)
    sharpe = sharpe.astype(object).where(pd.notnull(sharpe), None)
    alpha = alpha.astype(object).where(pd.notnull(alpha), None)
    beta = beta.astype(object).where(pd.notnull(beta), None)
    results = results.astype(object).where(pd.notnull(results), None)
    corr = corr.astype(object).where(pd.notnull(corr), None)

    tickers = tickers.reset_index(drop=True)

    DayDate = DailyReturn.index
    SimpleReturnDate = SimpleReturn.index
    CloseDate = close.index

    for namad in range(len(tickers)):

        ins1 = Results.insert().values(sharpe_ratio=sharpe.iloc[namad, 0], alpha=alpha.iloc[namad, 0],
                                       beta=beta.iloc[namad, 0], total_return=results.iloc[namad, 0],
                                       annual_return=results.iloc[namad, 1], std=results.iloc[namad, 2],
                                       daily_return_average=results.iloc[namad, 3], proc_namad=tickers.iloc[namad, 0],
                                       proc_name=tickers.iloc[namad, 1], namad_id=ID.iloc[0, namad],
                                       proc_sector=groups.iloc[group, 0], namad_sector_id=GroupsID.iloc[group, 0])
        engine.execute(ins1)

        for row in range(DailyReturn.shape[0]):
            ins2 = Daily.insert().values(date=DayDate[row], daily_re=DailyReturn.iloc[row, namad],
                                         proc_namad=tickers.iloc[namad, 0], proc_name=tickers.iloc[namad, 1],
                                         namad_id=ID.iloc[0, namad], proc_sector=groups.iloc[group, 0],
                                         namad_sector_id=GroupsID.iloc[group, 0])
            engine.execute(ins2)

        for row in range(SimpleReturn.shape[0]):
            ins3 = Simple.insert().values(date=SimpleReturnDate[row], simple_re=SimpleReturn.iloc[row, namad],
                                          proc_namad=tickers.iloc[namad, 0], proc_name=tickers.iloc[namad, 1],
                                          namad_id=ID.iloc[0, namad], proc_sector=groups.iloc[group, 0],
                                          namad_sector_id=GroupsID.iloc[group, 0])
            engine.execute(ins3)

        for row in range(close.shape[0]):
            ins4 = Close.insert().values(date=CloseDate[row], close_price=close.iloc[row, namad],
                                         proc_namad=tickers.iloc[namad, 0], proc_name=tickers.iloc[namad, 1],
                                         vol=volume.iloc[row, namad], namad_id=ID.iloc[0, namad],
                                         proc_sector=groups.iloc[group, 0],
                                         namad_sector_id=GroupsID.iloc[group, 0])
            engine.execute(ins4)

        corr_ind = 0
        for row in range(corr.shape[0]):
            ins5 = Correlation.insert().values(correlation=corr.iloc[row, namad], namad_1=tickers.iloc[namad, 0],
                                               namad_id=ID.iloc[0, namad], namad_2=tickers.iloc[corr_ind, 0],
                                               corr_to_namad=ID.iloc[0, corr_ind], proc_sector=groups.iloc[group, 0],
                                               namad_sector_id=GroupsID.iloc[group, 0])
            engine.execute(ins5)
            corr_ind = corr_ind + 1

    s.close()
    # WholeTickers = pd.concat([WholeTickers, tickers], ignore_index=True)
    CompletedTable = pd.concat([CompletedTable, close.iloc[:, 0:len(tickers)]], axis=1)

    table = close.iloc[:, 0:len(tickers)]
    del close
    del results

    table.fillna(method='ffill', inplace=True)

    num_simulations = 1000

    if group == 0:

        check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                               " 'database_name' AND table_name = 'proc_mc')")
        checkin = check.fetchall()
        if np.sum(checkin) != 0:
            engine.execute("DROP TABLE proc_mc")

        check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                               " 'database_name' AND table_name = 'proc_mc_days')")
        checkin = check.fetchall()
        if np.sum(checkin) != 0:
            engine.execute("DROP TABLE proc_mc_days")

        MCTablemeta = MetaData(engine)
        namad = Table('namad', MCTablemeta, autoload=True)
        namad_sector = Table('namad_sector', MCTablemeta, autoload=True)

        ProcDays = Table('proc_mc_days', MCTablemeta,
                          Column('id', Integer, primary_key=True),
                          Column('day', String(64))
                          )

        MonteCarlo = Table('proc_mc', MCTablemeta,
                           Column('id', Integer, primary_key=True),
                           Column('x_plt', Float),
                           Column('y_plt', Float),
                           Column('proc_mc_days_id', ForeignKey('proc_mc_days.id')),
                           Column('proc_namad', String(64)),
                           Column('proc_name', String(64)),
                           Column('namad_id', ForeignKey('namad.id')),
                           Column('proc_sector', String(64)),
                           Column('namad_sector_id', ForeignKey('namad_sector.id'))
                           )

        Base = automap_base(metadata=MCTablemeta)
        Base.prepare()
        session = sessionmaker()
        session.configure(bind=engine)
        Base.metadata.create_all(engine)
        s = session()

        for d in range(32):
            ins1 = ProcDays.insert().values(day='Day' + str(d + 1))
            engine.execute(ins1)

    s = session()
    for tick in range(table.shape[1]):  # arg:= MC_table.shape[1]

        MC_cl = MC(table.iloc[:, tick])

        x_plt = pd.DataFrame()
        y_plt = pd.DataFrame()
        NumBins = 20
        for i in range(MC_cl.shape[0]):
            n, bins = np.histogram(MC_cl.iloc[i, :], NumBins, normed=1)
            a = []
            for j in range(bins.shape[0] - 1):
                a.append((bins[j] + bins[j + 1]) / 2)
            a = pd.DataFrame(a)
            n = pd.DataFrame(n)
            x_plt = pd.concat([x_plt, a], axis=1)
            y_plt = pd.concat([y_plt, n], axis=1)

        for col in range(MC_cl.shape[0]):
            for row in range(x_plt.shape[0]):
                ins2 = MonteCarlo.insert().values(x_plt=x_plt.iloc[row, col], y_plt=y_plt.iloc[row, col],
                                                  proc_mc_days_id=col + 1, proc_namad=tickers.iloc[tick, 0],
                                                  proc_name=tickers.iloc[tick, 1], namad_id=ID.iloc[0, tick],
                                                  proc_sector=groups.iloc[group, 0],
                                                  namad_sector_id=GroupsID.iloc[group, 0])
                engine.execute(ins2)

    s.close()

    tickers_num = table.shape[1]

    returns = table.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_portfolios = 25000
    risk_free_rate = 0.23

    results, max_sharpe_allocation, min_vol_allocation, sdp, rp, sdp_min, rp_min = \
        display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

    results = pd.DataFrame(results)
    results = results.astype(object).where(pd.notnull(results), None)

    if group == 0:
        engine = create_engine('mysql+mysqlconnector://root:password@host/database_name', echo=False)

        check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                               " 'database_name' AND table_name = 'proc_sector_plt')")
        checkin = check.fetchall()
        if np.sum(checkin) != 0:
            engine.execute("DROP TABLE proc_sector_plt")

        check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                               " 'database_name' AND table_name = 'proc_sector_optim')")
        checkin = check.fetchall()
        if np.sum(checkin) != 0:
            engine.execute("DROP TABLE proc_sector_optim")

        check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                               " 'database_name' AND table_name = 'proc_sector_portfolio')")
        checkin = check.fetchall()
        if np.sum(checkin) != 0:
            engine.execute("DROP TABLE proc_sector_portfolio")

        SectorPortfolioTablemeta = MetaData(engine)
        namad = Table('namad', SectorPortfolioTablemeta, autoload=True)
        namad_sector = Table('namad_sector', SectorPortfolioTablemeta, autoload=True)

        Sector_Optimal = Table('proc_sector_optim', SectorPortfolioTablemeta,
                                 Column('id', Integer, primary_key=True),
                                 Column('sdp', Float),
                                 Column('rp', Float),
                                 Column('sdp_min', Float),
                                 Column('rp_min', Float),
                                 Column('proc_sector', String(64)),
                                 Column('namad_sector_id', ForeignKey('namad_sector.id'))
                                 )

        Sector_Portfolio = Table('proc_sector_portfolio', SectorPortfolioTablemeta,
                                   Column('id', Integer, primary_key=True),
                                   Column('proc_namad', String(64)),
                                   Column('proc_name', String(64)),
                                   Column('max_sharpe_allocation', Float),
                                   Column('min_vol_allocation', Float),
                                   Column('namad_id', ForeignKey('namad.id')),
                                   Column('proc_sector', String(64)),
                                   Column('namad_sector_id', ForeignKey('namad_sector.id'))
                                   )

        Sector_Plt = Table('proc_sector_plt', SectorPortfolioTablemeta,
                             Column('id', Integer, primary_key=True),
                             Column('ann_volatility', Float),
                             Column('ann_return', Float),
                             Column('proc_sector', String(64)),
                             Column('namad_sector_id', ForeignKey('namad_sector.id'))
                             )

        Base = automap_base(metadata=SectorPortfolioTablemeta)
        Base.prepare()
        session = sessionmaker()
        session.configure(bind=engine)
        Base.metadata.create_all(engine)
    s = session()

    if np.isnan(sdp) == 1:
        continue  # sdp = 0
    if np.isnan(rp) == 1:
        continue  # rp = 0
    if np.isnan(sdp_min) == 1:
        continue  # sdp_min = 0
    if np.isnan(rp_min) == 1:
        continue  # rp_min = 0

    min_vol_allocation = min_vol_allocation.astype(object).where(pd.notnull(min_vol_allocation), None)
    max_sharpe_allocation = max_sharpe_allocation.astype(object).where(pd.notnull(max_sharpe_allocation), None)
    for tick in range(len(tickers)):  # arg:= len(tickers)

        ins1 = Sector_Portfolio.insert().values(proc_namad=tickers.iloc[tick, 0],
                                                  proc_name=tickers.iloc[tick, 1],
                                                  max_sharpe_allocation=max_sharpe_allocation.iloc[0, tick],
                                                  min_vol_allocation=min_vol_allocation.iloc[0, tick],
                                                  namad_id=ID.iloc[0, tick],
                                                  proc_sector=groups.iloc[group, 0],
                                                  namad_sector_id=GroupsID.iloc[group, 0]
                                                  )
        engine.execute(ins1)

    for row in range(results.shape[1]):
        ins2 = Sector_Plt.insert().values(ann_volatility=results.iloc[0, row], ann_return=results.iloc[1, row],
                                            proc_sector=groups.iloc[group, 0],
                                            namad_sector_id=GroupsID.iloc[group, 0]
                                            )
        engine.execute(ins2)

    ins3 = Sector_Optimal.insert().values(sdp=sdp, rp=rp, sdp_min=sdp_min, rp_min=rp_min,
                                            proc_sector=groups.iloc[group, 0],
                                            namad_sector_id=GroupsID.iloc[group, 0]
                                            )
    engine.execute(ins3)

    s.close()

# Cleaning due to memory efficiency
del CloseTick, str_data, num_data, data, crude_data, DailyReturn, SimpleReturn, results

table = CompletedTable
del CompletedTable
table.fillna(method='ffill', inplace=True)

tickers_num = table.shape[1]

returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.02

results, max_sharpe_allocation, min_vol_allocation, sdp, rp, sdp_min, rp_min = \
    display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

engine = create_engine('mysql+mysqlconnector://root:password@host/database_name', echo=False)

check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                       " 'database_name' AND table_name = 'proc_plt')")
checkin = check.fetchall()
if np.sum(checkin) != 0:
    engine.execute("DROP TABLE proc_plt")

check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                       " 'database_name' AND table_name = 'proc_optim')")
checkin = check.fetchall()
if np.sum(checkin) != 0:
    engine.execute("DROP TABLE proc_optim")

check = engine.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema ="
                       " 'database_name' AND table_name = 'proc_portfolio_whole')")
checkin = check.fetchall()
if np.sum(checkin) != 0:
    engine.execute("DROP TABLE proc_portfolio_whole")

PortfolioTablemeta = MetaData(engine)
namad = Table('namad', PortfolioTablemeta, autoload=True)

Optimal = Table('proc_optim', PortfolioTablemeta,
                Column('id', Integer, primary_key=True),
                Column('sdp', Float),
                Column('rp', Float),
                Column('sdp_min', Float),
                Column('rp_min', Float)
                )

Portfolio = Table('proc_portfolio_whole', PortfolioTablemeta,
                  Column('id', Integer, primary_key=True),
                  Column('proc_namad', String(64)),
                  Column('proc_name', String(64)),
                  Column('max_sharpe_allocation', Float),
                  Column('min_vol_allocation', Float),
                  Column('namad_id', ForeignKey('namad.id'))
                  )

Plt = Table('proc_plt', PortfolioTablemeta,
            Column('id', Integer, primary_key=True),
            Column('ann_volatility', Float),
            Column('ann_return', Float),
            )

Base = automap_base(metadata=PortfolioTablemeta)
Base.prepare()
session = sessionmaker()
session.configure(bind=engine)
Base.metadata.create_all(engine)
s = session()

results = pd.DataFrame(results)
results = results.astype(object).where(pd.notnull(results), None)

for tick in range(len(WholeTickers)):  # arg:= len(tickers)

    ins1 = Portfolio.insert().values(proc_namad=WholeTickers.iloc[tick, 0], proc_name=WholeTickers.iloc[tick, 1],
                                     max_sharpe_allocation=max_sharpe_allocation.iloc[0, tick],
                                     min_vol_allocation=min_vol_allocation.iloc[0, tick],
                                     namad_id=WholeID.iloc[0, tick])
    engine.execute(ins1)

for row in range(results.shape[1]):
    ins2 = Plt.insert().values(ann_volatility=results.iloc[0, row], ann_return=results.iloc[1, row])
    engine.execute(ins2)

ins3 = Optimal.insert().values(sdp=sdp, rp=rp, sdp_min=sdp_min, rp_min=rp_min)
engine.execute(ins3)

s.close()


