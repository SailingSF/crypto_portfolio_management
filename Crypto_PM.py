import numpy as np
import pandas as pd

def sharpe_portfolio(portfolio, rfr):
    '''
    Takes a pandas series of prices from a portfolio and the risk free return (rfr) and returns the sharpe ratio
    Sharpe ratio is defined as return over the standard deviation of returns
    '''
    #returns and ratio are not annualized, will give sharpe for entire series
    
    returns = (portfolio.iloc[-1]-portfolio.iloc[0])/portfolio.iloc[0]
    std = portfolio.pct_change().std()
    
    return (returns-rfr)/std

def sortino_portfolio(portfolio, target):
    '''
    Takes a pandas series of prices from a portfolio and a target rate of return (often the risk free rate) and returns the Sortino Ratio
    The Sortino Ratio takes downside standard deviation as its risk parameter, only penalizing downward moves
    '''
    #returns and ratio are not annualized, will give Sortino for entire series

    returns = portfolio.pct_change()
    return_f = (portfolio.iloc[-1]-portfolio.iloc[0])/portfolio.iloc[0]
    down_returns = returns.where(returns<target).dropna()
    risk = down_returns.std()
    
    return (return_f - target)/risk

def beta_calc(portfolio, benchmark):
    '''
    calculates a beta between a given series of prices and a benchmark
    normally the benchmark is the S&P 500 although in crypto this can be something like BTC
    '''
    covariance = portfolio.pct_change().cov(benchmark.pct_change())
    
    return covariance/benchmark.pct_change().var()

def alpha_calc(portfolio, benchmark, rf=0.0):
    '''
    calculates Jensen Alpha based on:
    portfolio and benchmark, risk free rate of return: rf
    calculates beta
    annualizes returns
    '''

    #making sure portfolio and benchmark are the same size
    length = len(portfolio)
    
    if length > len(benchmark):
        length = len(benchmark)
        portfolio = portfolio.iloc[-length::]
    elif length < len(benchmark):
        benchmark = benchmark.iloc[-length::]
    
    #create annualized returns
    rp = (1+((portfolio.iloc[-1]-portfolio.iloc[0])/portfolio.iloc[0]))**(365/length)-1
    rb = (1+((benchmark.iloc[-1]-benchmark.iloc[0])/benchmark.iloc[0]))**(365/length)-1
    
    #calculate beta
    beta = beta_calc(portfolio, benchmark)

    #print statements for sanity check and verboseness
    print(f"Portfolio return: {rp}")
    print(f"Benchmark return: {rb}")
    print(f"Porfolio Beta: {beta}")
    
    alpha = rp - (rf + beta*(rb - rf))
    
    return alpha

def simple_forward_prices(start_price, desired_return, volatility, days=365, precision = 0.01):
    '''
    Function for creating a sample series of prices with a desired return and volatility
    Takes a starting price, the desired return, the desired volatility, the number of days/units of time, and a precision of the final result
    The precision will have the final return +/- the given percentage
    source: https://www.analyticsvidhya.com/blog/2021/05/create-a-dummy-stock-market-using-geometric-brownian-motion-in-python/
    '''
    desired_price = start_price*(1+desired_return)

    #loops a maximum of 1000 times to create a series within the precision of the desired return
    for k in range(0,1000):
        returns = np.random.normal(loc=desired_return/days, scale=volatility, size=days)
        prices = (1+returns).cumprod()
        prices = prices/prices[0]*start_price
        final_price = prices[-1]
        if final_price >= desired_price*(1-precision) and final_price <= desired_price*(1+precision):
            break
        else:
            pass

    return prices

def yield_value(prices, apy):
    '''
    Creates new pandas Series of prices given one series and an APY
    Daily compounds.
    Vectorized for performance in pandas
    '''

    #normalized prices, so result needs to multiply by an initial amount
    prices = prices/prices.iloc[0]
    prices.name = None #set name to None so we can reset name in dataframe creation
    df = pd.DataFrame(data=prices, columns=['prices'])
    
    daily_rate = (apy+1)**(1/365)-1
    df['rate'] = daily_rate
    df['return'] = (1+df['rate']).cumprod()
    df['yield'] = df['prices']*df['return']
    
    return df['yield']

def lp_value(prices, invest, apy):
    '''
    takes token prices, amount invested, and the quoted yield in apy
    returns values of the lp position for token and stable pair
    also returns comparisons for impermanent loss calculation
    '''
    
    price_enter = prices.iloc[0]
    
    #as prices given are one token to dollars token*dollars = 1*dollars = price_enter
    k = (invest/2)*((invest/2)/price_enter)
    #ratio as tokens to dollars which is the same as "prices"
    tokens = (k/prices)**(1/2)
    dollars = (k*prices)**(1/2)
    
    lp_value = (tokens*prices+dollars)
    lp_value.name = None
    full_hodl = invest/price_enter*prices
    diverse_hodl = full_hodl/2+(invest/2)
    
    df = pd.DataFrame(lp_value, columns=['lp_value'])
    df['lp_value_yield'] = yield_value(df['lp_value'], apy)*invest
    
    df['full_hodl'] = full_hodl
    df['diverse_hodl'] = diverse_hodl
    
    return df

def token_lp_portfolio(prices, ratio, apy, invest = 1000):
    '''
    Takes pandas series of token prices and a ratio of portfolio in LP with a stable coin, the APY, and an investment amount
    Returns the value of the given portfolio with the percentage in an LP combination
    '''
    
    token_invest = invest*(1-ratio)
    lp_invest = invest*ratio
    lp = lp_value(prices, lp_invest, apy)['lp_value_yield']
    portfolio = ((prices/prices.iloc[0])*token_invest) + lp
    
    return portfolio