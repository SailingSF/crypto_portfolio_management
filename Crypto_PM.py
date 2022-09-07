import numpy as np
import pandas as pd

def sharpe_portfolio(portfolio, rfr = 0.0):
    '''
    Takes a pandas series of prices from a portfolio and the risk free return (rfr) and returns the sharpe ratio
    Sharpe ratio is defined as return over the standard deviation of returns
    '''
    #returns and ratio are annualized, will give sharpe for entire series
    length = len(portfolio)

    returns = (1+((portfolio.iloc[-1]-portfolio.iloc[0])/portfolio.iloc[0]))**(365/length)-1
    std = portfolio.pct_change().std()*np.sqrt(365)
    
    return (returns-rfr)/std

def sortino_portfolio(portfolio, target = 0.0):
    '''
    Takes a pandas series of prices from a portfolio and a target rate of return (often the risk free rate) and returns the Sortino Ratio
    The Sortino Ratio takes downside standard deviation as its risk parameter, only penalizing downward moves
    '''
    #returns and ratio are annualized, will give Sortino for entire series
    length = len(portfolio)
    returns = portfolio.pct_change()
    return_f = (1+((portfolio.iloc[-1]-portfolio.iloc[0])/portfolio.iloc[0]))**(365/length)-1
    down_returns = returns.where(returns<target).dropna()
    risk = down_returns.std()*np.sqrt(365)
    
    return (return_f - target)/risk

def beta_calc(portfolio, benchmark):
    '''
    calculates a beta between a given series of prices and a benchmark
    normally the benchmark is the S&P 500 although in crypto this can be something like BTC
    '''
    covariance = portfolio.pct_change().cov(benchmark.pct_change())
    
    return covariance/(benchmark.pct_change().var())

def alpha_calc(portfolio, benchmark, rf=0.0):
    '''
    calculates Jensen Alpha based on:
    portfolio and benchmark, risk free rate of return: rf
    calculates beta
    annualizes returns
    '''

    #making sure portfolio and benchmark are the same size, and have a daily index
    daily_index = pd.date_range(start=portfolio.index.min(), end=portfolio.index.max())
    portfolio = portfolio.reindex(daily_index).fillna(method='ffill')
    length = len(portfolio)
    benchmark = benchmark.reindex(portfolio.index).fillna(method='ffill').dropna()
    
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

def lp_stable_value(prices, invest, apy):
    '''
    takes token prices, amount invested, and the quoted yield in apy
    returns values of the lp position for token and STABLE pair
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

def lp_tokens_value(prices0, prices1, invest, apy):
    '''
    takes two token prices, amount invested, and the quoted yield in apy
    returns values of the lp position for token/token pair
    also returns comparisons for impermanent loss calculation
    '''
    
    price_enter0 = prices0.iloc[0]
    price_enter1 = prices1.iloc[0]
    
    k = ((invest/2)/price_enter1)*((invest/2)/price_enter0) #amount of tokens entered with
    ratio = prices0/prices1 #ratio as tokens0 to tokens1
    
    tokens0 = (k/ratio)**(1/2)
    tokens1 = (k*ratio)**(1/2)
    
    lp_value = tokens0*prices0+tokens1*prices1
    lp_value.name = None
    
    hodl = (invest/2/price_enter0)*prices0 + (invest/2/price_enter1)*prices1
    
    df = pd.DataFrame(lp_value, columns=['lp_value'])
    df['lp_value_yield'] = yield_value(df['lp_value'], apy)*invest
    
    df['hodl'] = hodl
    
    return df

def token_lp_portfolio(prices, ratio, apy, invest = 1000):
    '''
    Takes pandas series of token prices and a ratio of portfolio in LP with a stable coin, the APY, and an investment amount
    Returns the value of the given portfolio with the percentage in an LP combination
    '''
    
    token_invest = invest*(1-ratio)
    lp_invest = invest*ratio
    lp = lp_stable_value(prices, lp_invest, apy)['lp_value_yield']
    portfolio = ((prices/prices.iloc[0])*token_invest) + lp
    
    return portfolio

def uni_v3_lp(prices0, prices1, tick_l, tick_h, invest, apy):
    '''
    Function to take the inputs and prices serieses of two assets in a UNI V3 LP position
    Outputs historical value of this position with a given static fee APY
    Inputs are prices series of two assets with a common counter asset(USD/EUR/etc.) and outputs value in same counter asset
    '''
    #getting price of 0 in terms of 1, if using USD equivalent passing a price series of np.ones(len(prices0)) is suggested
    ratio = prices0/prices1
    
    if ratio.iloc[0] >= tick_l and ratio.iloc[0] <= tick_h:
        
        #setting dummy amount of 0 to find liquidity mix
        amount0 = 1
        L = amount0 * np.sqrt(ratio.iloc[0]) * np.sqrt(tick_h) / (np.sqrt(tick_h) - np.sqrt(ratio.iloc[0]))
        amount1 = L * (np.sqrt(ratio.iloc[0]) - np.sqrt(tick_l))
        
        #get percentage mix based on Liquidity of X (0) value.
        mix0 = amount0*prices0.iloc[0]/(amount0*prices0.iloc[0] + amount1*prices1.iloc[0])
        value0 = mix0*invest
        value1 = (1-mix0)*invest
        
        amount0 = value0/prices0.iloc[0]
        amount1 = value1/prices1.iloc[0]
        
        #getting liquidity based on 0 asset. Assuming Lx == Ly here.
        L = amount0 * np.sqrt(ratio.iloc[0]) * np.sqrt(tick_h) / (np.sqrt(tick_h) - np.sqrt(ratio.iloc[0]))
        #L used to estimate values of assets 0 and 1 when price is in range

    elif ratio.iloc[0] < tick_l:

        #case when current price is below minimum tick value so we are single sided in asset0
        amount0 = invest/prices0.iloc[0]
        amount1 = 0
        
        L = amount0 * np.sqrt(ratio.iloc[0]) * np.sqrt(tick_h) / (np.sqrt(tick_h) - np.sqrt(ratio.iloc[0]))
        print(amount0)
    elif ratio.iloc[0] > tick_h:

        #case when current price is above maximum tick value so we are single sided in asset1
        amount1 = invest/prices1.iloc[0]
        amount0 = 0
        
        L = amount1 / (np.sqrt(ratio.iloc[0]) - np.sqrt(tick_l))
        print(amount1)
    else:
        #for debugging
        print("something weird happened")
        print(ratio) 
    
    
    def amount0_in_pool(L, price0, price1, tick_l, tick_h):
        #gets an amount in pool for the lambda expression later, optimized for pandas
        ratio = price0/price1

        if ratio > tick_h:
            amount0 = 0
        else:
            amount0 = L*(np.sqrt(tick_h) - np.sqrt(ratio)) / (np.sqrt(ratio) * np.sqrt(tick_h))

        return amount0

    def amount1_in_pool(L, price0, price1, tick_l, tick_h):
        #gets an amount in pool for the lambda expression later, optimized for pandas
        ratio = price0/price1
        if ratio < tick_l:
            amount1 = 0
        else:
            amount1 =  L*(np.sqrt(ratio) - np.sqrt(tick_l))

        return amount1
    
    #make price dataframe for use in lambda
    df_price = pd.concat([prices0.rename('prices0'), prices1.rename('prices1')], axis=1)
    df = pd.DataFrame()
    
    #use of lambda to get amounts in assets 0 and 1
    df['asset0_amount'] = df_price.apply(lambda x: amount0_in_pool(L, x['prices0'], x['prices1'], tick_l, tick_h), axis=1)
    df['asset1_amount'] = df_price.apply(lambda x: amount1_in_pool(L, x['prices0'], x['prices1'], tick_l, tick_h), axis=1)
    
    #value of each invidivual asset
    df['asset0_value'] = df['asset0_amount'] * df_price['prices0']
    df['asset1_value'] = df['asset1_amount'] * df_price['prices1']
    
    #value of pool only, no fees
    df['pool_value'] = df['asset0_value'] + df['asset1_value']
    
    #apply yield only when in range
    daily_rate = (apy+1)**(1/365)-1 #get daily rate from annualized APY
    df['daily_fees'] = df.apply(lambda x: daily_rate*x['pool_value'] if (x['asset0_amount'] > 0) and (x['asset1_amount'] > 0) else 0, axis=1)
    
    #total value as pool value plus fees accumulated to that point
    df['total_value'] = df['pool_value'] + df['daily_fees'].cumsum()
    
    #create hodl value for IL calculation and comparison
    #normalize prices and do 50/50 split between the two. Possibly compare to what actual initial mix was?
    df['hodl'] = (invest/2)*(prices0/prices0.iloc[0]) + (invest/2)*(prices1/prices1.iloc[0])
    
    return df