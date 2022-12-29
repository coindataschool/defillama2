import datetime as dt
import numpy as np
import pandas as pd 

def get_tokens_hist_prices(self, token_addrs_n_chains, start, end, freq='hourly'):
    """Get historical hourly or daily prices of tokens by contract address.
    Uses get_tokens_hist_snapshot_prices() to download iteratively since
    DeFiLlama currently doesn't offer an API for bulk download. If you want 
    to get daily open or close prices, use get_daily_open_close() instead 
    since it's faster.
    
    Parameters
    ----------
    token_addrs_n_chains : dictionary
        Each key is a token address; each value is a chain where the token 
        address resides. If getting price from coingecko, use token name as 
        key and 'coingecko' as value. For example, 
        {'0xdF574c24545E5FfEcb9a659c229253D4111d87e1':'ethereum',
            'ethereum':'coingecko'}
    start : string
        Start date, for example, '2021-01-01'
    end : string
        End date, for example, '2022-01-01'
    freq : string
        Data granularity, 'hourly' (default) or 'daily'. Does NOT support 
        other values at the moment.

    Returns 
    -------
    data frame
    """
    start = dt.datetime.strptime(start, '%Y-%m-%d')
    end   = dt.datetime.strptime(end, '%Y-%m-%d')
    timestamps = pd.date_range(start, end, freq='60min')
    timestamps = timestamps[timestamps < dt.datetime.now()-pd.Timedelta(hours=4)] # give 4 hours buffer to ensure DeFiLlama data are available
        
    # download historical hourly data
    df = pd.concat(self.get_tokens_hist_snapshot_prices(token_addrs_n_chains, dttm) for dttm in timestamps)

    # clean data so that the resulting frame has 
    #   - each row is a datetime
    #   - each column is a token
    #   - each value is a price (open or close)
    df = df.reset_index()
    df['datetime'] = [elt.round(freq='H') for elt in df.timestamp]
    # df[['timestamp', 'datetime']].head(10)
    # there can be duplicates in the col `datetime`, so take their avg price
    df = df.groupby(['datetime', 'symbol'])['price'].mean() 
    df = df.reset_index().pivot(index='datetime', columns='symbol', values='price')
    df.columns.name = None
    df.index = pd.to_datetime(df.index, utc=True)
    
    # derive daily prices if user requests them instead of hourly data
    if freq == 'daily':
        symbols = df.columns
        # calculate daily prices using hourly data
        daily_open = df.asfreq('1d')
        daily_low = df.resample('D').min()
        daily_high = df.resample('D').max()
        daily_close = daily_open.shift(-1)
        daily_med = df.resample('D').median()
        daily_avg = df.resample('D').mean()
        daily_std = df.resample('D').std()
        # assign header 
        daily_open.columns = pd.MultiIndex.from_tuples([('open', symbol) for symbol in symbols])
        daily_low.columns = pd.MultiIndex.from_tuples([('low', symbol) for symbol in symbols])
        daily_high.columns = pd.MultiIndex.from_tuples([('high', symbol) for symbol in symbols])
        daily_close.columns = pd.MultiIndex.from_tuples([('close', symbol) for symbol in symbols])
        daily_med.columns = pd.MultiIndex.from_tuples([('median', symbol) for symbol in symbols])
        daily_avg.columns = pd.MultiIndex.from_tuples([('mean', symbol) for symbol in symbols])
        daily_std.columns = pd.MultiIndex.from_tuples([('std', symbol) for symbol in symbols])
        # join together
        df = pd.concat([daily_open, daily_low, daily_high, daily_close, daily_med, daily_avg, daily_std], axis=1)
    return df

