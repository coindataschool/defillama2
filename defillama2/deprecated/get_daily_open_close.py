import datetime as dt
import numpy as np
import pandas as pd 

def get_daily_open_close(self, token_addrs_n_chains, start, end, kind='close'):
    """Get historical daily open and close prices of tokens by contract address.
    Uses get_tokens_hist_snapshot_prices() to download iteratively since
    DeFiLlama currently doesn't offer an API for bulk download. 
    
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
    kind : string
        Either 'close' (default, at 23:59:59) or 'open' (at 00:00:00). Does NOT 
        support other values at the moment.

    Returns 
    -------
    data frame
    """
    start = dt.datetime.strptime(start, '%Y-%m-%d')
    end   = dt.datetime.strptime(end, '%Y-%m-%d')
    if (end.date() == dt.date.today()) and (kind == 'close'): 
        end -= pd.Timedelta(days=1)
    dates = pd.date_range(start, end)

    if kind == 'close':
        dttms = [date.replace(hour=23, minute=59, second=59) for date in dates] 
    elif kind == 'open':
        dttms = [date.replace(hour=0, minute=0, second=0) for date in dates] 
    else: 
        raise Exception("Only 'open' or 'close' are supported.")

    # download historical snapshots one by one    
    df = pd.concat(self.get_tokens_hist_snapshot_prices(token_addrs_n_chains, dttm) for dttm in dttms)

    # clean data so that the resulting frame has 
    #   - each row is a date
    #   - each column is a token
    #   - each value is a price (open or close)
    df = df.reset_index()
    if kind == 'close':
        df['date'] = np.where(df.timestamp.dt.hour == 0, 
                                df.timestamp.dt.date - pd.Timedelta(days=1), 
                                df.timestamp.dt.date)
    if kind == 'open':
        df['date'] = np.where(df.timestamp.dt.hour == 0, 
                                df.timestamp.dt.date, 
                                df.timestamp.dt.date + pd.Timedelta(days=1))
    df = df.groupby(['date', 'symbol'])['price'].mean()
    df = df.reset_index().pivot(index='date', columns='symbol', values='price')
    df.columns.name = None
    df.index = pd.to_datetime(df.index, utc=True)
    return df
