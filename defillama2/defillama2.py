import requests
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta

TVL_BASE_URL = "https://api.llama.fi"
COINS_BASE_URL = "https://coins.llama.fi"
STABLECOINS_BASE_URL = "https://stablecoins.llama.fi"
YIELDS_BASE_URL = "https://yields.llama.fi"
ABI_DECODER_BASE_URL = "https://abi-decoder.llama.fi"

class DefiLlama:
    """ 
    Implements methods for calling DeFiLlama APIs and cleaning returned data. 
    """

    def __init__(self):
        self.session = requests.Session()

    def _tidy_frame_tvl(self, df):
        """Set `date` of input data frame as index and shorten TVL column name.
        
        Parameters
        ----------
        df : data frame 
            Must contains two columns: `date` and 'totalLiquidityUSD'.
        Returns 
        -------
        data frame
        """
        df['date'] = df.date.apply(lambda x: dt.datetime.utcfromtimestamp(int(x)))
        df = df.set_index('date').rename(columns={'totalLiquidityUSD': 'tvl'})
        return df

    def _get(self, api_name, endpoint, params=None):
        """Send 'GET' request.

        Parameters
        ----------
        api_name : string
            Which API to call. Possible values are 'TVL', 'COINS', 'STABLECOINS',
            'YIELDS', and 'ABI_DECODER'. Each type has a different base url.
        endpoint : string 
            Endpoint to be added to base URL.
        params : dictionary
            HTTP request parameters.
        
        Returns
        -------
        JSON response
        """
        if api_name == 'TVL':
            url = TVL_BASE_URL + endpoint
        elif api_name == 'COINS':
            url = COINS_BASE_URL + endpoint  
        elif api_name == 'STABLECOINS':    
            url = STABLECOINS_BASE_URL + endpoint
        elif api_name == 'YIELDS':
            url = YIELDS_BASE_URL + endpoint 
        else: 
            url = ABI_DECODER_BASE_URL + endpoint
        return self.session.request('GET', url, params=params, timeout=30).json()

    def get_protocol_curr_tvl(self, protocol):
        """Get current TVL of a protocol.

        Parameters
        ----------
        protocol : string
            Protocol name.
        
        Returns 
        -------
        float
        """
        return self._get('TVL', f'/tvl/{protocol}')

    def get_chains_curr_tvl(self):
        """Get current TVL of all chains.
        
        Returns 
        -------
        data frame
        """
        resp = self._get('TVL', f'/chains/')
        df = pd.DataFrame(resp).loc[:, ['name', 'tokenSymbol', 'tvl']]
        df = df.rename(columns={'name':'chain', 'tokenSymbol':'token'})
        df = df.set_index('chain')
        return df

    def get_defi_hist_tvl(self):
        """Get historical TVL of DeFi on all chains.

        Returns 
        -------
        data frame
        """
        resp = self._get('TVL', '/charts')
        df = pd.DataFrame(resp)
        return self._tidy_frame_tvl(df)

    def get_chain_hist_tvl(self, chain):
        """Get historical TVL of a chain.

        Parameters
        ----------
        chain : string
            Chain name.
        
        Returns 
        -------
        data frame
        """
        resp = self._get('TVL', f'/charts/{chain}')
        df = pd.DataFrame(resp)
        return self._tidy_frame_tvl(df)
        
    def get_protocols(self):
        """Get detailed information on all protocols. 
        
        Returns 
        -------
        data frame
        """
        return pd.DataFrame(self._get('TVL', '/protocols'))

    def get_protocols_fundamentals(self):
        """Get current TVL, MCap, FDV, 1d and 7d TVL % change on all protocols.
        
        Parameters
        ----------
        protocol : string
            Protocol name.
        
        Returns 
        -------
        data frame
        """
        df = pd.DataFrame(self._get('TVL', '/protocols'))
        cols = ['name', 'symbol', 'chain', 'category', 'chains', 
                'tvl', 'change_1d', 'change_7d', 
                'fdv', 'mcap', 'forkedFrom']
        df = df.loc[:, cols].rename(columns={'forkedFrom':'forked_from'})
        return df

    def get_protocol(self, protocol):
        """Get detailed info on a protocol and breakdowns by token and chain.
        
        Parameters
        ----------
        protocol : string
            Protocol name.
        
        Returns 
        -------
        dictionary
        """
        return self._get('TVL', f'/protocol/{protocol}')

    def get_protocol_curr_tvl_by_chain(self, protocol):
        """Get current TVL of a protocol.

        Parameters
        ----------
        protocol : string
            Protocol name.
        
        Returns 
        -------
        data frame
        """
        dd = self.get_protocol(protocol)['currentChainTvls']
        if 'staking' in dd:
            dd.pop('staking')
        ss = pd.Series(dd)
        ss.name='tvl'
        return ss.to_frame()
    
    def get_protocol_hist_tvl_by_chain(self, protocol):
        """Get historical TVL of a protocol by chain.

        Parameters
        ----------
        protocol : string
            Protocol name.
        
        Returns 
        -------
        dict of data frames
        """
        dd = self.get_protocol(protocol)
        d1 = dd['currentChainTvls']
        if 'staking' in d1:
            d1.pop('staking')
        chains = list(d1.keys())
        return {chain: self._tidy_frame_tvl(pd.DataFrame(dd['chainTvls'][chain]['tvl'])) for chain in chains}

    def _tidy_frame_price(self, resp):
        # convert json response to data frame
        ha = pd.DataFrame([item.split(':') for item in resp['coins'].keys()])
        ha.columns = ['chain', 'token_address']
        df = ha.join(pd.DataFrame([v for k, v in resp['coins'].items()]))
        # convert epoch timestamp to human-readable datetime
        df['timestamp'] = df.timestamp.apply(
            lambda x: dt.datetime.utcfromtimestamp(int(x)))
        return df

    def get_tokens_curr_prices(self, token_addrs_n_chains):
        """Get current prices of tokens by contract address.

        Parameters
        ----------
        token_addrs_n_chains : dictionary
            Each key is a token address; each value is a chain where the token 
            address resides. If getting price from coingecko, use token name as 
            key and 'coingecko' as value. For example, 
            {'0xdF574c24545E5FfEcb9a659c229253D4111d87e1':'ethereum',
             'ethereum':'coingecko'}

        Returns 
        -------
        data frame
        """
        ss = ','.join([v + ':' +k for k, v in token_addrs_n_chains.items()])
        resp = self._get('COINS', f'/prices/current/{ss}')
        df = self._tidy_frame_price(resp)
        df = df.loc[:, ['timestamp', 'symbol', 'price', 'confidence', 'chain', 
                        'token_address', 'decimals']]
        df = df.set_index('timestamp')
        return df

    def get_tokens_hist_snapshot_prices(self, token_addrs_n_chains, timestamp):
        """Get historical snapshot prices of tokens by contract address.

        Parameters
        ----------
        token_addrs_n_chains : dictionary
            Each key is a token address; each value is a chain where the token 
            address resides. If getting price from coingecko, use token name as 
            key and 'coingecko' as value. For example, 
            {'0xdF574c24545E5FfEcb9a659c229253D4111d87e1':'ethereum',
             'ethereum':'coingecko'}
        timestamp : string
            Human-readable timestamp, for example, '2021-09-25 00:27:53'

        Returns 
        -------
        data frame
        """
        ss = ','.join([v + ':' +k for k, v in token_addrs_n_chains.items()])
        unix_ts = pd.to_datetime(timestamp).value / 1e9
        resp = self._get('COINS', f'/prices/historical/{unix_ts}/{ss}')
        df = self._tidy_frame_price(resp)
        df = df.loc[:, ['timestamp', 'symbol', 'price', 'chain', 'token_address', 'decimals']]
        df = df.set_index('timestamp')
        return df 

    def get_tokens_hist_prices(self, token_addrs_n_chains, start, end, type='close'):
        """Get historical prices of tokens by contract address.
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
        type : string
            Price type, either 'close' (default) or 'open'. Does NOT support 
            other values at the moment.

        Returns 
        -------
        data frame
        """
        start = dt.datetime.strptime(start, '%Y-%m-%d')
        end   = dt.datetime.strptime(end, '%Y-%m-%d')
        dates = pd.date_range(start, end)

        if type == 'close':
            dttms = [date.replace(hour=23, minute=59, second=59) for date in dates] 
        elif type == 'open':
            dttms = [date.replace(hour=0, minute=0, second=0) for date in dates] 
        else: 
            raise Exception("Only 'open' or 'close' are supported for `type`.")

        # download historical snapshots one by one    
        df = pd.concat(self.get_tokens_hist_snapshot_prices(token_addrs_n_chains, dttm) for dttm in dttms)

        # clean data so that the resulting frame has 
        #   - each row is a date
        #   - each column is a token
        #   - each value is a price (open or close)
        df = df.reset_index()
        if type == 'close':
            df['date'] = np.where(df.timestamp.dt.hour == 0, 
                                  df.timestamp.dt.date - relativedelta(days=1), 
                                  df.timestamp.dt.date)
        if type == 'open':
            df['date'] = np.where(df.timestamp.dt.hour == 0, 
                                  df.timestamp.dt.date, 
                                  df.timestamp.dt.date + relativedelta(days=1))
        df = df.groupby(['date', 'symbol'])['price'].mean()
        df = df.reset_index().pivot(index='date', columns='symbol', values='price')
        df.columns.name = None
        df.index = pd.to_datetime(df.index)
        return df

    def get_closest_block(self, chain, timestamp):
        """Get the closest block to a timestamp.

        Parameters
        ----------
        chain : string
            Name of the chain.
        timestamp : string
            Human-readable timestamp, for example, '2021-09-25 00:27:53'.

        Returns 
        -------
        data frame
        """
        unix_ts = pd.to_datetime(timestamp).value / 1e9
        resp = self._get('COINS', f'/block/{chain}/{unix_ts}')
        df = pd.DataFrame(resp, index=range(1))
        df['timestamp'] = df.timestamp.apply(
            lambda x: dt.datetime.utcfromtimestamp(int(x)))
        return df

    def get_stablecoins_circulating(self, include_price=False):
        """Get the circulating amounts for all stablecoins.

        Parameters
        ----------
        include_price : logical (default=False)
            Whether to include current stablecoin prices. Seems like it doesn't
            do anything and the returned data doesn't return current price even 
            set to True.

        Returns 
        -------
        data frame
        """
        resp = self._get('STABLECOINS', f'/stablecoins?includePrices={include_price}')
        lst = resp['peggedAssets']
        
        res = []
        for d0 in lst:
            _ = d0.pop('chainCirculating')
            _ = d0.pop('chains')
            res.append(pd.DataFrame(d0).reset_index(drop=True))
        df = pd.concat(res)
        df['id'] = df.id.astype(int)
        df = df.set_index('id')
        return df

    def get_stablecoins_circulating_by_chain(self, include_price=False):
        """Get the circulating amounts for all stablecoins, broken down by chain.

        Parameters
        ----------
        include_price : logical (default=False)
            Whether to include current stablecoin prices. Seems like it doesn't
            do anything and the returned data doesn't return current price even 
            set to True.

        Returns 
        -------
        dictionary where the keys are stablecoin symbols and values are data frames.
        """
        resp = self._get('STABLECOINS', f'/stablecoins?includePrices={include_price}')
        lst = resp['peggedAssets']
        
        dict_of_dfs = dict()
        for d0 in lst:
            d1 = d0.pop('chainCirculating')
            _  = d0.pop('chains')
            haha = []
            for k, v in d1.items():
                da = pd.DataFrame(v)
                da['chain'] = k
                da = da.reset_index().rename(columns={'index':'type'}).set_index('chain')
                haha.append(da)
            dict_of_dfs[d0['symbol']] = pd.concat(haha)
        return dict_of_dfs

    def get_stablecoin_hist_mcap(self, id):
        """Get all available historical mcap values for a stablecoin.

        Parameters
        ----------
        id : int 
            Stablecoin ID, you can get these from get_stablecoins_circulating().
            For example, USDT has id 1, USDC 2, Frax 6.

        Returns 
        -------
        data frame
        """
        resp = self._get('STABLECOINS', f'/stablecoincharts/all?stablecoin={id}')
        df = pd.concat([pd.DataFrame(d) for d in resp])        
        df['date'] = df.date.apply(lambda x: dt.datetime.utcfromtimestamp(int(x)))
        df = df.set_index('date')
        return df

    def get_stablecoin_hist_mcap_on_a_chain(self, id, chain):
        """Get all available historical mcap values for a stablecoin on a 
        particular chain.

        Parameters
        ----------
        id : int 
            Stablecoin ID, you can get it from get_stablecoins_circulating().
            For example, USDT has id 1, USDC 2, Frax 6.
        chain : string
            Name of the chain where the stablecoin resides.

        Returns 
        -------
        data frame
        """
        resp = self._get('STABLECOINS', f'/stablecoincharts/{chain}?stablecoin={id}')
        df = pd.concat([pd.DataFrame(d) for d in resp])        
        df['date'] = df.date.apply(lambda x: dt.datetime.utcfromtimestamp(int(x)))
        df = df.set_index('date')
        return df

    def get_stablecoins_curr_mcap_by_chain(self):
        """Get current mcap sum of all stablecoins on each chain.

        Returns 
        -------
        data frame
        """
        resp = self._get('STABLECOINS', f'/stablecoinchains')
        df = pd.concat([pd.DataFrame(d) for d in resp])
        df = df.reset_index().rename(columns={'index':'type'})
        df = df.set_index('name').drop(['gecko_id', 'tokenSymbol'], axis=1)
        df.index.name = 'chain'
        return df

    def get_stablecoins_prices(self):
        """Get historical prices of all stablecoins.

        Returns 
        -------
        data frame
        """
        resp = self._get('STABLECOINS', f'/stablecoinprices')
        df = pd.concat([pd.DataFrame(d) for d in resp])
        df = df.reset_index().rename(columns={'index':'stablecoin'})
        df['date'] = df.date.apply(lambda x: dt.datetime.utcfromtimestamp(int(x)))
        df = df.set_index('date')
        return df

    def get_pools_yields(self):
        """Get the latest data for all pools, including enriched info such as 
        predictions.

        Returns 
        -------
        data frame
        """
        resp = self._get('YIELDS', f'/pools')
        lst = resp['data']
        df = pd.json_normalize(lst)
        df.columns = df.columns.str.replace('predictions.', '', regex=False)
        df['apyPct30D'] = df.apyPct30D.astype(float)
        return df

    def get_pool_hist_apy(self, pool_id):
        """Get historical APY and TVL of a pool.

        Parameters
        ----------
        pool_id : str 
            Pool id, you can get it from the `pool` column after calling 
            get_pools_yields().

        Returns 
        -------
        data frame
        """
        resp = self._get('YIELDS', f'/chart/{pool_id}')
        df = pd.DataFrame(resp['data'])
        df['date'] = df.timestamp.apply(
            lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f%z').date())
        df = df.drop(columns='timestamp')
        df['apyReward'] = df.apyReward.astype(float)
        df['apyBase'] = df.apyBase.astype(float)
        df = df.groupby('date').mean()
        return df
