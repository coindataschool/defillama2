import requests
import pandas as pd
# import numpy as np
# from datetime import datetime
from urllib.parse import urlencode, quote

TVL_BASE_URL = "https://api.llama.fi"
COINS_BASE_URL = "https://coins.llama.fi"
STABLECOINS_BASE_URL = "https://stablecoins.llama.fi"
YIELDS_BASE_URL = "https://yields.llama.fi"
ABI_DECODER_BASE_URL = "https://abi-decoder.llama.fi"
VOLUMES_BASE_URL = "https://api.llama.fi"

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
        df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)
        df = df.set_index('date').rename(columns={'totalLiquidityUSD': 'tvl'})
        return df

    def _get(self, api_name, endpoint, params=None):
        """Send 'GET' request.

        Parameters
        ----------
        api_name : string
            Which API to call. Possible values are 'TVL', 'COINS', 'STABLECOINS',
            'YIELDS', 'VOLUMES', and 'ABI_DECODER'. Each type has a different 
            base url.
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
        elif api_name == 'VOLUMES':
            url = VOLUMES_BASE_URL + endpoint
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
        """ Convert json resp (dict) of snapshot prices to data frame. """
        ha = pd.DataFrame([item.split(':') for item in resp['coins'].keys()])
        ha.columns = ['chain', 'token_address']
        df = ha.join(pd.DataFrame([v for k, v in resp['coins'].items()]))
        # convert unix time (seconds) to utc datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        return df

    def _tidy_frame_hist_batch_prices(resp):
        """ Convert json resp (dict) of batch prices to data frame. """
        # extract chains and token addrs and put in a data frame
        dfl = pd.DataFrame([item.split(':') for item in resp['coins'].keys()])
        dfl.columns = ['chain', 'token_address']
        # extract prices for all tokens and all timestamps
        lst = list(resp['coins'].values())
        # add token symbol as a col to dfl to merge with dfr later
        dfl['symbol'] = [elt['symbol'] for elt in lst]
        # flatten the nested list of prices for all tokens and all timestamps 
        # into a data frame
        dfr = pd.json_normalize(lst, record_path='prices', meta='symbol')
        # convert unix time (seconds) to utc datetime
        dfr['timestamp'] = pd.to_datetime(dfr['timestamp'], unit='s', utc=True)
        # merge on the symbol col
        return pd.merge(dfl, dfr)

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
        df = df.loc[:, ['timestamp', 'symbol', 'price', 'chain', 'decimals']]
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
            Human-readable timestamp in utc, for example, '2021-09-25 00:27:53'

        Returns 
        -------
        data frame
        """
        ss = ','.join([v + ':' +k for k, v in token_addrs_n_chains.items()])
        unix_ts = pd.to_datetime(timestamp, utc=True).value / 1e9
        resp = self._get('COINS', f'/prices/historical/{unix_ts}/{ss}')
        df = self._tidy_frame_price(resp)
        df = df.loc[:, ['timestamp', 'symbol', 'price', 'chain', 'decimals']]
        df = df.set_index('timestamp')
        return df 

    def get_tokens_hist_batch_prices(self, chain_token_addr_timestamps):
        """Get historical prices of tokens by chain at multiple timestamps.

        Parameters
        ----------
        chain_token_addr_timestamps : dictionary
            Each key is a chain:token_address; each value is a list of unix 
            timestamps in seconds. For example, 
            {"avax:0xb97ef9ef8734c71904d8002f8b6bc66dd9c48a6e": [1666876743, 1666862343],
             "coingecko:ethereum": [1666869543, 1666862343]}

        Returns 
        -------
        data frame
        """
        val = str(chain_token_addr_timestamps).replace("'", '"')
        param = urlencode(dict(coins=val), quote_via=quote)
        resp = self._get('COINS', 'batchHistorical/', params = param)
        df = self._tidy_frame_hist_batch_price(resp)
        df = df.loc[:, ['timestamp', 'symbol', 'price', 'chain']]
        df = df.set_index('timestamp')
        return df 




    def get_closest_block(self, chain, timestamp):
        """Get the closest block to a timestamp.

        Parameters
        ----------
        chain : string
            Name of the chain.
        timestamp : string
            Human-readable timestamp in utc, for example, '2021-09-25 00:27:53'.

        Returns 
        -------
        data frame
        """
        unix_ts = pd.to_datetime(timestamp, utc=True).value / 1e9
        resp = self._get('COINS', f'/block/{chain}/{unix_ts}')
        df = pd.DataFrame(resp, index=range(1))
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
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
        resp = self._get('STABLECOINS', 
                         f'/stablecoins?includePrices={include_price}')
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
        resp = self._get('STABLECOINS', 
                         f'/stablecoincharts/all?stablecoin={id}')
        df = pd.concat([pd.DataFrame(d) for d in resp])        
        df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)
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
        resp = self._get('STABLECOINS', 
                         f'/stablecoincharts/{chain}?stablecoin={id}')
        df = pd.concat([pd.DataFrame(d) for d in resp])        
        df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)
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
        df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)
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
        df['date'] = pd.to_datetime(
            df['timestamp'],
            format='%Y-%m-%dT%H:%M:%S.%f%z').dt.normalize()
        df = df.drop(columns='timestamp')
        df['apyReward'] = df.apyReward.astype(float)
        df['apyBase'] = df.apyBase.astype(float)
        df = df.groupby('date').mean()
        return df

    def get_dexes_volumes(self, data_type='dailyVolume'):
        """Get transaction volumes of all dexes, 
        including 'Dexes', 'Derivatives', and 'Yield' protocols.

        Parameters
        ----------
        data_type : string
            'dailyVolume' or 'totalVolume'. It seems 'totalVolume' isn't 
            used on DeFiLlama's website. So use 'dailyVolume' for most 
            cases.
        
        Returns 
        -------
        dictionary of data frames: 
            - volume_overall
            - volume_by_dex 
            - volume_by_dex_by_chain
            - daily_volume
            - daily_volume_by_dex
        """
        resp = self._get(
            'VOLUMES', f'/overview/dexs?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=false&dataType={data_type}')

        # overall volume across all dexes and chains
        volume_overall = pd.DataFrame(
            dict(total24h = resp['total24h'], 
                total7d = resp['total7d'],
                change_1d = resp['change_1d'],
                change_7d = resp['change_7d'],
                change_1m = resp['change_1m'],
                change_7dover7d = resp['change_7dover7d']),
            index=range(1))

        # volume by dex
        df = pd.DataFrame(resp['protocols'])\
            .query("latestFetchIsOk == True & disabled == False")
        ha = df['breakdown24h']
        volume_by_dex = df.drop(columns=[
            'latestFetchIsOk', 'disabled', 'module', 'logo', 'protocolType', 'displayName', 'methodology', 'methodologyURL', 'breakdown24h', 'protocolsStats'])

        # volume by dex by chain
        volume_by_dex_by_chain = \
            pd.concat([pd.DataFrame(ha.iloc[i]) for i in range(len(ha))])\
            .stack().reset_index()
        volume_by_dex_by_chain.columns = ['protocol', 'chain', 'total24h']

        # daily volume of all dexes
        daily_volume = pd.DataFrame(resp['totalDataChart'])
        daily_volume.columns = ['date', 'volume']
        daily_volume['date'] = pd.to_datetime(
            daily_volume['date'], unit='s', utc=True)
        
        # daily volume by dex
        daily_volume_by_dex = pd.DataFrame(resp['totalDataChartBreakdown'])
        daily_volume_by_dex.columns = ['date', 'dex_vol_dict']
        daily_volume_by_dex['date'] = \
            pd.to_datetime(daily_volume_by_dex['date'], unit='s', utc=True)
        daily_volume_by_dex = daily_volume_by_dex[['date']]\
            .join(pd.DataFrame(daily_volume_by_dex['dex_vol_dict'].tolist()))

        return {'volume_overall': volume_overall, 
                'volume_by_dex': volume_by_dex, 
                'volume_by_dex_by_chain': volume_by_dex_by_chain, 
                'daily_volume':daily_volume, 
                'daily_volume_by_dex': daily_volume_by_dex}