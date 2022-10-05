# defillama2

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

## Python client for DeFiLlama API

Download data from DefiLlama.com via its [APIs](https://defillama.com/docs/api). 
This package gets you tidy pandas data frames that are ready for downstream 
analysis and modeling.

![](https://github.com/coindataschool/defillama2/blob/main/splash.png)

### Installation

`pip install defillama2`

### Quick Start

```
from defillama2 import DefiLlama

# create a DefiLlama instance
obj = DefiLlama()

# get historical DeFi TVL on all chains
obj.get_defi_hist_tvl()                   # don't give any input

# get historical TVL of a specific chain
obj.get_chain_hist_tvl('Arbitrum')

# get current TVL of all chains
obj.get_chains_curr_tvl()                 # don't give any input

# get current TVL of a specific protocol
obj.get_protocol_curr_tvl('gmx')     

# get current TVL of a specific protocol by chain
obj.get_protocol_curr_tvl_by_chain('gmx') 

# get historical TVL of a specific protocol by chain
obj.get_protocol_hist_tvl_by_chain('gmx') 

# get fundamentals for all protocols
obj.get_protocols_fundamentals()          # don't give any input

# consider the following tokens and chains
dd = {# GMX on arbitrum
      '0xfc5a1a6eb076a2c7ad06ed22c90d7e710e35ad0a':'arbitrum',  
      # GMX on avalanche
      '0x62edc0692BD897D2295872a9FFCac5425011c661':'avax',      
      # GLP on arbitrum
      '0x4277f8f2c384827b5273592ff7cebd9f2c1ac258':'arbitrum',  
      # GLP on avalanche
      '0x01234181085565ed162a948b6a5e88758CD7c7b8':'avax',      
      }

# get their current prices
obj.get_tokens_curr_prices(dd)

# get their prices at a specific time '2022-09-15 13:25:43'
obj.get_tokens_hist_snapshot_prices(dd, '2022-09-15 13:25:43')

# get their historical daily close prices 
obj.get_tokens_hist_prices(dd, start='2022-08-01', end='2022-09-01', type='close')

# get basic info on all stablecoins, along with their circulating amounts
obj.get_stablecoins_circulating()          # don't give any input

# get all stablecoins' circulating amounts for each chain
obj.get_stablecoins_circulating_by_chain() # don't give any input

# get historical mcaps of a stablecoin, for example, USDT
obj.get_stablecoin_hist_mcap(1) # 1 is USDT

# get historical mcaps of a stablecoin on a particular chain, for example, 
# USDT on ethereum
obj.get_stablecoin_hist_mcap_on_a_chain(1, 'ethereum') 

# get current total mcap of all stablecoins on each chain
obj.get_stablecoins_curr_mcap_by_chain()   # don't give any input

# get historical prices of all stablecoins
obj.get_stablecoins_prices()               # don't give any input

# get the latest yields for all available pools, along with other information
obj.get_pools_yields()

# get the historical APY and TVL of a pool
obj.get_pool_hist_apy(pool_id)  # pool_id can be obtained from get_pools_yields()
```

### Demo Code

- [Get TVL and other fundamental data](https://github.com/coindataschool/defillama2/blob/main/notebooks/defillama_api_tvl.ipynb).
- [Get on-chain prices, including exotic tokens](https://github.com/coindataschool/defillama2/blob/main/notebooks/defillama_api_coins.ipynb).
- [Get circulating amount, mcap, prices and other data points for stablecoins](https://github.com/coindataschool/defillama2/blob/main/notebooks/defillama_api_stablecoins.ipynb).
- [Get liquidity pools' yields data](https://github.com/coindataschool/defillama2/blob/main/notebooks/defillama_api_yields.ipynb).
