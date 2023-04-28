# Task: 
# Get top 30 protocols by TVL, excluding CEX and Chain, and get their historical
# TVLs by chain.

from defillama2 import DefiLlama
import numpy as np

# initialize api client
llama = DefiLlama()
# get all protocols
df = llama.get_protocols() 

# check protocol categories
df['category'].unique()

# exclude CEX and Chain categories
exclude_cats = ['CEX', 'Chain']
df = df[~df['category'].isin(exclude_cats)]

# df is granular. For example, AAVE is broken down by AAVE V2 
# and V3; Uniswap is broken down by Uniswap V3 and V3. So we 
# want to aggregate TVL based on parent protocol first. If you pass
# the granular name, say AAVE V2, to get_protocol() or 
# get_protocol_hist_tvl_by_chain(), you'd get an error because the api
# only recognize the parent protocol, which is just AAVE in this case.
xs = df['parentProtocol'].str.replace('parent#', '')
parent_protocols = np.where(xs.isna(), df['name'], xs)
protocols_by_tvl = df.groupby(parent_protocols)['tvl'].sum()\
    .sort_values(ascending=False)

# get historical tvls by chain for the top 30 protocols
top30 = protocols_by_tvl.head(30).index.to_list()
# replace space with dash to be recognized by the api
top30 = [string.replace(' ', '-') for string in top30]
     
for name in top30[8:]:
    historical_tvl_by_chain = llama.get_protocol_hist_tvl_by_chain(protocol=name)
    for chain, frame in historical_tvl_by_chain.items():
        frame.to_csv(name+'-'+chain+'.csv')

