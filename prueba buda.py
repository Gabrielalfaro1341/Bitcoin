import requests
import json
import time

market_id = 'btc-clp'
url = f'https://www.buda.com/api/v2/markets/{market_id}/ticker'
response = requests.get(url).json()
response=response['ticker']
response['timestamp']=time.time()

print(response)