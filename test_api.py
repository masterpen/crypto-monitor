import asyncio
from src.data.binance_client import BinanceClient
import yaml

async def test():
    with open('config/config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['binance']
    
    client = BinanceClient(cfg['api_key'], cfg['api_secret'], testnet=False)
    async with client:
        balance = await client.get_ticker('BTCUSDT')
        print('BTC price:', balance['lastPrice'])

asyncio.run(test())