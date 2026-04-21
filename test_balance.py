import asyncio
from src.data.binance_client import BinanceClient
import yaml

async def test():
    with open('config/config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['binance']
    
    client = BinanceClient(cfg['api_key'], cfg['api_secret'], testnet=False)
    async with client:
        account = await client.get_account()
        print("=== 账户余额 ===")
        for bal in account['balances']:
            if float(bal['free']) > 0 or float(bal['locked']) > 0:
                print(f"{bal['asset']}: 可用 {bal['free']}, 锁定 {bal['locked']}")

asyncio.run(test())