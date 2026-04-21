import aiohttp
import pandas as pd
import asyncio

async def test():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=10"
    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r:
            data = await r.json()
    print(f"Got {len(data)} candles")
    print(data[0][:5])

asyncio.run(test())