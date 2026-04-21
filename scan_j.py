import aiohttp
import pandas as pd
import asyncio

async def scan():
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'APTUSDT', 'XRPUSDT', 'ADAUSDT', 
              'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 
              'ATOMUSDT', 'UNIUSDT', 'ETCUSDT', 'NEARUSDT', 'FILUSDT', 'ARBUSDT', 'OPUSDT']
    results = []
    
    for symbol in symbols:
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=30"
            async with aiohttp.ClientSession() as s:
                async with s.get(url) as r:
                    data = await r.json()
            
            df = pd.DataFrame(data)[[0,1,2,3,4]]
            df.columns = ['t','o','h','l','c']
            for col in ['o','h','l','c']:
                df[col] = df[col].astype(float)
            
            low_min = df['l'].rolling(9).min()
            high_max = df['h'].rolling(9).max()
            rsv = (df['c'] - low_min) / (high_max - low_min) * 100
            k = rsv.ewm(2).mean()
            d = k.ewm(2).mean()
            j = 3*k - 2*d
            
            results.append((symbol, float(k.iloc[-1]), float(d.iloc[-1]), float(j.iloc[-1])))
        except:
            pass
    
    results.sort(key=lambda x: x[3])
    print("=== J值排序 ===")
    print(f"{'币种':10} {'K':>5} {'D':>5} {'J':>6} {'状态'}")
    print("-" * 35)
    for s,k,d,j in results:
        flag = "←买!" if j < 0 else ("超买>" if j > 100 else "")
        print(f"{s:10} {k:5.1f} {d:5.1f} {j:6.1f} {flag}")

asyncio.run(scan())