import asyncio
from src.data.binance_client import BinanceClient
from src.strategies import KDJStrategy
import yaml

async def find_signals():
    with open('config/config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['binance']
    
    client = BinanceClient(cfg['api_key'], cfg['api_secret'], testnet=False)
    strategy = KDJStrategy(period=9)
    
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'DOGEUSDT',
        'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
        'LINKUSDT', 'LTCUSDT', 'ATOMUSDT', 'UNIUSDT', 'ETCUSDT',
        'XLMUSDT', 'NEARUSDT', 'APTUSDT', 'FILUSDT', 'ARBUSDT',
        'OPUSDT', 'INJUSDT', 'SANDUSDT', 'MANAUSDT', 'AAVEUSDT'
    ]
    
    results = []
    
    async with client:
        for symbol in symbols:
            try:
                data = await client.get_klines(symbol=symbol, interval='1h', limit=50)
                if data is None or len(data) < 20:
                    continue
                    
                k, d, j = strategy._calculate_kdj(data)
                
                current_k = k.iloc[-1]
                current_d = d.iloc[-1]
                current_j = j.iloc[-1]
                
                results.append({
                    'symbol': symbol,
                    'k': current_k,
                    'd': current_d,
                    'j': current_j
                })
            except Exception as e:
                print(f"获取 {symbol} 失败: {e}")
    
    # 按K值排序
    results.sort(key=lambda x: x['k'])
    
    print("=== KDJ分布 (按K值升序) ===\n")
    for r in results:
        flag = ""
        if r['k'] <= 10:
            flag = " ←超卖!"
        elif r['k'] < 30:
            flag = " ←接近超卖"
        elif r['k'] > 90:
            flag = " ←严重超买"
        elif r['k'] > 70:
            flag = " 超买→"
        print(f"{r['symbol']:10} K={r['k']:5.1f} D={r['d']:5.1f} J={r['j']:6.1f} {flag}")

asyncio.run(find_signals())