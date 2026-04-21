import os
import aiohttp
import asyncio
import requests
import pandas as pd
from datetime import datetime

# WxPusher config from Environment Variables
APP_TOKEN = os.environ.get('WXPUSHER_APP_TOKEN')
UIDS = os.environ.get('WXPUSHER_UIDS') # 可以是多个uid，用逗号分隔

def get_top_symbols(limit=100):
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        # 如果返回不是列表（说明是报错字典），打印并返回空
        if not isinstance(data, list):
            print(f"币安 API 返回异常: {data}")
            return []
            
        # 获取USDT交易对，过滤掉杠杆代币 (UP/DOWN/BULL/BEAR)
        exclude_keywords = ['UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT', 'USDCUSDT', 'BUSDUSDT', 'FDUSDUSDT', 'DAIUSDT']
        valid_symbols = []
        
        for d in data:
            symbol = d['symbol']
            if symbol.endswith('USDT') and not any(kw in symbol for kw in exclude_keywords):
                valid_symbols.append((symbol, float(d['quoteVolume'])))
                
        # 按24小时交易额降序排序
        valid_symbols.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in valid_symbols[:limit]]
    except Exception as e:
        print(f"获取排行榜出错: {e}")
        return []

async def fetch_klines(session, symbol):
    # 使用1h级别，为了算KDJ(9,3,3)至少需要最近15-20根线，这里取30
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=30"
    try:
        async with session.get(url) as response:
            data = await response.json()
            return symbol, data
    except Exception as e:
        print(f"获取 {symbol} 数据出错: {e}")
        return symbol, None

def calculate_kdj(data):
    df = pd.DataFrame(data)[[0,1,2,3,4]]
    df.columns = ['t','o','h','l','c']
    for col in ['o','h','l','c']:
        df[col] = df[col].astype(float)
        
    low_min = df['l'].rolling(9).min()
    high_max = df['h'].rolling(9).max()
    
    # 避免除以0
    diff = high_max - low_min
    diff = diff.replace(0, 1e-10)
    
    rsv = (df['c'] - low_min) / diff * 100
    
    k = rsv.ewm(com=2, min_periods=0).mean()
    d = k.ewm(com=2, min_periods=0).mean()
    j = 3 * k - 2 * d
    
    return k.iloc[-1], d.iloc[-1], j.iloc[-1]

async def analyze():
    print(f"[{datetime.now()}] 开始获取交易额前100的币种...")
    symbols = get_top_symbols(100)
    print(f"前10名展示: {symbols[:10]}")
    
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_klines(session, sym) for sym in symbols]
        responses = await asyncio.gather(*tasks)
        
        for symbol, data in responses:
            if not data or len(data) < 9:
                continue
            try:
                k, d, j = calculate_kdj(data)
                results.append((symbol, k, d, j))
            except Exception as e:
                print(f"计算KDJ出错 {symbol}: {e}")

    # 筛选出 J < 0 的信号
    signals = [x for x in results if x[3] < 0]
    
    if not signals:
        print(f"[{datetime.now()}] 扫描完成，未发现 1h J < 0 的币种。")
        return
        
    # 按 J 值从小到大排序
    signals.sort(key=lambda x: x[3])
    
    msg_lines = ["[警报] 量化监控：1小时级别 KDJ (J < 0) 触发"]
    msg_lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    msg_lines.append(f"共发现 {len(signals)} 个高交易额币种出现超卖区域指标：\n")
    
    for symbol, k, d, j in signals:
        msg_lines.append(f"- {symbol}: K={k:.1f}, D={d:.1f}, J={j:.1f}")
        
    content = "\n".join(msg_lines)
    print("====== 触发警报 ======")
    print(content)
    
    send_wxpusher(content)

def send_wxpusher(content):
    if not APP_TOKEN or not UIDS:
        print("未配置 WXPUSHER_APP_TOKEN 或 WXPUSHER_UIDS，跳过发送微信推送。")
        return
        
    uid_list = [uid.strip() for uid in UIDS.split(',') if uid.strip()]
    
    payload = {
        "appToken": APP_TOKEN,
        "content": content,
        "contentType": 1, # 1:文本 2:HTML 3:MD
        "uids": uid_list
    }
    
    url = "https://wxpusher.zjiecode.com/api/send/message"
    try:
        resp = requests.post(url, json=payload)
        res = resp.json()
        if res.get("code") == 1000:
            print("WxPusher 微信消息推送成功！")
        else:
            print(f"WxPusher 推送失败: {res}")
    except Exception as e:
        print(f"WxPusher 请求发送异常: {e}")

if __name__ == "__main__":
    asyncio.run(analyze())
