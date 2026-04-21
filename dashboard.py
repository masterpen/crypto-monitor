"""
量化交易系统可视化 Dashboard
基于 Streamlit 构建
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 页面配置
st.set_page_config(
    page_title="量化交易系统",
    page_icon="📈",
    layout="wide"
)


def load_sample_data():
    """加载示例数据用于展示"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # 生成模拟的K线数据
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 200)
    
    df = pd.DataFrame({
        'date': dates,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, len(dates)),
    })
    df['open'] = df['close'] * (1 + np.random.randn(len(dates)) * 0.01)
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.rand(len(dates)) * 0.02)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.rand(len(dates)) * 0.02)
    
    return df


def plot_equity_curve(equity_data):
    """绘制权益曲线"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_data.index,
        y=equity_data.values,
        mode='lines',
        name='权益',
        line=dict(color='#2E86AB', width=2)
    ))
    
    # 添加回撤阴影
    cummax = equity_data.cummax()
    drawdown = equity_data - cummax
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='回撤',
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)',
        line=dict(color='rgba(231, 76, 60, 0.5)', width=1)
    ))
    
    fig.update_layout(
        title='权益曲线与回撤',
        xaxis_title='日期',
        yaxis_title='资金 (USDT)',
        template='plotly_dark',
        height=400
    )
    return fig


def plot_price_with_signals(data, signals):
    """绘制价格图表和交易信号"""
    fig = go.Figure()
    
    # K线
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='K线'
    ))
    
    # 买入信号
    buy_signals = signals[signals['signal'] == 'long']
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['price'],
            mode='markers',
            name='买入',
            marker=dict(symbol='triangle-up', size=15, color='green')
        ))
    
    # 卖出信号
    sell_signals = signals[signals['signal'] == 'short']
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['price'],
            mode='markers',
            name='卖出',
            marker=dict(symbol='triangle-down', size=15, color='red')
        ))
    
    fig.update_layout(
        title='交易信号',
        xaxis_title='日期',
        yaxis_title='价格',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False
    )
    return fig


def plot_performance_metrics(metrics):
    """绘制性能指标"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "总收益率",
            f"{metrics['total_return']:.2%}",
            delta=f"{metrics['total_pnl']:.2f} USDT"
        )
    
    with col2:
        st.metric(
            "夏普比率",
            f"{metrics['sharpe_ratio']:.2f}",
        )
    
    with col3:
        st.metric(
            "最大回撤",
            f"{metrics['max_drawdown']:.2%}",
        )
    
    with col4:
        st.metric(
            "胜率",
            f"{metrics['win_rate']:.1%}",
            delta=f"{metrics['winning_trades']} 笔盈利"
        )
    
    # 详细指标
    st.markdown("---")
    st.subheader("📊 详细统计")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**交易统计**")
        stats_df = pd.DataFrame({
            '指标': ['总交易次数', '盈利交易', '亏损交易', '平均交易盈亏', '盈亏比'],
            '数值': [
                metrics['total_trades'],
                metrics['winning_trades'],
                metrics['losing_trades'],
                f"{metrics['avg_trade_pnl']:.2f}",
                f"{metrics['profit_factor']:.2f}"
            ]
        })
        st.table(stats_df)
    
    with col2:
        st.write("**风险指标**")
        risk_df = pd.DataFrame({
            '指标': ['索提诺比率', '卡尔玛比率', '平均盈利', '平均亏损'],
            '数值': [
                f"{metrics['sortino_ratio']:.2f}",
                f"{metrics['calmar_ratio']:.2f}",
                f"{metrics['avg_winning']:.2f}",
                f"{metrics['avg_losing']:.2f}"
            ]
        })
        st.table(risk_df)


def plot_drawdown_chart(equity_data):
    """绘制回撤图表"""
    cummax = equity_data.cummax()
    drawdown = (equity_data - cummax) / cummax * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.3)',
        line=dict(color='#E74C3C', width=1.5),
        name='回撤'
    ))
    
    fig.update_layout(
        title='回撤分析',
        xaxis_title='日期',
        yaxis_title='回撤 (%)',
        template='plotly_dark',
        height=300
    )
    return fig


def plot_distribution(returns):
    """绘制收益分布"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='收益分布',
        marker_color='#3498DB'
    ))
    
    fig.update_layout(
        title='收益分布',
        xaxis_title='收益率',
        yaxis_title='频次',
        template='plotly_dark',
        height=300
    )
    return fig


def main():
    st.title("📈 量化交易系统 Dashboard")
    
    # 侧边栏 - 设置
    st.sidebar.header("⚙️ 设置")
    
    with st.sidebar:
        st.subheader("策略选择")
        selected_strategies = st.multiselect(
            "选择策略",
            ["双均线策略", "MACD策略", "布林带策略", "RSI策略", "动量策略"],
            default=["双均线策略"]
        )
        
        st.subheader("参数配置")
        fast_period = st.slider("快速周期", 5, 50, 10)
        slow_period = st.slider("慢速周期", 20, 100, 30)
        
        st.subheader("风控设置")
        max_position = st.slider("最大仓位", 0.1, 1.0, 0.3)
        stop_loss = st.slider("止损比例", 0.01, 0.1, 0.02)
        
        st.subheader("回测设置")
        start_date = st.date_input("开始日期", datetime(2024, 1, 1))
        end_date = st.date_input("结束日期", datetime(2024, 12, 31))
        initial_capital = st.number_input("初始资金", 1000, 1000000, 10000)
        
        if st.button("🚀 运行回测", type="primary"):
            st.session_state['run_backtest'] = True
    
    # 主内容区域
    tab1, tab2, tab3, tab4 = st.tabs(["📊 概览", "📈 图表", "📋 交易记录", "⚙️ 配置"])
    
    with tab1:
        # 模拟的性能数据
        metrics = {
            'total_return': 0.356,
            'total_pnl': 3560,
            'sharpe_ratio': 1.85,
            'sortino_ratio': 2.12,
            'calmar_ratio': 1.45,
            'max_drawdown': -0.082,
            'total_trades': 45,
            'winning_trades': 28,
            'losing_trades': 17,
            'win_rate': 0.622,
            'avg_trade_pnl': 79.11,
            'avg_winning': 180.5,
            'avg_losing': -95.2,
            'profit_factor': 1.89
        }
        
        plot_performance_metrics(metrics)
    
    with tab2:
        # 加载数据
        df = load_sample_data()
        df.set_index('date', inplace=True)
        
        # 生成模拟权益曲线
        equity = initial_capital * (1 + np.cumsum(np.random.randn(len(df)) * 0.02 + 0.0003))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(plot_equity_curve(pd.Series(equity, index=df.index)))
        with col2:
            st.plotly_chart(plot_drawdown_chart(pd.Series(equity, index=df.index)))
        
        st.plotly_chart(plot_distribution(np.random.randn(1000) * 0.02))
    
    with tab3:
        # 交易记录表格
        trades_df = pd.DataFrame({
            '时间': pd.date_range(start='2024-01-15', periods=20, freq='3D'),
            '交易对': ['BTCUSDT'] * 20,
            '方向': np.random.choice(['买入', '卖出'], 20),
            '价格': 50000 + np.random.randn(20) * 1000,
            '数量': np.random.uniform(0.01, 0.1, 20),
            '盈亏': np.random.randn(20) * 100,
            '状态': ['已完成'] * 20
        })
        trades_df['金额'] = trades_df['价格'] * trades_df['数量']
        trades_df['盈亏'] = trades_df['盈亏'].round(2)
        
        st.dataframe(trades_df, use_container_width=True)
    
    with tab4:
        st.subheader("当前配置")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.write("**数据源**")
            st.write("- 交易所: Binance Testnet")
            st.write("- 交易对: BTCUSDT, ETHUSDT")
            st.write("- 数据周期: 1小时")
        
        with config_col2:
            st.write("**风险控制**")
            st.write(f"- 最大仓位: {max_position:.0%}")
            st.write(f"- 止损比例: {stop_loss:.0%}")
            st.write("- 日度最大亏损: 5%")


if __name__ == "__main__":
    main()