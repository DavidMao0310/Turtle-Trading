from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib as ta
import yfinance as yf
from pyecharts import options as opts
from pyecharts.charts import *

pd.set_option('display.max_columns', None)
plt.style.use('fivethirtyeight')


def get_daily_data(name, start, end):
    '''

    :param name: stock code, e.g: ^GSPC
    :param start: yyyy-mm-dd
    :param end: yyyy-mm-dd
    :return: daily data
    '''
    x = yf.download(name, start=start, end=end, interval="1d")
    x['ret'] = x['Close'].pct_change(1)
    x.dropna(inplace=True)
    return x


name1 = input('Code:')
start1 = input('start_date(yyyy-mm-dd):')
end1 = input('end_date(yyyy-mm-dd):')

data = get_daily_data(name1, start1, end1)

data['up'] = ta.MAX(data['High'], timeperiod=20).shift(1)
data['down'] = ta.MIN(data['Low'], timeperiod=10).shift(1)
data['ATR'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=20)
data.dropna(inplace=True)


def look_strategy(data):
    x1 = data.Close > data.up
    x2 = data.Close.shift(1) < data.up.shift(1)
    x = x1 & x2
    y1 = data.Close < data.down
    y2 = data.Close.shift(1) > data.down.shift(1)
    y = y1 & y2
    data.loc[x, 'signal'] = 'buy'
    data.loc[y, 'signal'] = 'sell'
    buy_date = (data[data.signal == 'buy'].index).strftime('%Y-%m-%d')
    sell_date = (data[data.signal == 'sell'].index).strftime('%Y-%m-%d')
    buy_close = data[data.signal == 'buy'].Close.round(2).tolist()
    sell_close = data[data.signal == 'sell'].Close.round(2).tolist()
    return (buy_date, buy_close, sell_date, sell_close)


attr = [str(t) for t in data.index.strftime('%Y-%m-%d')]
v1 = np.array(data.loc[:, ['Open', 'Close', 'Low', 'High']]).tolist()
v2 = np.array(data.up)
v3 = np.array(data.down)

kline = (
    Kline()
        .add_xaxis(attr)
        .add_yaxis('Kline', v1)
        .set_global_opts(
        xaxis_opts=opts.AxisOpts(is_scale=True, is_show=True),
        yaxis_opts=opts.AxisOpts(is_scale=True),
        datazoom_opts=[opts.DataZoomOpts(is_show=True)],
        title_opts=opts.TitleOpts(title='Donchian Channel'))

)

bar = (
    Bar()
        .add_xaxis(attr)
        .add_yaxis(y_axis=np.array(data['Volume']).tolist(), series_name='Volume', color='green')
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
)

line = (
    Line()
        .add_xaxis(attr)
        .add_yaxis(y_axis=v2.round(1), series_name='Upper bound', is_smooth=True, is_symbol_show=False)
        .add_yaxis(y_axis=v3.round(1), series_name='Lower bound', is_smooth=True, is_symbol_show=False)
        .add_yaxis(
        series_name="SMA50",
        y_axis=ta.SMA(data['Close'].values, timeperiod=50),
        is_smooth=True,
        is_hover_animation=False,
        linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5),
        label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis(
        series_name="EMA15",
        y_axis=ta.EMA(data['Close'].values, timeperiod=15),
        is_smooth=True,
        is_hover_animation=False,
        linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5),
        label_opts=opts.LabelOpts(is_show=False)
    )

)

bd, bc, sd, sc = look_strategy(data)
es1 = (
    EffectScatter()
        .add_xaxis(sd)
        .add_yaxis(series_name='Sell', y_axis=sc, color='Blue')
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))

)

es2 = (
    EffectScatter()
        .add_xaxis(bd)
        .add_yaxis(series_name='Buy', y_axis=bc, symbol='triangle', color='gold')
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))

)

kline.overlap(line).overlap(es1).overlap(es2)
kline.render('DC.html')
grid_chart = Grid()
grid_chart.add(
    kline,
    grid_opts=opts.GridOpts(pos_left="15%", pos_right="8%", height="55%"),
)
grid_chart.add(
    bar,
    grid_opts=opts.GridOpts(pos_left="15%", pos_right="8%", pos_top="70%", height="18%"),
)

grid_chart.render('DC(withvolume).html')


def test_strategy(stock, start, end, N1=20, N2=10):
    df = get_daily_data(stock, start, end)
    df['H_N1'] = ta.MAX(df.High, timeperiod=N1)
    df['L_N2'] = ta.MIN(df.Low, timeperiod=N2)
    buy_index = df[df.Close > df['H_N1'].shift(1)].index
    df.loc[buy_index, 'close signal'] = 1
    sell_index = df[df.Close < df['L_N2'].shift(1)].index
    df.loc[sell_index, 'close signal'] = 0
    df['daily position size'] = df['close signal'].shift(1)
    df['daily position size'].fillna(method='ffill', inplace=True)
    d = df[df['daily position size'] == 1].index[0] - timedelta(days=1)
    df1 = df.loc[d:].copy()
    df1['ret'][0] = 0
    df1['daily position size'][0] = 0
    df1['Net Value on strategy'] = (df1.ret.values * df1['daily position size'].values + 1.0).cumprod()
    df1['Net Value on index'] = (df1.ret.values + 1.0).cumprod()
    df1['Return on strategy'] = df1['Net Value on strategy'] / df1['Net Value on strategy'].shift(1) - 1
    df1['Return on index'] = df1.ret
    total_ret = df1[['Net Value on strategy', 'Net Value on index']].iloc[-1] - 1
    annual_ret = pow(1 + total_ret, 250 / len(df1)) - 1
    dd = (df1[['Net Value on strategy', 'Net Value on index']].cummax() - df1[
        ['Net Value on strategy', 'Net Value on index']]) / df1[
             ['Net Value on strategy', 'Net Value on index']].cummax()
    d = dd.max()
    beta = df1[['Return on strategy', 'Return on index']].cov().iat[0, 1] / df1['Return on index'].var()
    alpha = (annual_ret['Net Value on strategy'] - annual_ret['Net Value on index'] * beta)
    exReturn = df1['Return on strategy'] - 0.03 / 250
    sharper_atio = np.sqrt(len(exReturn)) * exReturn.mean() / exReturn.std()
    TA1 = round(total_ret['Net Value on strategy'] * 100, 2)
    TA2 = round(total_ret['Net Value on index'] * 100, 2)
    AR1 = round(annual_ret['Net Value on strategy'] * 100, 2)
    AR2 = round(annual_ret['Net Value on index'] * 100, 2)
    MD1 = round(d['Net Value on strategy'] * 100, 2)
    MD2 = round(d['Net Value on index'] * 100, 2)
    S = round(sharper_atio, 2)
    df1[['Net Value on strategy', 'Net Value on index']].plot(figsize=(15, 7))
    plt.title('Return on Turtle Trading', size=15)
    bbox = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    plt.text(df1.index[int(len(df1) / 5)], df1['Net Value on index'].max() / 1.5, f'Total return:\
Strategy{TA1}%, index{TA2}% \nAnnual_Return:Strategy{AR1}%, index{AR2}% \nmax_drawdown:Strategy{MD1}%, index{MD2}%;\n\
Strategy_alpha:{round(alpha, 2)}, Strategy_beta:{round(beta, 2)} \nSharpe_ratio:{S}', size=13, bbox=bbox)
    plt.xlabel('')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.show()
    #return df1.loc[:,['close','ret','H_N1','L_N2','daily position size','Net Value on strategy','Net Value on index']]


test_strategy(name1, start1, end1)
