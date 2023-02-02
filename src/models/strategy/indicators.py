import warnings
warnings.filterwarnings("ignore")
import talib as ta
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from yahooquery import Ticker
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import deque
from math import floor
from termcolor import colored as cl 

import src.models.strategy as s1

# plt.style.use("seaborn-talk")
sm, med, lg = 10, 15, 20
plt.rc("font", size=sm)  # controls default text sizes
plt.rc("axes", titlesize=med)  # fontsize of the axes title
plt.rc("axes", labelsize=med)  # fontsize of the x & y labels
plt.rc("xtick", labelsize=sm)  # fontsize of the tick labels
plt.rc("ytick", labelsize=sm)  # fontsize of the tick labels
plt.rc("legend", fontsize=sm)  # legend fontsize
plt.rc("figure", titlesize=lg)  # fontsize of the figure title
plt.rc("axes", linewidth=2)  # linewidth of plot lines
plt.rcParams["figure.figsize"] = [15, 8]
plt.rcParams["figure.dpi"] = 100




def get_company_longName(symbol):
    d = Ticker(symbol).quote_type
    return list(d.values())[0]["longName"]





class Indicator_Ike(object):
    
    
    def __init__(self, ticker, date1, cc=0.0, ccc=0.0, graphit=True):
        self.stock = ticker
        self.date1 = date1
        self.cc = cc
        self.ccc = ccc
        self.graphit = graphit


    def bollinger_bands(self, df):
        df["upper_band"], df["middle_band"], df["lower_band"] = ta.BBANDS(df["adjclose"], timeperiod=20)
        df["Signal"] = 0.0
        df["Signal"] = np.where(df["adjclose"] > df["middle_band"], 1.0, 0.0)
        df["Position"] = df["Signal"].diff()
        df_pos = df[(df["Position"] == 1) | (df["Position"] == -1)]
        df_pos["Position"] = df_pos["Position"].apply(lambda x: "Buy" if x == 1 else "Sell")

        if self.graphit is True:
            fig, ax = plt.subplots()
            plt.tick_params(axis="both", labelsize=15)
            df["adjclose"].plot(color="k", lw=2, label="adjclose")
            df["upper_band"].plot(color="g", lw=1, label="upper_band", linestyle="dashed")
            df["middle_band"].plot(color="r", lw=1, label="middle_band")
            df["lower_band"].plot(color="b", lw=1, label="lower_band", linestyle="dashed")
            plt.plot(df[df["Position"] == 1].index, df["adjclose"][df["Position"] == 1], "^", markersize=15, color="g", alpha=0.7, label="buy")       # plot 'buy' signals
            plt.plot(df[df["Position"] == -1].index, df["adjclose"][df["Position"] == -1], "v", markersize=15, color="r", alpha=0.7, label="sell")    # plot 'sell' signals        
            plt.ylabel("Price", fontsize=20, fontweight="bold")
            plt.xlabel("Date", fontsize=20, fontweight="bold")
            plt.title(f"{self.stock} - bollinger bands", fontsize=30, fontweight="bold")
            plt.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
            ax.legend(loc="best", prop={"size": 16})
            plt.tight_layout()
            plt.show()
            st.pyplot(fig)

        if df_pos["Position"][-1] == "Buy":
            st.metric(
                f"No. {self.cc} / {self.ccc} In Portfolio",
                f"{self.stock}",
                f"{df_pos['Position'][-1]}",
                )
            return self.stock
        
        elif df_pos["Position"][-1] == "Sell":
            st.metric(
                f"No. {self.cc} / {self.ccc} In Portfolio",
                f"{self.stock}",
                f"- {df_pos['Position'][-1]}",
                )



    def macd(self, data):
        data["macd"], data["macdsignal"], data["macdhist"] = ta.MACD(data["adjclose"], fastperiod=12, slowperiod=26, signalperiod=9)
        stock_df = pd.DataFrame(data)
        stock_df["Signal"] = 0.0
        stock_df["Signal"] = np.where(stock_df["macd"] > stock_df["macdsignal"], 1.0, 0.0)
        stock_df["Position"] = stock_df["Signal"].diff()
        df_pos = stock_df[(stock_df["Position"] == 1) | (stock_df["Position"] == -1)]
        df_pos["Position"] = df_pos["Position"].apply(lambda x: "Buy" if x == 1 else "Sell")
        stock_df.dropna(inplace=True)

        if self.graphit is True:  # plot adjclose price, short-term and long-term moving averages
            fig, ax = plt.subplots()
            plt.tick_params(axis="both", labelsize=15)
            stock_df["macdhist"].plot(color="r", lw=1.5, label="macdhist")
            stock_df["macd"].plot(color="b", lw=2, label="macd")
            stock_df["macdsignal"].plot(color="g", lw=2, label="macdsignal")
            plt.plot(stock_df[stock_df["Position"] == 1].index, stock_df["macd"][stock_df["Position"] == 1], "^", markersize=15, color="g", alpha=0.7, label="buy")     # plot 'buy' signals
            plt.plot(stock_df[stock_df["Position"] == -1].index, stock_df["macd"][stock_df["Position"] == -1], "v", markersize=15, color="r", alpha=0.7, label="sell")  # plot 'sell' signals
            plt.ylabel("MACD", fontsize=20, fontweight="bold")
            plt.xlabel("Date", fontsize=20, fontweight="bold")
            plt.title(f"{self.stock} - MACD", fontsize=30, fontweight="bold")
            plt.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
            ax.legend(loc="best", prop={"size": 16})
            plt.tight_layout()
            plt.show()
            st.pyplot(fig)

        if df_pos["Position"][-1] == "Buy":
            st.metric(
                f"No. {self.cc} / {self.ccc} In Portfolio",
                f"{self.stock}",
                f"{df_pos['Position'][-1]}",
            )
            return self.stock
        
        elif df_pos["Position"][-1] == "Sell":
            st.metric(
                f"No. {self.cc} / {self.ccc} In Portfolio",
                f"{self.stock}",
                f"- {df_pos['Position'][-1]}",
            )

        act_lst = []
        for i in stock_df["Position"]:
            if i == 1.0:
                act_lst.append("Buy")
            elif i == -1.0:
                act_lst.append("Sell")
            else:
                act_lst.append("")
        stock_df["action"] = act_lst
        del stock_df["open"]
        del stock_df["high"]
        del stock_df["low"]
        del stock_df["adjclose"]
        del stock_df["volume"]
        stock_df = stock_df[stock_df["action"] != ""]
        
        
        
    



    def get_historical_data(self, symbol, start_date=None):
        df = yf.download(symbol, period='1y').reset_index()
        df.columns = [x.lower() for x in df.columns]
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        for i in df.columns:
            df[i] = df[i].astype(float)
        if start_date:
            df = df[df.index <= start_date]
        return df


    def get_rsi(self, close, lookback):
        ret = close.diff()
        up = []
        down = []
        for i in range(len(ret)):
            if ret[i] < 0:
                up.append(0)
                down.append(ret[i])
            else:
                up.append(ret[i])
                down.append(0)
        up_series = pd.Series(up)
        down_series = pd.Series(down).abs()
        up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
        down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
        rs = up_ewm/down_ewm
        rsi = 100 - (100 / (1 + rs))
        rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
        rsi_df = rsi_df.dropna()
        return rsi_df#[3:]


    def implement_rsi_strategy(self, prices, rsi):    
        self.buy_price = []
        self.sell_price = []
        self.rsi_signal = []
        signal = 0
        for i in range(len(rsi)):
            if rsi[i-1] > 30 and rsi[i] < 30:
                if signal != 1:
                    self.buy_price.append(prices[i])
                    self.sell_price.append(np.nan)
                    signal = 1
                    self.rsi_signal.append(signal)
                else:
                    self.buy_price.append(np.nan)
                    self.sell_price.append(np.nan)
                    self.rsi_signal.append(0)
            elif rsi[i-1] < 70 and rsi[i] > 70:
                if signal != -1:
                    self.buy_price.append(np.nan)
                    self.sell_price.append(prices[i])
                    signal = -1
                    self.rsi_signal.append(signal)
                else:
                    self.buy_price.append(np.nan)
                    self.sell_price.append(np.nan)
                    self.rsi_signal.append(0)
            else:
                self.buy_price.append(np.nan)
                self.sell_price.append(np.nan)
                self.rsi_signal.append(0)
        return self.buy_price, self.sell_price, self.rsi_signal


    def graph_rsi(self):
        fig, ax = plt.subplots()
        ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
        ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)
        ax1.plot(self.rsi_df['close'], linewidth = 2.5, color = 'skyblue', label=f"{self.stock}")
        ax1.plot(self.rsi_df.index, self.buy_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')
        ax1.plot(self.rsi_df.index, self.sell_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')
        ax1.set_title(f'{self.stock} RSI TRADE SIGNALS')
        ax2.plot(self.rsi_df['rsi_14'], color = 'orange', linewidth = 2.5)
        ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
        ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
        st.pyplot(fig)
        
        
    def rsi(self):        
        self.rsi_df = self.get_historical_data(self.stock, self.date1)
        self.rsi_df['rsi_14'] = self.get_rsi(self.rsi_df['close'], 14)
        self.rsi_df = self.rsi_df.dropna()
        self.buy_price, self.sell_price, self.rsi_signal = self.implement_rsi_strategy(self.rsi_df['close'], self.rsi_df['rsi_14'])
        if self.graphit == True:
            self.graph_rsi()
                        

        position = []
        for i in range(len(self.rsi_signal)):
            if self.rsi_signal[i] > 1:
                position.append(0)
            else:
                position.append(1)
                
        for i in range(len(self.rsi_df['close'])):
            if self.rsi_signal[i] == 1:
                position[i] = 1
            elif self.rsi_signal[i] == -1:
                position[i] = 0
            else:
                position[i] = position[i-1]
                
                
        rsi = self.rsi_df['rsi_14']
        close_price = self.rsi_df['close']
        self.rsi_signal = pd.DataFrame(self.rsi_signal).rename(columns = {0:'rsi_signal'}).set_index(self.rsi_df.index)
        position = pd.DataFrame(position).rename(columns = {0:'rsi_position'}).set_index(self.rsi_df.index)
        frames = [close_price, rsi, self.rsi_signal, position]
        strategy = pd.concat(frames, join = 'inner', axis = 1)
        ibm_ret = pd.DataFrame(np.diff(self.rsi_df['close'])).rename(columns = {0:'returns'})
        rsi_strategy_ret = []

        for i in range(len(ibm_ret)):
            returns = ibm_ret['returns'][i]*strategy['rsi_position'][i]
            rsi_strategy_ret.append(returns)
            
        rsi_strategy_ret_df = pd.DataFrame(rsi_strategy_ret).rename(columns = {0:'rsi_returns'})
        investment_value = 100000
        number_of_stocks = floor(investment_value/self.rsi_df['close'][-1])
        rsi_investment_ret = []

        for i in range(len(rsi_strategy_ret_df['rsi_returns'])):
            returns = number_of_stocks*rsi_strategy_ret_df['rsi_returns'][i]
            rsi_investment_ret.append(returns)


        signal_df = pd.DataFrame(strategy[strategy['rsi_signal'] != 0.0])
        if signal_df.empty == False:
            res = signal_df.iloc[-1]['rsi_signal']
            
            if res == 1:      
                st.metric(
                    f"No. {self.cc} / {self.ccc} ~ [rsi = {round(strategy['rsi_14'][-1], 2)}]",
                    f"{self.stock}",
                    f"BUY",
                    )
                return self.stock
            
            else:
                st.metric(
                    f"No. {self.cc} / {self.ccc} ~ [rsi = {round(strategy['rsi_14'][-1], 2)}]",
                    f"{self.stock}",
                    f"- SELL",
                    )                    
                return
        else:
            st.metric(
                f"No. {self.cc} / {self.ccc} ~ [rsi = {round(strategy['rsi_14'][-1], 2)}]",
                f"{self.stock}",
                f"Hold",
                )            
            return self.stock
            
            
            
    def ma_crossover(self, data, moving_avg, short_window, long_window):
        data = pd.DataFrame(data)
        data.columns = [x.lower() for x in data.columns]
        data.index = pd.to_datetime(data.index)
        stock_df = pd.DataFrame(data["adjclose"])
        stock_df = stock_df.fillna(0.0)

        # column names for long and short moving average columns
        short_window_col = moving_avg + "_" + str(short_window) 
        long_window_col = moving_avg + "_" + str(long_window) 

        # Create a short simple WMA (Weighted Moving Average) - [short_ema] & create a long WMA (Weighted Moving Average) - [long_ema]
        if moving_avg == "SMA":
            stock_df[short_window_col] = ta.SMA(stock_df['adjclose'], short_window)
            stock_df[long_window_col] = ta.SMA(stock_df['adjclose'], long_window)
            
        # Create a short simple WMA (Weighted Moving Average) - [short_ema] & create a long WMA (Weighted Moving Average) - [long_ema]
        if moving_avg == "WMA":
            stock_df[short_window_col] = ta.WMA(stock_df['adjclose'], short_window)
            stock_df[long_window_col] = ta.WMA(stock_df['adjclose'], long_window)

        # Create a short EMA (Exponential Moving Average) - [short_ema] & create a long EMA (Exponential Moving Average) - [long_ema]
        elif moving_avg == "EMA":
            stock_df[short_window_col] = ta.EMA(stock_df['adjclose'], short_window)
            stock_df[long_window_col] = ta.EMA(stock_df['adjclose'], long_window)
            
        # Create a short DEMA (Double Exponential Moving Average) - [short_ema] & create a long DEMA (Double Exponential Moving Average) - [long_ema]
        elif moving_avg == "DEMA":
            stock_df[short_window_col] = ta.DEMA(stock_df['adjclose'], short_window)
            stock_df[long_window_col] = ta.DEMA(stock_df['adjclose'], long_window)
            
        # Create a short TEMA (Triple Exponential Moving Average) - [short_ema] & create a long TEMA (Triple Exponential Moving Average) - [long_ema]
        elif moving_avg == "TEMA":
            stock_df[short_window_col] = ta.TEMA(stock_df['adjclose'], short_window)
            stock_df[long_window_col] = ta.TEMA(stock_df['adjclose'], long_window)    
            
        # Create a short TRIMA (Triangular Moving Average) - [short_ema] & create a long TRIMA (Triangular Moving Average) - [long_ema]
        elif moving_avg == "TRIMA":
            stock_df[short_window_col] = ta.TRIMA(stock_df['adjclose'], short_window)
            stock_df[long_window_col] = ta.TRIMA(stock_df['adjclose'], long_window)                              

        """ 
            > create a new column 'Signal' such that if faster moving average is 
            > greater than slower moving average
            > then set Signal as 1 else 0.
            > create a new column 'Position' which is a day-to-day difference of the 'Signal' column.
            > Determine current BUY/SELL Status of Security
        """
        stock_df["Signal"] = 0.0
        stock_df["Signal"] = np.where(stock_df[short_window_col] > stock_df[long_window_col], 1.0, 0.0)
        stock_df["Position"] = stock_df["Signal"].diff()
        stock_df = stock_df.loc['2021-05':].dropna()
        df_pos = stock_df[(stock_df["Position"] == 1) | (stock_df["Position"] == -1)]
        df_pos["Position"] = df_pos["Position"].apply(lambda x: "Buy" if x == 1 else "Sell")


        if self.graphit == True:                 
            fig = go.Figure()
            fig.add_scattergl(x=stock_df.index, y=stock_df['adjclose'], name=f"{self.stock} Price", line=dict(color="black", width=3))
            fig.update_traces(mode="markers+lines")
            fig.add_scattergl(x=stock_df.index, y=stock_df[f'{short_window_col}'], name=short_window_col, line=dict(color="salmon", width=2)) 
            fig.add_scattergl(x=stock_df.index, y=stock_df[f'{long_window_col}'], name=long_window_col, line=dict(color="mediumpurple", width=2))
            fig.add_scattergl(
                mode = 'markers',
                marker_symbol='triangle-up',
                x = stock_df[stock_df["Position"] == 1].index,
                y = stock_df['adjclose'][stock_df["Position"] == 1],
                marker = dict(color='green', size=20, line=dict(color='darkgreen', width=2)), name='BUY'
                )
            fig.add_scattergl(
                mode = 'markers',
                marker_symbol='triangle-down',            
                x = stock_df.index[stock_df["Position"] == -1],
                y = stock_df['adjclose'][stock_df["Position"] == -1],            
                marker = dict(color='red', size=20, line=dict(color='crimson', width=2 )), name='SELL'
                )
            fig.update_layout(
                title=f"{self.stock} Buy/Sell Status",
                title_font_color="royalblue",
                title_font_family="Times New Roman",
                xaxis_title="Date",
                xaxis_title_font_color="darkred",
                yaxis_title="Price ($)",
                yaxis_title_font_color="darkred",
                legend_title="Legend Title",
                legend_title_font_color="darkred",
                font=dict(family="Times New Roman", size=18, color="black"),
            )
            st.plotly_chart(fig, use_container_width=True)   
            
        if not df_pos.empty:
            if df_pos["Position"][-1] == "Buy":
                st.metric(
                    f"No. {self.cc} / {self.ccc} ~ [{moving_avg}]",
                    f"{self.stock}",
                    f"{df_pos['Position'][-1]}",
                )
                st.dataframe(df_pos.iloc[-1:])
                return self.stock
            elif df_pos["Position"][-1] == "Sell":
                st.metric(
                    f"No. {self.cc} / {self.ccc} ~ [{moving_avg}]",
                    f"{self.stock}",
                    f"- {df_pos['Position'][-1]}",
                )
                st.dataframe(df_pos.iloc[-1:])
                return            
            else:
                st.metric(
                    f"No. {self.cc} / {self.ccc} ~ [{moving_avg}]",
                    f"{self.stock}",
                    f"- SELL",
                )
                st.dataframe(df_pos.iloc[-1:])
                return
            
            


    def kingpin(self, mod1, data=None):            

        if mod1 == 'SMA':
            S, L = s1.optimal_2sma(self.stock).grab_data(data, mod1)
            ret = self.ma_crossover(data, mod1, S, L)
            if ret == self.stock:
                return self.stock
            else:
                return              
            
        if mod1 == 'EMA':
            S, L = s1.optimal_2sma(self.stock).grab_data(data, mod1)
            ret = self.ma_crossover(data, mod1, S, L)
            if ret == self.stock:
                return self.stock
            else:
                return  
            
        elif mod1 == 'WMA':
            S, L = s1.optimal_2sma(self.stock).grab_data(data, mod1)
            ret = self.ma_crossover(data, mod1, S, L)
            if ret == self.stock:
                return self.stock
            else:
                return
            
        elif mod1 == 'DEMA':
            S, L = s1.optimal_2sma(self.stock).grab_data(data, mod1)
            ret = self.ma_crossover(data, mod1, S, L)
            if ret == self.stock:
                return self.stock
            else:
                return                                                      
        
        elif mod1 == 'TEMA':
            S, L = s1.optimal_2sma(self.stock).grab_data(data, mod1)
            ret = self.ma_crossover(data, mod1, S, L)
            if ret == self.stock:
                return self.stock
            else:
                return    
            
        elif mod1 == 'TRIMA':
            S, L = s1.optimal_2sma(self.stock).grab_data(data, mod1)
            ret = self.ma_crossover(data, mod1, S, L)
            if ret == self.stock:
                return self.stock
            else:
                return             
            
        elif mod1 == 'SAR':
            ret = s1.SAR_Build(self.stock, self.date1, self.graphit, self.cc, self.ccc).build1(data)
            if ret == self.stock:
                return self.stock
            else:
                return                                              

        elif mod1 == "BB":
            ret = self.bollinger_bands(data)
            if ret == self.stock:
                return self.stock
            else:
                return

        elif mod1 == "MACD":
            ret = self.macd(data)
            if ret == self.stock:
                return self.stock
            else:
                return

        elif mod1 == "RSI":
            ret = self.rsi()
            if ret == self.stock:
                return self.stock
            else:
                return