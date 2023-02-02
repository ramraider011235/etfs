from ast import Pass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.tools.functions import company_longName
from yahooquery import Ticker
import yfinance as yf

from src.tools import functions as f0



class Model_Concept(object):
    
    
    
    def __init__(self, today_stamp, initial_cash=20000.0, namer=None, save_output=True, graphit=True,):
        self.save_output = save_output
        self.graphit = graphit
        self.namer = namer
        self.initial_cash = initial_cash        

        self.start_date = str(today_stamp)[:10]
        self.end_date = str(datetime.now())[:10]
        self.day2 = int(str(today_stamp)[8:10])
        self.month1 = str(today_stamp)[:7]        
        self.month2 = int(str(today_stamp)[5:7])
        self.year2 = int(str(today_stamp)[:4])
        self.official_date = datetime(self.year2, self.month2, self.day2) + timedelta(days=1)

        self.final_loc = Path(f"reports/port_results/{self.month1}/{self.start_date}/")
        if not self.final_loc.exists():
            self.final_loc.mkdir(parents=True)
                        
            
            
    def setup(self, portfolio_file, data):         
        port_tics = list(portfolio_file['ticker'])
        df_hist = yf.download(port_tics, start=self.start_date)
        df_hist.index = pd.to_datetime(df_hist.index)
        df_hist = pd.DataFrame(df_hist.copy())
        df_wt = pd.DataFrame(portfolio_file.copy()).sort_values('ticker', ascending=True)
        try:
            del df_wt['Unnamed: 0']
        except Exception:
            pass
        
        df_wt['investment'] = self.initial_cash * (df_wt['allocation'] / 100)
        try:
            df_wt['start_price'] = list(df_hist['Open'].iloc[1].round(2))
        except Exception:
            df_wt['start_price'] = list(df_hist['Close'].iloc[0].round(2))
        df_wt['shares'] = round(df_wt['investment'] / df_wt['start_price'], 0)
        df_wt['investment'] = round(df_wt['shares'] * df_wt['start_price'], 2)
        # df_wt['current_price'] = list(df_hist['Close'].iloc[-1].round(2))
        # df_wt['max_price'] = list(df_hist['High'].max().round(2))
        # df_wt['min_price'] = list(df_hist['Low'].min().round(2))
        # df_wt['present_value'] = round(round(df_wt['shares'],0) * df_wt['current_price'], 2)
        # df_wt['max_value'] = round(round(df_wt['shares'],0) * df_wt['max_price'], 2)
        # df_wt['min_value'] = round(round(df_wt['shares'],0) * df_wt['min_price'], 2)
        # df_wt['return'] = ((df_wt['present_value'] - df_wt['investment']) / df_wt['investment']) * 100
        
        b = []
        for i in df_wt["ticker"]:
            b.append(f0.company_longName(i))
        df_wt["companyName"] = b                 
        
        fd = pd.DataFrame(df_wt)
        col_1 = fd.pop('companyName')
        col_2 = fd.pop('ticker')
        col_3 = fd.pop('allocation')
        col_4 = fd.pop('shares')
        col_5 = fd.pop('start_price')
        # col_6 = fd.pop('current_price')
        # col_7 = fd.pop('max_price')
        # col_8 = fd.pop('min_price')
        col_9 = fd.pop('investment')
        # col_10 = fd.pop('present_value')
        # col_11 = fd.pop('max_value')
        # col_12 = fd.pop('min_value')
        # col_13 = fd.pop('return')
        
        # fd.insert(0, 'return', col_13)
        # fd.insert(0, 'min_value', col_12)
        # fd.insert(0, 'max_value', col_11)           
        # fd.insert(0, 'present_value', col_10)
        fd.insert(0, 'investment', col_9)
        # fd.insert(0, 'min_price', col_8)
        # fd.insert(0, 'max_price', col_7)
        # fd.insert(0, 'current_price', col_6)
        fd.insert(0, 'start_price', col_5)           
        fd.insert(0, 'shares', col_4)     
        fd.insert(0, 'allocation', col_3)  
        fd.insert(0, 'ticker', col_2)
        fd.insert(0, 'companyName', col_1)
        df_wt = pd.DataFrame(fd)
        
        
        df_live = df_hist['Close'].iloc[1:].copy()
        tickers = list(df_live.columns)
        for ticker in tickers:
            share_val = float(df_wt[df_wt['ticker'] == ticker]['shares'])
            df_live[ticker] = df_live[ticker] * share_val
        df_live['portfolio'] = df_live.sum(axis=1)
        
        # compute daily returns using pandas pct_change()
        df_daily_returns = df_live.copy().pct_change()[1:]

        # Calculate the cumulative daily returns
        df_cum_daily_returns = (1 + df_daily_returns).cumprod() - 1
        df_cum_daily_returns = df_cum_daily_returns.reset_index()   
        
        # reset the index, moving `date` as column
        df_daily_returns = df_daily_returns.reset_index()
        df1 = df_daily_returns.melt(id_vars=['Date'], var_name='ticker', value_name='daily_return')
        df1['daily_return_pct'] = df1['daily_return'] * 100          
        
        df2 = df_cum_daily_returns.melt(id_vars=['Date'], var_name='ticker', value_name='cum_return')
        df2['cum_return_pct'] = df2['cum_return'] * 100      
        
        
        hammerTime = Ticker(["SPY"], asynchronous=True, formatted=False, backoff_factor=0.34, validate=True, verify=True)
        hammer_hist = pd.DataFrame(hammerTime.history(start=self.start_date)).reset_index().set_index("date")
        hammer_hist.index = pd.to_datetime(hammer_hist.index)
        hammer_hist = hammer_hist.rename(columns={"symbol": "ticker"})
        spy_hist = pd.DataFrame(hammer_hist.copy())#.iloc[1:]
        
        proof_spy = pd.DataFrame(["SPY"], columns=["SPY"])
        try:
            proof_spy['start_price'] = list(spy_hist['open'].iloc[1].round(2))
        except Exception:
            proof_spy["start_price"] = round(spy_hist["adjclose"][0], 2)      
        
        # proof_spy["current_price"] = round(spy_hist["adjclose"][0], 2)
        proof_spy["investment"] = round(df_wt['investment'].sum() / len(proof_spy["SPY"]), 2)
        proof_spy["shares"] = round(proof_spy["investment"] / proof_spy["start_price"], 0)
        # proof_spy["present_value"] = round(proof_spy["shares"] * proof_spy["current_price"], 2)
        # proof_spy['max_price'] = spy_hist["high"].max()
        # proof_spy['min_price'] = spy_hist["low"].min()
        # proof_spy['max_value'] = round(round(proof_spy['shares'],0) * proof_spy['max_price'], 2)
        # proof_spy['min_value'] = round(round(proof_spy['shares'],0) * proof_spy['min_price'], 2)        
        # proof_spy["return"] = round(((proof_spy["present_value"] - proof_spy["investment"])/ proof_spy["investment"])* 100,2,)
        
        fd = pd.DataFrame(proof_spy)
        col_4 = fd.pop('shares')
        col_5 = fd.pop('start_price')
        # col_6 = fd.pop('current_price')
        # col_7 = fd.pop('max_price')
        # col_8 = fd.pop('min_price')
        col_9 = fd.pop('investment')
        # col_10 = fd.pop('present_value')
        # col_11 = fd.pop('max_value')
        # col_12 = fd.pop('min_value')
        # col_13 = fd.pop('return')
        
        # fd.insert(0, 'return', col_13)
        # fd.insert(0, 'min_value', col_12)
        # fd.insert(0, 'max_value', col_11)        
        # fd.insert(0, 'present_value', col_10)
        fd.insert(0, 'investment', col_9)
        # fd.insert(0, 'min_price', col_8)
        # fd.insert(0, 'max_price', col_7)
        # fd.insert(0, 'current_price', col_6)
        fd.insert(0, 'start_price', col_5)           
        fd.insert(0, 'shares', col_4)     
        proof_spy = pd.DataFrame(fd)
        self.proof_spy = pd.DataFrame(proof_spy.copy())        
              
        
        # gdp = pd.DataFrame(["Recommended Stocks", "SPY Index"], columns=["strategy_vs_benchmark"])
        # a1 = (f"{round(df_wt['investment'].sum(),2):,}")
        # b1 = (f"{round(df_wt['present_value'].sum(),2):,}")
        # c1 = (f"{round(df_live['portfolio'].max(),2):,}")
        # d1 = (f"{round(df_live['portfolio'].min(),2):,}")
        # e1 = (f"{round((((df_wt['present_value'].sum() - df_wt['investment'].sum()) / df_wt['investment'].sum())*100),2):,}")
        # f1 = (f"{round((((df_live['portfolio'].max() - df_wt['investment'].sum()) / df_wt['investment'].sum())*100),2):,}")
        # g1 = (f"{round((((df_live['portfolio'].min() - df_wt['investment'].sum()) / df_wt['investment'].sum())*100),2):,}")        
        
        
        # gdp["starting_money"] = [
        #     f"${a1}",
        #     f"${round(self.proof_spy['investment'].sum(),2):,}",
        #     ]
        # gdp["ending_money"] = [
        #     f"${b1}",
        #     f"${round(self.proof_spy['present_value'].sum(), 2):,}",
        #     ]
        # gdp["high_mark"] = [
        #     f"${c1}",
        #     f"${round((spy_hist['high'].max() * proof_spy['shares'][0]),2):,}",
        #     ]
        # gdp["low_mark"] = [
        #     f"${d1}",
        #     f"${round((spy_hist['low'].min() * proof_spy['shares'][0]),2):,}",
        #     ]
        # gdp["return"] = [
        #     f"{e1}%",
        #     f"{round(float(self.proof_spy['return']),2):,}%",
        #     ]
        # gdp["high_mark2"] = [
        #     f"{f1}%",
        #     f"{round((((proof_spy['max_value'][0] - proof_spy['investment'][0]) / proof_spy['investment'][0]) * 100), 2):,}%",
        #     ]
        # gdp["low_mark2"] = [
        #     f"{g1}%",
        #     f"{round((((proof_spy['min_value'][0] - proof_spy['investment'][0]) / proof_spy['investment'][0]) * 100), 2):,}%",
        #     ]        
        
        # st.write(f"▶ __Portfolio vs SPY__")
        # st.table(gdp.set_index('strategy_vs_benchmark'))
        
        st.write(f"▶ __Portfolio Snapshot__")
        table_a = pd.DataFrame(df_wt.sort_values('allocation', ascending=False))
        try:
            del table_a['index']
        except Exception as e:
            pass
        try:
            del table_a['level_0']
        except Exception as e:
            pass
        
        
        st.table(table_a.set_index('companyName'))        
                  
        

        
        
        if self.graphit == True:

            start_l = [df_wt['investment'].sum()] * len(df_live)
            win_l = [df_wt['investment'].sum() * 1.1] * len(df_live)
            loss_l = [df_wt['investment'].sum() * 0.9] * len(df_live)
            df_live["start_line"] = start_l
            df_live["win_l"] = win_l
            df_live["loss_l"] = loss_l     
            
            fig = go.Figure()
            fig.add_scattergl(x=df_live.index, y=df_live['portfolio'], line={"color": "black"}, name="Portfolio Line")
            fig.update_traces(line=dict(color="Black", width=2.5))
            fig.update_traces(mode="markers+lines")
            fig.add_trace(go.Scatter(x=df_live.index, y=df_live['start_line'], name="starting_balance", line_shape="hvh", line=dict(color="#7f7f7f", width=4)))
            fig.add_trace(go.Scatter(x=df_live.index, y=df_live['win_l'], name="Win Threshold", line_shape="hvh", line=dict(color="#2ca02c", width=4)))
            fig.add_trace(go.Scatter(x=df_live.index, y=df_live['loss_l'], name="Loss Threshold", line_shape="hvh", line=dict(color="#d62728", width=4)))
            fig.update_layout(
                title="Portfolio Performance",
                title_font_color="royalblue",
                title_font_family="Times New Roman",
                xaxis_title="Days Since Bought Portfolio",
                xaxis_title_font_color="darkred",
                yaxis_title="Portfolio Value ($)",
                yaxis_title_font_color="darkred",
                legend_title="Legend Title",
                legend_title_font_color="darkred",
                font=dict(family="Times New Roman", size=18, color="black"),
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = px.line(
                df2[df2['ticker'] == 'portfolio'],
                x='date',
                y='cum_return_pct', 
                color='ticker',
                title='Performance - Daily Cumulative Returns',
                labels={'cum_return_pct':'daily cumulative returns (%)', }
            )      
            st.plotly_chart(fig, use_container_width=True)  
            
            
            fig = px.line(
                df2[df2['ticker'] != 'portfolio'],
                x='date',
                y='cum_return_pct', 
                color='ticker',
                title='Performance - Daily Cumulative Returns',
                labels={'cum_return_pct':'daily cumulative returns (%)', }
            )                
            st.plotly_chart(fig, use_container_width=True)  
            
            
        if self.save_output == True:
            # gdp.to_csv(self.final_loc / f"spy_vs_{self.namer}.csv")
            table_a.to_csv(self.final_loc / f"{self.namer}.csv")
            self.proof_spy.to_csv(self.final_loc / f"spy.csv")
            
            
        st.markdown(list(df_wt["ticker"]))
