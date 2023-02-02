from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from src.tools.functions import company_longName
from yahooquery import Ticker
import numpy as np

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.options.display.float_format = "{:,}".format

np.random.seed(42)




class Proof_of_Concept_Builder(object):
    
    
    def __init__(self, today_stamp):
        self.ender_date = str(datetime.now())[:10]
        self.save_output = True
        self.graphit = True
        self.today_stamp = str(today_stamp)[:10]
        self.saveMonth = self.today_stamp[:7]
        
        self.final_loc = Path(f"reports/port_results/{self.saveMonth}/{self.today_stamp}/")
        if not self.final_loc.exists():
            self.final_loc.mkdir(parents=True)
        
        self.saveReport = Path(f"reports/portfolio/{self.saveMonth}/{self.today_stamp}/")    
        if not self.saveReport.exists():
            self.saveReport.mkdir(parents=True)
            
        self.saveAdvisor = Path(f"data/advisor/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveAdvisor.exists():
            self.saveAdvisor.mkdir(parents=True)
        
        self.saveProof = Path(f"data/proof/{self.today_stamp}/{self.ender_date}/")
        if not self.saveProof.exists():
            self.saveProof.mkdir(parents=True)
            
        self.day = int(str(today_stamp)[8:10])
        self.month = int(str(today_stamp)[5:7])
        self.year = int(str(today_stamp)[:4])
        self.starter_date = datetime(self.year, self.month, self.day) + timedelta(days=1)
        self.og_day = str(self.today_stamp)[:10]


    def setup(self, portfolio_file, namer, data, initial_cash=5000):
        self.namer = namer
        self.initial_cash = initial_cash
        df_0 = pd.DataFrame(data).round(2)
                
        og_wt = portfolio_file["allocation"].sum()
        new_wt_lst = []
        for i in portfolio_file["allocation"]:
            new_wt_lst.append((i * 100) / og_wt)
        portfolio_file["allocation"] = new_wt_lst
        portfolio_file = pd.DataFrame(portfolio_file).sort_values("ticker")
        
        divisor = len(portfolio_file["ticker"])
        total_allocation = portfolio_file["allocation"].sum() / 100
        
        
        proof = pd.DataFrame(portfolio_file[["ticker", "allocation"]])
        proof = proof.sort_values("ticker")
        b = []
        for i in proof["ticker"]:
            b.append(company_longName(i))
        proof["companyName"] = b
        
        
        port_tics = sorted(list(proof["ticker"]))
        string_tickers = ''
        for i in port_tics:
            if i != port_tics[-1]:
                string_tickers += (i+' ')
            else:
                string_tickers += (i)        
        tickers = Ticker(string_tickers, asynchronous=True)
        df3 = tickers.history(period='1y')  
        df3.columns = [e.title() for e in df3.columns]        
        df_close = pd.DataFrame()
        df_open = pd.DataFrame()        
        for s in port_tics:
            try:            
                one = df3.T[s]
                df = one.T
                df_close[s] = df['Adjclose']
                df_open[s] = df['Open']
            except:
                print(f"failed ticker {i}")
                proof = proof.drop(proof[proof.ticker == i].index)
                port_tics.remove(i)
        df_close = df_close.dropna(axis="columns")
        df_open = df_open.dropna(axis="columns")
        df_close.index = pd.to_datetime(df_close.index)
        df_open.index = pd.to_datetime(df_open.index)        
        
        
        try:
            proof["start_price"] = list(df_open.iloc[-1])
        except Exception:
            proof["start_price"] = list(df_close.iloc[-1])
        proof["current_price"] = list(df_close.iloc[-1])
        proof["investment"] = round(self.initial_cash * (proof["allocation"] / 100), 2)
        proof["shares"] = round(proof["investment"] / proof["start_price"], 2)
        proof["cash_now"] = round(proof["shares"] * proof["current_price"], 2)
        proof["return"] = round(((proof["cash_now"] - proof["investment"]) / proof["investment"]) * 100, 2,)
        
        
        spy_tics = ['SPY']
        tickers = Ticker('SPY', asynchronous=True)
        df3 = tickers.history(period='1y')  
        df3.columns = [e.title() for e in df3.columns]      
        for s in spy_tics:
            one = df3.T[s]
            spy_hist = pd.DataFrame(one.T)        
        proof_spy = pd.DataFrame(["SPY"], columns=["SPY"])
        proof_spy["start_price"] = spy_hist["Open"][-1]
        proof_spy["current_price"] = spy_hist["Adjclose"][-1]
        proof_spy["investment"] = round(self.initial_cash / len(proof_spy["SPY"]), 2)
        proof_spy["shares"] = round(proof_spy["investment"] / proof_spy["start_price"], 2)
        proof_spy["cash_now"] = round(proof_spy["shares"] * proof_spy["current_price"], 2)
        proof_spy["return"] = round(((proof_spy["cash_now"] - proof_spy["investment"])/ proof_spy["investment"])* 100,2,)


        high_watermark_spy = round(proof_spy["return"].max(), 2)
        low_watermark_spy = round(proof_spy["return"].min(), 2)
        beat_num = proof_spy["return"][0]
        proof_2 = proof[proof["return"] > 0.0]
        proof_3 = proof_2[proof_2["return"] > beat_num]
        winning_percentage = round((len(proof_2["ticker"]) / divisor) * 100, 2)
        beat_spy_percentage = round((len(proof_3["ticker"]) / divisor), 2)


        one = pd.DataFrame(df_0.copy())
        proof_tickers_list = list(proof["ticker"])
        proof_allocation_list = list(proof["allocation"])
        cash_list=[]
        for i in proof_allocation_list:
            cash_list.append((i / 100)* 5000)
        shares = []
        for k, v in enumerate(proof_tickers_list):
            shares.append(((proof_allocation_list[k] / 100) * initial_cash) / one[v].iloc[0])
        for k, v in enumerate(list(proof["ticker"])):
            one[v] = one[v] * shares[k]
            
            
        lst = list(proof["ticker"])
        one["portfolio"] = one[lst].sum(axis=1)
        start_cash = round(proof["investment"].sum(), 2)
        avg_1 = round(one["portfolio"].mean(), 2)
        high_1 = round(one["portfolio"].max(), 2)
        low_1 = round(one["portfolio"].min(), 2)
        mean_watermark = round(((avg_1 - start_cash) / start_cash) * 100, 2)
        high_watermark = round(((high_1 - start_cash) / start_cash) * 100, 2)
        low_watermark = round(((low_1 - start_cash) / start_cash) * 100, 2)
        mean_watermark_spy = round(proof_spy["return"].mean(), 2)
        high_watermark_spy = round(proof_spy["return"].max(), 2)
        low_watermark_spy = round(proof_spy["return"].min(), 2)
        beat_num = proof_spy["return"][0]
        proof_2 = proof[proof["return"] > 0.0]
        proof_3 = proof_2[proof_2["return"] > beat_num]


        for i in list(one["portfolio"]):
            if float(i) > high_1:
                high_1 = float(i)
            else:
                pass
        

        one["since_open"] = round(((one["portfolio"] - start_cash) / start_cash) * 100, 2)
        try:
            act_ror = round(((list(one["portfolio"])[-1] - list(one["portfolio"])[0])/ list(one["portfolio"])[0])* 100,2,)
        except Exception:
            act_ror = "ongoing"
        one["since_open"] = round(((one["portfolio"] - start_cash) / start_cash) * 100, 2)


        gdp = pd.DataFrame(
            ["Recommended Stocks", "SPY Index"], columns=["strategy_vs_benchmark"]
        )
        gdp["starting_money"] = [
            f"${round(list(one['portfolio'])[0],2)}",
            f"${round(proof_spy['investment'].sum(),2)}",
        ]
        gdp["ending_money"] = [
            f"${round(list(one['portfolio'])[-1],2)}",
            f"${round(proof_spy['cash_now'].sum(), 2)}",
        ]
        gdp["return"] = [
            f"{round(act_ror,2)}%",
            f"{round(float(proof_spy['return']),2)}%",
        ]
        gdp["mean_mark"] = [
            f"{mean_watermark}%",
            f"{mean_watermark_spy}%",
        ]
        gdp["high_mark"] = [
            f"{high_watermark}%",
            f"{high_watermark_spy}%",
        ]
        gdp["low_mark"] = [
            f"{low_watermark}%",
            f"{low_watermark_spy}%",
        ]
        gdp = gdp.set_index("strategy_vs_benchmark")


        st.write('𝄗𝄗𝄗'*20)
        st.header(f"> __Monte Carlo Cholesky [Sharpe] vs SPY__")
        st.write(f" - Start Position [{self.starter_date}] ")
        st.write(f" - Today's Position [{str(datetime.now())[:10]}] ")
        st.write(f" - Total Allocation: {round(total_allocation*100,2)}%")
        st.write(f" - __Initial Portfolio Optimization Modeled On {self.starter_date}__")  
        st.table(gdp)


        proof = proof.sort_values("return", ascending=False)
        first_column = proof.pop('companyName')
        proof.insert(0, 'companyName', first_column)          
                
        st.write('__Portfolio Composition__')
        st.table(proof.set_index("ticker"))




        if self.save_output == True:
            gdp = pd.DataFrame(gdp)
            proof = pd.DataFrame(proof)
            proof_spy = pd.DataFrame(proof_spy)
            one = pd.DataFrame(one)
            del one["since_open"]

            gdp.to_csv(self.final_loc / f"spy_vs_{namer}.csv")
            proof.to_csv(self.final_loc / f"{namer}.csv")
            proof_spy.to_csv(self.final_loc / f"spy.csv")
            one.to_csv(self.final_loc / f"one_{self.namer}.csv")

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(proof)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"{str(self.final_loc)}/{str(self.namer)}.csv",
                mime="text/csv",
                key=str(self.namer),
            )
            return
        else:
            return
