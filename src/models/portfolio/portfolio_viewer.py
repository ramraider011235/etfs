from datetime import datetime, timedelta
from genericpath import exists
from pathlib import Path
from yahooquery import Ticker
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import datetime as dt
import streamlit as st
import yfinance as yf
from src.tools.functions import company_longName
import src.tools.functions as f0

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.options.display.float_format = "{:,}".format



class Proof_of_Concept_Viewer(object):
    
    
    def __init__(self, day_101, interval1='1d', save_output=True, graphit=True):
        self.save_output = save_output
        self.graphit = graphit
        self.inter1 = interval1
        self.day_0 = str(day_101)[:10]
        self.month_0 = str(self.day_0)[:7]
        self.year_0 = str(self.day_0)[:4]
        day = int(str(day_101)[8:10])
        month = int(str(day_101)[5:7])
        year = int(str(day_101)[:4])
        self.day_1 = datetime(year, month, day) + timedelta(days=1)
        now = datetime.now()
        diff1 = timedelta(days=1)
        self.future_1 = (now + diff1).strftime("%Y-%m-%d")              
        diff2 = timedelta(days=-1)
        self.past_1 = (now + diff2).strftime("%Y-%m-%d")                      
        
        self.saveRec = Path(f"data/recommenders/{self.year_0}/{self.month_0}/{self.day_0}/")
        self.saveReport = Path(f"reports/portfolio/{self.month_0}/{self.day_0}/")
        self.final_loc = Path(f"reports/port_results/{self.month_0}/{self.day_0}/")            
        self.saveAdvisor = Path(f"data/advisor/{self.month_0}/{self.day_0}/")
        
        
        cherry_pick = day_101
        self.cherry_pick = cherry_pick
        self.start1 = str(cherry_pick)[:10]
        self.day1 = str(cherry_pick)[8:]
        self.month1 = str(cherry_pick)[:7]
        self.month2 = str(cherry_pick)[5:7]
        self.year1 = str(cherry_pick)[:4]
        self.ender_date = str(dt.datetime.now())[:10]
        start_date_101 = dt.date(int(str(cherry_pick)[:4]), int(str(cherry_pick)[5:7]), int(str(cherry_pick)[8:]))
        self.years_ago = str(start_date_101 - relativedelta(years=1, days=0))[:10]        
        self.saveRaw = Path(f"data/raw/{self.month1}/{self.start1}/")        


    def performance(self, portfolio_file):
        self.file = portfolio_file
        self.port_tics = sorted(list(portfolio_file["ticker"]))
        self.namer = "monte_carlo_cholesky"
        self.namer2 = "Monte Carlo Cholesky (Sharpe Style)"


        def section_proof_df():
            proof = pd.DataFrame(self.file)
            proof = proof.sort_values("ticker")       
            b = []
            for i in proof["ticker"]:
                b.append(company_longName(i))
            proof["companyName"] = b                  
            start_value = float(proof['investment'].sum())
            
            
            def pull_history_2(ticker_list):
                df = pd.DataFrame(yf.download(ticker_list, interval=self.inter1, start=self.start1, rounding=True)['Adj Close'])
                df.index = pd.to_datetime(df.index)
                df = pd.DataFrame(df.loc[self.start1:])
                df.to_csv(self.final_loc / f"{self.namer}_history_{self.day_0}_{str(datetime.now())[:10]}.csv")
                port_tics_0 = list(df.columns)             
                return df.dropna(),  port_tics_0


            # if exists(self.final_loc / f"{self.namer}_history_{self.day_0}_{str(datetime.now())[:10]}.csv"):
            #     try:
            #         df_test_data = pd.read_csv(self.final_loc / f"{self.namer}_history_{self.day_0}_{str(datetime.now())[:10]}.csv").set_index('Date')
            #     except Exception as e:
            #         df_test_data, self.port_tics = pull_history_2(self.port_tics)
            
            df_test_data, self.port_tics = pull_history_2(self.port_tics)            
                        
            
            proof["current_price"] = df_test_data.iloc[-1].values
            proof["cash_now"] = proof["shares"] * proof["current_price"]
            proof["return"] = round(((proof["cash_now"] - proof["investment"]) / proof["investment"]) * 100,2,)
            proof["return_cash"] = (proof["cash_now"] - proof["investment"])
            proof['allocation'] = round((proof['investment'] / start_value) * 100, 2)
            proof = proof.round(2)
            shares = list(proof['shares'])
            one = pd.DataFrame(df_test_data.copy())
            col_lst = one.columns
            for s in range(len(col_lst)):
                one[col_lst[s]] = one[col_lst[s]] * shares[s] 
            one['portfolio'] = one.sum(axis=1)  
            return proof, one


        def section_spy_df():
            spy_data = yf.download("SPY", start=self.day_1, interval=self.inter1)
            proof_spy = pd.DataFrame(["SPY"], columns=["SPY"])
            proof_spy["start_price"] = spy_data["Open"][0]
            proof_spy["current_price"] = spy_data["Adj Close"][-1]
            proof_spy["investment"] = round(proof['investment'].sum(), 2)
            proof_spy["shares"] = round(proof_spy["investment"] / proof_spy["start_price"], 2)
            proof_spy["cash_now"] = round(proof_spy["shares"] * proof_spy["current_price"], 2)
            proof_spy["return"] = round(((proof_spy["cash_now"] - proof_spy["investment"])/ proof_spy["investment"])* 100,2,)
            proof_spy["return_cash"] = (proof_spy["cash_now"] - proof_spy["investment"])            
            divisor = len(self.file["ticker"])
            total_allocation = self.file["allocation"].sum() / 100
            beat_num = proof_spy["return"][0]
            proof_2 = proof[proof["return"] > 0.0]
            proof_3 = proof_2[proof_2["return"] > beat_num]
            winning_percentage = round((len(proof_2["ticker"]) / divisor) * 100, 2)
            beat_spy_percentage = round((len(proof_3["ticker"]) / divisor), 2)
            ret_lst = [proof_spy, divisor, total_allocation, beat_num, proof_2, proof_3, winning_percentage, beat_spy_percentage, spy_data]
            return ret_lst


        def section_one_df(proof, one, spy_data):
            port_start = start_cash = round(proof['investment'].sum(), 2)
            port_end = round(one['portfolio'].iloc[-1], 2)
            port_return = round(((port_end - port_start) / port_start)*100,2)
            spy_start = round(proof_spy['investment'].sum(), 2)
            spy_end = round(proof_spy['cash_now'].sum(), 2)
            spy_return = round(((spy_end - spy_start) / spy_start)*100,2)
            spy_low = min(spy_data["Low"])
            spy_high = max(spy_data["High"])
            spy_opener = spy_data["Open"][0]
            high_1 = round(one["portfolio"].max(), 2)
            low_1 = round(one["portfolio"].min(), 2)            
            high_watermark = round(((high_1 - start_cash) / start_cash) * 100, 2)
            high_watermark_spy = round(((spy_high - spy_opener) / spy_opener) * 100, 2)
            low_watermark = round(((low_1 - start_cash) / start_cash) * 100, 2)
            low_watermark_spy = round(((spy_low - spy_opener) / spy_opener) * 100, 2)

            gdp = pd.DataFrame(["Recommended Stocks", "SPY Index"], columns=["strategy_vs_benchmark"])
            
            if self.day_0 == '2022-09-15':
                port_start = (port_start + 5237.31)
                port_end = (port_end+5965.44)
                gdp["starting_money"] = [port_start, spy_start]
                gdp["current_money"] = [port_end, spy_end]
                gdp["return_cash"] = [round((port_end - port_start), 2), round((spy_end - spy_start), 2)]
                
            elif self.day_0 == '2022-10-25':
                port_start = (port_start + 3304.13)
                port_end = (port_end+3298.9)
                gdp["starting_money"] = [port_start, spy_start]
                gdp["current_money"] = [port_end, spy_end]
                gdp["return_cash"] = [round((port_end - port_start), 2), round((spy_end - spy_start), 2)]                
                
            else:          
                gdp["starting_money"] = [port_start, spy_start]
                gdp["current_money"] = [port_end, spy_end]
                gdp["return_cash"] = [round((port_end - port_start), 2), round((spy_end - spy_start), 2)]
                
                
            gdp["return"] = [f"{port_return}%", f"{spy_return}%"]
            gdp["high_mark"] = [f"{high_watermark}%",f"{high_watermark_spy}%",]
            gdp["low_mark"] = [f"{low_watermark}%",f"{low_watermark_spy}%",]
            gdp = gdp.set_index("strategy_vs_benchmark")
            gdp['starting_money'] = [f"${x:,.2f}" for x in list(gdp['starting_money'])]
            gdp['current_money'] = [f"${x:,.2f}" for x in list(gdp['current_money'])]
            gdp['return_cash'] = [f"${x:,.2f}" for x in list(gdp['return_cash'])]

            for i in list(one["portfolio"]):
                if float(i) > high_1:
                    high_1 = float(i)
                else:
                    pass

            spy_data = pd.DataFrame(yf.download("SPY", start='2022-08-11', interval='1d')['Adj Close']).round(2)
            spy_data.columns = ["SPY"]        
            spy_data["shares"] = [round(round(one['portfolio'].iloc[0],2) / spy_data["SPY"].iloc[0], 0)] * len(spy_data)
            spy_data["SPY_Portfolio"] = spy_data["SPY"] * spy_data["shares"]
            one["SPY_Portfolio"] = spy_data["SPY_Portfolio"]
            one["since_open"] = round(((one["portfolio"] - start_cash) / start_cash) * 100, 2)
            r2 = [one, gdp, high_1, low_1]
            return r2


        def section_dictate_to_web_app(proof, gdp):
            watermark_up = gdp['high_mark'].iloc[0]
            watermark_down = gdp['low_mark'].iloc[0]
            st.header(f"__❰◈❱ {self.namer2} vs SPY__")
            st.subheader(f"◾ ↳ Live Portfolio : [{str(self.day_1)[:10]}]")
            st.write(f" ╠══► Initial Portfolio Optimization Modeled On {self.day_1}")
            st.write(f" ╠══► High Watermark ⟿ ❰${round(high_1,2)}❱   ❰{watermark_up}❱")
            st.write(f" ╚══► Low Watermark ⟿ ❰${round(low_1,2)}❱   ❰{watermark_down}❱")            
            st.write('__◾ Combined Portfolio vs SPY (S&P 500 Stock Market Index Fund)__')
            st.table(gdp)
            proof = proof.sort_values("return", ascending=False)
            proof["rank"] = proof["return"].rank(ascending=False)
            del proof['rank']
            proof = proof.round(2)
            st.table(proof.set_index("companyName"))
            return


        def grapher(one):
            one = pd.DataFrame(one)
            two = one.copy()
            df11 = one.copy()
            try:
                start1 = round(proof['investment'].sum(), 2)
                win1 = start1 * 1.1
                loss1 = start1 * 0.9
                start_l = [start1] * len(df11)
                win_l = [win1] * len(df11)
                loss_l = [loss1] * len(df11)
                df11["start_line"] = start_l
                df11["win_l"] = win_l
                df11["loss_l"] = loss_l
                fig = go.Figure()
                fig.add_scattergl(x=df11.index, y=df11.portfolio, line={"color": "black"}, name="Portfolio Line",)
                fig.update_traces(line=dict(color="Black", width=2.0))
                fig.add_scattergl(x=df11.index, y=df11.SPY_Portfolio, line={"color": "crimson"}, name="SPY_Portfolio",)
                fig.update_traces(mode="markers+lines")
                fig.add_trace(go.Scatter(x=df11.index, y=df11.start_line, name="starting_balance", line_shape="hvh", line=dict(color="#7f7f7f", width=4)))
                fig.add_trace(go.Scatter(x=df11.index, y=df11.win_l, name="Win Threshold", line_shape="hvh", line=dict(color="#2ca02c", width=4)))
                fig.add_trace(go.Scatter(x=df11.index, y=df11.loss_l, name="Loss Threshold", line_shape="hvh", line=dict(color="#d62728", width=4)))
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
                    font=dict(family="Times New Roman",size=18,color="black"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.write("_" * 25)
            except Exception:
                st.write("Failed Graph 1")

            try:
                df12 = one.copy()
                df12["start_line"] = start1
                df12["win_l"] = win1
                df12["loss_l"] = loss_l
                fig = px.area(df12["portfolio"], facet_col_wrap=2)
                fig.update_traces(mode="markers+lines")
                fig.add_trace(go.Scatter(x=df12.index, y=df12.start_line, name="starting_balance", line_shape="hvh", line=dict(color="#7f7f7f", width=4), ))
                fig.add_trace(go.Scatter(x=df12.index, y=df12.win_l, name="Win Threshold", line_shape="hvh", line=dict(color="#2ca02c", width=4), ))
                fig.add_trace(go.Scatter(x=df12.index, y=df12.loss_l, name="Loss Threshold", line_shape="hvh", line=dict(color="#d62728", width=4), ))
                fig.update_layout(
                    title="Performance Balance Tracker",
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
                st.write("_" * 25)
            except Exception:
                st.write("Failed Graph 2")

            try:
                df13 = two.copy()
                del df13["portfolio"]
                del df13["since_open"]
                del df13["SPY_Portfolio"]
                fig = px.area(df13, facet_col_wrap=2)
                fig.update_traces(mode="markers+lines")
                fig.update_layout(
                    title="Portfolio Position Balance Tracker",
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
                st.write("_" * 25)
            except Exception:
                st.write("Failed Graph 3")

            try:
                df14 = two.copy()
                try:
                    del df14["portfolio"]
                    del df14["since_open"]
                    del df14["SPY_Portfolio"]
                except Exception as e:
                    pass
                df_daily_returns = pd.DataFrame(df14.pct_change())[1:]
                df_cum_daily_returns = (1 + df_daily_returns).cumprod() - 1
                df_cum_daily_returns = df_cum_daily_returns.reset_index()
                df15 = pd.DataFrame(df_cum_daily_returns.copy()).reset_index()
                try:
                    del df15['index']
                except Exception as e:
                    pass                    
                df15 = df15.set_index("Date")
                df15 = df15 * 100
                start_l = [0.0] * len(df15)
                win_l = [10.0] * len(df15)
                loss_l = [-10.0] * len(df15)            
                fig = px.line(df15, x=df15.index, y=df15.columns)
                fig.update_traces(mode="markers+lines")
                fig.add_trace(go.Scatter(x=df15.index, y=start_l, name="Starting Return", line_shape="hvh", line=dict(color="#7f7f7f", width=4), ) )
                fig.add_trace(go.Scatter(x=df15.index, y=win_l, name="Win Threshold", line_shape="hvh", line=dict(color="#2ca02c", width=4), ) )
                fig.add_trace(go.Scatter(x=df15.index, y=loss_l, name="Loss Threshold", line_shape="hvh", line=dict(color="#d62728", width=4), ) )
                fig.update_layout(
                    title="Portfolio Performance - Daily Simple Returns",
                    title_font_color="royalblue",
                    title_font_family="Times New Roman",
                    xaxis_title="Days Since Bought Portfolio",
                    xaxis_title_font_color="darkred",
                    yaxis_title="Portfolio Returns (%)",
                    yaxis_title_font_color="darkred",
                    legend_title="Legend Title",
                    legend_title_font_color="darkred",
                    font=dict(family="Times New Roman", size=18, color="black"),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write("Failed Graph 4")


        def file_saver(gdp, proof, proof_spy, one):
            if self.save_output == True:
                gdp = pd.DataFrame(gdp)
                proof = pd.DataFrame(proof)
                proof_spy = pd.DataFrame(proof_spy)
                one = pd.DataFrame(one)
                gdp.to_csv(self.final_loc / f"spy_vs_{self.namer}.csv")
                proof.to_csv(self.final_loc / f"{self.namer}.csv")
                proof_spy.to_csv(self.final_loc / f"spy.csv")
                one.to_csv(self.final_loc / f"one_{self.namer}.csv")



        proof, one = section_proof_df()
        ret_lst1 = section_spy_df()
        
        proof_spy = ret_lst1[0]
        divisor = ret_lst1[1]
        total_allocation = ret_lst1[2]
        beat_num = ret_lst1[3]
        proof_2 = ret_lst1[4]
        proof_3 = ret_lst1[5]
        winning_percentage = ret_lst1[6]
        beat_spy_percentage = ret_lst1[7]
        spy_data = ret_lst1[8]

        r202 = section_one_df(proof, one, spy_data)
        one = r202[0]
        gdp = r202[1]
        high_1 = r202[2]
        low_1 = r202[3]

        section_dictate_to_web_app(proof, gdp )

        if self.graphit == True:
            grapher(one)

        file_saver(gdp, proof, proof_spy, one)

        st.caption(f"{'__'*25}\n{'__'*25}")


    def setup(self):
        mcc_0 = pd.read_csv(f"reports/trade_stamped/trade_stamped_mcc_{self.day_0}.csv")
        self.performance(mcc_0)

        print("\n", f">>> {self.day_0} <<<", "\n")
        return