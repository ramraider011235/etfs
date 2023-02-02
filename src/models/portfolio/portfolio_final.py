from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import yfinance as yf
from src.tools.functions import company_longName

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.options.display.float_format = "{:,}".format



class Proof_of_Concept_Final(object):
    
    
    def __init__(self, day_101, interval1='1d', period='1y', save_output=True, graphit=True):
        self.save_output = save_output
        self.graphit = graphit
        self.interval = interval1
        self.per1 = period
        
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
                try:
                    x = company_longName(i)
                    if not x:
                        b.append(i)
                    else:
                        b.append(x)                
                except Exception as e:
                    b.append(x)

            proof["companyName"] = b                  
            start_value = float(proof['investment'].sum())
            final = proof['close_mkt'].sum()
            
            del proof['return']
            proof['allocation'] = round((proof['investment'] / start_value) * 100, 2)
            proof['allocation_fin'] = round((proof['investment'] / final) * 100, 2)
            proof["return"] = round(((proof["close_mkt"] - proof["investment"]) / proof["investment"]) * 100,2,)
            
            day_101 = str(proof['exit'][1])[:10]
            day = int(str(day_101)[8:10])
            month = int(str(day_101)[5:7])
            year = int(str(day_101)[:4])
            self.end1 = datetime(year, month, day) + timedelta(days=1)
            
            df_close = yf.download(self.port_tics, start=self.day_1)["Adj Close"]
            df_close = df_close.fillna(0.0)
                        
            shares = list(proof['shares'])
            one = pd.DataFrame(df_close.copy())
            col_lst = one.columns
            for s in range(len(col_lst)):
                one[col_lst[s]] = one[col_lst[s]] * shares[s] 
            one['portfolio'] = one.sum(axis=1)  
            
            del proof['exit']
            return proof, one


        def section_spy_df():
            spy_data = yf.download("SPY", start=self.day_1, end=self.end1, interval=self.interval)
            proof_spy = pd.DataFrame(["SPY"], columns=["SPY"])
            proof_spy["start_price"] = spy_data["Open"][0]
            proof_spy["close_price"] = spy_data["Adj Close"][-1]
            proof_spy["investment"] = round(proof['investment'].sum(), 2)
            proof_spy["shares"] = round(proof_spy["investment"] / proof_spy["start_price"], 2)
            proof_spy["close_mkt"] = round(proof_spy["shares"] * proof_spy["close_price"], 2)
            proof_spy["return"] = round(((proof_spy["close_mkt"] - proof_spy["investment"])/ proof_spy["investment"])* 100,2,)
            
            
            divisor = len(self.file["ticker"])
            total_allocation = self.file["allocation"].sum() / 100
            beat_num = proof_spy["return"][0]
            proof_2 = proof[proof["return"] > 0.0]
            proof_3 = proof_2[proof_2["return"] > beat_num]
            winning_percentage = round((len(proof_2["ticker"]) / divisor) * 100, 2)
            beat_spy_percentage = round((len(proof_3["ticker"]) / divisor), 2)
            ret_lst = [
                proof_spy,
                divisor,
                total_allocation,
                beat_num,
                proof_2,
                proof_3,
                winning_percentage,
                beat_spy_percentage,
                spy_data
            ]
            return ret_lst


        def section_one_df(proof, one, spy_data):
            port_start = round(proof['investment'].sum(), 2)
            spy_start = round(proof_spy['investment'].sum(), 2)
            port_end = round(proof['close_mkt'].sum(), 2)
            spy_end = round(proof_spy['close_mkt'].sum(), 2)
            port_return = round(((port_end - port_start) / port_start)*100,2)
            spy_return = round(((spy_end - spy_start) / spy_start)*100,2)
            
            spy_low = min(spy_data["Low"])
            spy_high = max(spy_data["High"])
            spy_opener = spy_data["Open"][0]
            
            start_cash = round(proof['investment'].sum(), 2)
            high_1 = round(one["portfolio"].max(), 2)
            low_1 = round(one["portfolio"].min(), 2)            
            high_watermark = round(((high_1 - start_cash) / start_cash) * 100, 2)
            high_watermark_spy = round(((spy_high - spy_opener) / spy_opener) * 100, 2)
            low_watermark = round(((low_1 - start_cash) / start_cash) * 100, 2)
            low_watermark_spy = round(((spy_low - spy_opener) / spy_opener) * 100, 2)
            
            gdp = pd.DataFrame(["Recommended Stocks", "SPY Index"], columns=["strategy_vs_benchmark"])
            gdp["starting_money"] = [f"${port_start}", f"${spy_start}"]
            gdp["ending_money"] = [f"${port_end}", f"${spy_end}"]
            gdp["return"] = [f"{port_return}%", f"{spy_return}%"]
            gdp["high_mark"] = [f"{high_watermark}%",f"{high_watermark_spy}%",]
            gdp["low_mark"] = [f"{low_watermark}%",f"{low_watermark_spy}%",]
            gdp = gdp.set_index("strategy_vs_benchmark")

            for i in list(one["portfolio"]):
                if float(i) > high_1:
                    high_1 = float(i)
                else:
                    pass

            spy_data = yf.download("SPY", start=self.day_1, interval=self.interval)
            spy_data.columns = ["Open", "High", "Low", "Close", "SPY", "Volume"]
            del spy_data["Open"]
            del spy_data["High"]
            del spy_data["Low"]
            del spy_data["Close"]
            del spy_data["Volume"]
            
            start1 = round(one['portfolio'].iloc[0],2)
            shares = round(start1 / spy_data["SPY"].iloc[0], 0)
            spy_data["shares"] = [shares] * len(spy_data)
            spy_data["SPY_Portfolio"] = spy_data["SPY"] * spy_data["shares"]
            one["SPY_Portfolio"] = spy_data["SPY_Portfolio"]

            one["since_open"] = round(
                ((one["portfolio"] - start_cash) / start_cash) * 100, 2
            )
            
            r2 = [one, gdp, high_1, low_1]
            return r2


        def section_dictate_to_web_app(proof, gdp, one):
            watermark_up = gdp['high_mark'].iloc[0]
            watermark_down = gdp['low_mark'].iloc[0]
                        
            st.header(f"__❰◈❱ {self.namer2} vs SPY__")
            st.subheader(f"◾ ↳ Live Portfolio : [{self.day_1}]")
            st.write(f" ╠══► Initial Portfolio Optimization Modeled On {self.day_1}")
            st.write(f" ╠══► High Watermark ⟿ ❰${round(high_1,2)}❱   ❰{watermark_up}❱")
            st.write(f" ╚══► Low Watermark ⟿ ❰${round(low_1,2)}❱   ❰{watermark_down}❱")            

            st.write('__◾ Combined Portfolio vs SPY (S&P 500 Stock Market Index Fund)__')
            st.table(gdp)

            proof = proof.sort_values("return", ascending=False)
            proof["rank"] = proof["return"].rank(ascending=False)
            del proof['rank']
            proof=proof.round(2)
            st.dataframe(proof.set_index("companyName"))

            return


        def grapher(one):
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
                fig.add_scattergl(x=df11.index, y=df11.SPY_Portfolio, line={"color": "crimson"}, name="SPY_Portfolio",)
                fig.update_traces(mode="markers+lines")

                fig.add_trace(go.Scatter(x=df11.index, y=df11.start_line, name="starting_balance", line_shape="hvh", line=dict(color="#7f7f7f", width=4)))
                fig.add_scattergl(x=df11.index, y=df11.portfolio.where(df11.portfolio >= 5500.0), line={"color": "green"}, name="Winning Line")
                fig.add_trace(go.Scatter(x=df11.index, y=df11.win_l, name="Win Threshold", line_shape="hvh", line=dict(color="#2ca02c", width=4)))
                fig.add_scattergl(x=df11.index, y=df11.portfolio.where(df11.portfolio <= 4500.0), line={"color": "red"}, name="Losing Line")
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
                fig.write_image("/home/gdp/i4m/data/images/featurization/fig1.png")
                # st.plotly_chart(fig, use_container_width=False, width=1000, height=500)
                st.plotly_chart(fig, use_container_width=True)
                st.write("_" * 25)
            except Exception:
                pass            

            try:
                df12 = two.copy()
                df12["start_line"] = start1
                df12["win_l"] = win1
                df12["loss_l"] = loss1
                fig = px.area(df12["portfolio"], facet_col_wrap=2)
                fig.update_traces(mode="markers+lines")
                fig.add_trace(
                    go.Scatter(
                        x=df12.index, y=df12.start_line, name="starting_balance", line_shape="hvh", line=dict(color="#7f7f7f", width=4), 
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=df12.index, y=df12.win_l, name="Win Threshold", line_shape="hvh", line=dict(color="#2ca02c", width=4), 
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=df12.index, y=df12.loss_l, name="Loss Threshold", line_shape="hvh", line=dict(color="#d62728", width=4), 
                        )
                    )
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
                pass

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
                pass                

            try:
                df14 = two.copy()
                del df14["portfolio"]
                del df14["since_open"]
                del df14["SPY_Portfolio"]
                start_l = [0.0] * len(df14)
                win_l = [10.0] * len(df14)
                loss_l = [-10.0] * len(df14)
                cum_return = (df14.iloc[-1] - df14.iloc[0]) / df14.iloc[0]
                cum_return * 100
                df_daily_returns = df14.pct_change()
                df_daily_returns = df_daily_returns[1:]
                df_cum_daily_returns = (1 + df_daily_returns).cumprod() - 1
                df_cum_daily_returns = df_cum_daily_returns.reset_index()
                df15 = pd.DataFrame(df_cum_daily_returns.copy()).reset_index()
                try:
                    df15 = df15.set_index("Date")
                except Exception:
                    df15 = df15.set_index("Datetime")
                df15 = df15 * 100
                
                try:
                    del df15['index']
                except Exception:
                    pass

                fig = px.line(df15, x=df15.index, y=df15.columns)
                fig.update_traces(mode="markers+lines")
                fig.add_trace(go.Scatter(x=df15.index, y=start_l, name="starting_balance", line_shape="hvh", line=dict(color="#7f7f7f", width=4), ) )
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
                pass                


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
        
        section_dictate_to_web_app(proof, gdp, one)
        if self.graphit == True:
            grapher(one)
        file_saver(gdp, proof, proof_spy, one)

        

# 【 〔 〕 〖 〗 〘 〙 ❰ ❱
        st.caption(f"{'__'*25}\n{'__'*25}")


    def setup(self):
        mcc_0 = pd.read_csv(f"reports/final/final_mcc_{self.day_0}.csv")
        self.performance(mcc_0)

        print("\n", f">>> {self.day_0} <<<", "\n")
        return