import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.optimize as sco
import streamlit as st
from matplotlib import pyplot as plt

plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 100
pd.options.display.max_rows = 999
pd.get_option("display.max_rows")
np.random.seed(43)


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class The_Portfolio_Optimizer(object):
    
    
    def __init__(self, today_stamp, return_files=True, graphit=False):
        d1 = str(today_stamp)[:10]
        m1 = str(d1)[:7]       
        self.d1 = d1
        self.m1 = str(d1)[:7]
        self.today_stamp = d1
        self.saveMonth = str(d1)[:7]
        self.return_files = return_files
        self.graphit = graphit
        self.write_it = True
        
        self.saveReport = Path(f"reports/portfolio/{m1}/{d1}/")
        if not self.saveReport.exists():
            self.saveReport.mkdir(parents=True)
            
        self.savePortOpt = Path(f"data/images/portfolio_optimizer/{m1}/{d1}/")      
        if not self.savePortOpt.exists():
            self.savePortOpt.mkdir(parents=True)               
            
        self.saveAdvisor = Path(f"data/advisor/{m1}/{d1}/")  
        if not self.saveAdvisor.exists():
            self.saveAdvisor.mkdir(parents=True)        



    def optimize(self, money, num_sims, max_allocations, goods, risk_free_rate):
        st.write('--------------', '\n', '--------------')
        st.header("__❱ Portfolio · Optimization [MPT]:__")
        st.write('-------------')
                    
        self.investment = money
        self.num_portfolios = num_sims
        self.max_allocations = max_allocations
        self.port_tics = list(goods.columns)
        self.port_count = len(self.port_tics)
        self.rfr = risk_free_rate

        PT = pd.DataFrame(goods.iloc[1:])
        PT = PT.loc[~(PT == 0).all(axis=1)]
        tickers = list(PT.columns)
        returns = PT.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()


        def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
            Returns = np.sum(mean_returns * weights) * 252
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return std, Returns


        def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
            results = np.zeros((3, num_portfolios))
            weights_record = []
            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)
                w_lst_record = []
                for w in weights:
                    if w < self.max_allocations:
                        w_lst_record.append(w)
                    else:
                        w_lst_record(self.max_allocations)
                weights_record.append(w_lst_record)
                portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
                results[0, i] = portfolio_std_dev
                results[1, i] = portfolio_return
                results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
            return results, weights_record
        
        
        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
            p_var, p_ret = portfolio_annualised_performance(weights,mean_returns,cov_matrix)
            return -(p_ret - risk_free_rate) / p_var


        def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
            num_assets = len(mean_returns)
            args = (mean_returns, cov_matrix, risk_free_rate)
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            bound = (0.0, 1.0)
            bounds = tuple(bound for asset in range(num_assets))
            result = sco.minimize(neg_sharpe_ratio, num_assets * [1.0 / num_assets,], args=args, method="SLSQP", bounds=bounds, constraints=constraints)
            return result
        
        
        def portfolio_volatility(weights, mean_returns, cov_matrix):
            return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


        def min_variance(mean_returns, cov_matrix):
            num_assets = len(mean_returns)
            args = (mean_returns, cov_matrix)
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            bound = (0.0, 1.0)
            bounds = tuple(bound for asset in range(num_assets))
            result = sco.minimize(portfolio_volatility, num_assets * [1.0 / num_assets,], args=args, method="SLSQP", bounds=bounds, constraints=constraints)
            return result


        def portfolio_return_a(weights):
            return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]


        def efficient_return(mean_returns, cov_matrix, target):
            num_assets = len(mean_returns)
            args = (mean_returns, cov_matrix)
            constraints = ({"type": "eq", "fun": lambda x: portfolio_return_a(x) - target}, {"type": "eq", "fun": lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for asset in range(num_assets))
            result = sco.minimize(portfolio_volatility, num_assets * [1.0 / num_assets,], args=args, method="SLSQP", bounds=bounds, constraints=constraints)
            return result


        def efficient_frontier(mean_returns, cov_matrix, returns_range):
            efficients = []
            for ret in returns_range:
                efficients.append(efficient_return(mean_returns, cov_matrix, ret))
            return efficients        



        def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
            results, weights = random_portfolios(num_portfolios, mean_returns,  cov_matrix,  risk_free_rate)
            
            max_sharpe_idx = np.argmax(results[2])
            sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
            max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=PT.columns,  columns=["allocation"])
            max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
            max_sharpe_allocation = max_sharpe_allocation.T
            temp = pd.DataFrame(max_sharpe_allocation.T).reset_index()
            temp.columns = ["tickers", "allocation"]
            temp = temp.sort_values("allocation", ascending=False)
            temp_allocation = []
            temp_ticker = list(temp["tickers"])
            temp_allocation = list(temp["allocation"])
            max_sharpe_allocation_df = pd.DataFrame(zip(temp_ticker, temp_allocation), columns=["ticker", "allocation"]).sort_values("allocation", ascending=False)
            t_lst = []
            for t in list(max_sharpe_allocation_df["allocation"]):
                if t > self.max_allocations:
                    t_lst.append(self.max_allocations)
                else:
                    t_lst.append(t)
            max_sharpe_allocation_df["allocation"] = t_lst
            rank = []
            [
                rank.append(x)
                for x in range(1, len(max_sharpe_allocation_df["ticker"]) + 1)
            ]
            max_sharpe_allocation_df["rank"] = rank
            max_sharpe_allocation_df = max_sharpe_allocation_df.set_index("rank")
            max_sharpe_allocation_df = max_sharpe_allocation_df[max_sharpe_allocation_df["allocation"] != 0]
                        
            min_vol_idx = np.argmin(results[0])
            sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
            min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=PT.columns,  columns=["allocation"])
            min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
            min_vol_allocation = min_vol_allocation.T
            temp = pd.DataFrame(min_vol_allocation.T).reset_index()
            temp.columns = ["tickers", "allocation"]
            temp = temp.sort_values("allocation", ascending=False)
            temp_allocation = []
            temp_ticker = list(temp["tickers"])
            temp_allocation = list(temp["allocation"])
            min_vol_allocation_df = pd.DataFrame(zip(temp_ticker, temp_allocation), columns=["ticker", "allocation"]).sort_values("allocation", ascending=False)
            t_lst = []
            for t in list(min_vol_allocation_df["allocation"]):
                if t > self.max_allocations:
                    t_lst.append(self.max_allocations)
                else:
                    t_lst.append(t)
            min_vol_allocation_df["allocation"] = t_lst
            rank = []
            [rank.append(x) for x in range(1, len(min_vol_allocation_df["ticker"]) + 1)]
            min_vol_allocation_df["rank"] = rank
            min_vol_allocation_df = min_vol_allocation_df.set_index("rank")
            min_vol_allocation_df = min_vol_allocation_df[min_vol_allocation_df["allocation"] != 0]

            def storage_a():
                st.write("▶ __Maximum Sharpe Ratio Portfolio Allocation__")
                st.write(f"► Total Stocks Allocated: [{len(min_vol_allocation_df.ticker)} / {self.port_count}]")
                st.write(f"► Annualised Return: {round(rp * 100, 2)}%")
                st.write(f"► Annualised Volatility: {round(sdp * 100, 2)}%")
                st.write(f"► Sharpe Ratio : {round((rp / sdp), 2)}%")
                st.dataframe(min_vol_allocation_df)
                st.write("▶ __Minimum Volatility Portfolio Allocation__")
                st.write(f"► Total Stocks Allocated: [{len(min_vol_allocation_df.ticker)} / {self.port_count}]")
                st.write(f"► Annualised Return: {round(rp_min * 100, 2)}%")
                st.write(f"► Annualised Volatility: {round(sdp_min * 100, 2)}%")
                st.write(f"► Sharpe Ratio : {round((rp_min / sdp_min), 2)}%")
                st.dataframe(min_vol_allocation_df)
            if self.write_it == True:
                storage_a()
            
            def graph_a():
                fig, ax = plt.subplots()
                plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap="YlGnBu", marker="o", s=10, alpha=0.3)
                plt.colorbar()
                plt.scatter(sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe ratio")
                plt.scatter(sdp_min,rp_min,marker="*",color="g",s=500,label="Minimum volatility",)
                plt.title("Simulated · Efficient · Frontier · Portfolio · Optimization",fontsize=20,fontweight="bold",)
                plt.xlabel("annualised volatility", fontsize=20, fontweight="bold")
                plt.ylabel("annualised returns", fontsize=20, fontweight="bold")
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontsize(15)
                ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
                ax.legend(loc="best", prop={"size": 16})
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.write("𝄖𝄖𝄖" * 17)
            if self.graphit == True:
                graph_a()

            return (max_sharpe_allocation_df, min_vol_allocation_df)

        
        
        def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
            results, _ = random_portfolios(num_portfolios, mean_returns,  cov_matrix,  risk_free_rate)\
                
            max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
            sdp, rp = portfolio_annualised_performance(max_sharpe["x"], mean_returns, cov_matrix)
            max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index=PT.columns, columns=["allocation"])
            max_sharpe_allocation["allocation"] = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
            t_lst_0 = []
            temp_lst_0 = list(max_sharpe_allocation["allocation"])
            for t in temp_lst_0:
                if t < self.max_allocations:
                    t_lst_0.append(t)
                else:
                    t_lst_0.append(self.max_allocations)
            max_sharpe_allocation["allocation"] = t_lst_0
            max_sharpe_allocation = max_sharpe_allocation.T
            temp = pd.DataFrame(max_sharpe_allocation.T).reset_index()
            temp.columns = ["tickers", "allocation"]
            temp = temp.sort_values("allocation", ascending=False)
            temp_allocation = []
            temp_ticker = list(temp["tickers"])
            temp_allocation = list(temp["allocation"])
            max_sharpe_allocation_df = pd.DataFrame(zip(temp_ticker, temp_allocation), columns=["ticker", "allocation"]).sort_values("allocation", ascending=False)
            t_lst = []
            for t in list(max_sharpe_allocation_df["allocation"]):
                if t > self.max_allocations:
                    t_lst.append(self.max_allocations)
                else:
                    t_lst.append(t)
            max_sharpe_allocation_df["allocation"] = t_lst
            rank = []
            [
                rank.append(x)
                for x in range(1, len(max_sharpe_allocation_df["ticker"]) + 1)
            ]
            max_sharpe_allocation_df["rank"] = rank
            max_sharpe_allocation_df = max_sharpe_allocation_df.set_index("rank")
            max_sharpe_allocation_df = max_sharpe_allocation_df[max_sharpe_allocation_df["allocation"] != 0]            


            min_vol = min_variance(mean_returns, cov_matrix)
            sdp_min, rp_min = portfolio_annualised_performance(min_vol["x"], mean_returns, cov_matrix)
            min_vol_allocation = pd.DataFrame(min_vol.x, index=PT.columns, columns=["allocation"])
            min_vol_allocation["allocation"] = [round(i * 100, 2) for i in min_vol_allocation.allocation]
            t_lst_00 = []
            temp_lst_00 = list(min_vol_allocation["allocation"])
            for t in temp_lst_00:
                if t < self.max_allocations:
                    t_lst_00.append(t)
                else:
                    t_lst_00.append(self.max_allocations)
            min_vol_allocation["allocation"] = t_lst_00
            min_vol_allocation = min_vol_allocation.T
            temp = pd.DataFrame(min_vol_allocation.T).reset_index()
            temp.columns = ["tickers", "allocation"]
            temp = temp.sort_values("allocation", ascending=False)
            temp_allocation = []
            temp_ticker = list(temp["tickers"])
            temp_allocation = list(temp["allocation"])
            min_vol_allocation_df = pd.DataFrame(zip(temp_ticker, temp_allocation), columns=["ticker", "allocation"]).sort_values("allocation", ascending=False)
            t_lst = []
            for t in list(min_vol_allocation_df["allocation"]):
                if t > self.max_allocations:
                    t_lst.append(self.max_allocations)
                else:
                    t_lst.append(t)
            min_vol_allocation_df["allocation"] = t_lst
            rank = []
            [rank.append(x) for x in range(1, len(min_vol_allocation_df["ticker"]) + 1)]
            min_vol_allocation_df["rank"] = rank
            min_vol_allocation_df = min_vol_allocation_df.set_index("rank")
            min_vol_allocation_df = min_vol_allocation_df[min_vol_allocation_df["allocation"] != 0]


            def storage_b():
                st.write("▶ __Maximum Sharpe Ratio Portfolio Allocation__")
                st.write(f"► Total Stocks Allocated: [{len(max_sharpe_allocation_df.ticker)} / {self.port_count}]")
                st.write(f"► Annualised Return: {round(rp *100, 2)}%")
                st.write(f"► Annualised Volatility: {round(sdp * 100, 2)}%")
                st.write(f"► Sharpe Ratio : {round((rp / sdp) * 100, 2)}%")
                st.write(f"► Total Allocation: {round(max_sharpe_allocation_df['allocation'].sum(),2)}")
                st.dataframe(max_sharpe_allocation_df)
                st.write("▶ __Minimum Volatility Portfolio Allocation__")
                st.write(f"► Total Stocks Allocated: [{len(min_vol_allocation_df.ticker)} / {self.port_count}]")
                st.write(f"► Annualised Return: {round(rp_min * 100, 2)}%")
                st.write(f"► Annualised Volatility: {round(sdp_min * 100, 2)}%")
                st.write(f"► Sharpe Ratio : {round((rp_min / sdp_min) * 100, 2)}%")
                st.write(f"► Total Allocation: {round(min_vol_allocation_df['allocation'].sum(),2)}")
                st.dataframe(min_vol_allocation_df)
            if self.write_it == True:
                storage_b()

            def graph_b():
                fig, ax = plt.subplots()
                plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap="YlGnBu", marker="o", s=10, alpha=0.3)
                plt.colorbar()
                plt.scatter(sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe ratio")
                plt.scatter(sdp_min, rp_min, marker="*", color="g", s=500, label="Minimum volatility")
                target = np.linspace(rp_min, 0.32, 50)
                efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
                plt.plot([p["fun"] for p in efficient_portfolios], target, linestyle="-.", color="black", label="efficient frontier")
                plt.title("Calculated Portfolio Optimization based on Efficient Frontier", fontsize=30, fontweight="bold")
                plt.xlabel("annualised volatility", fontsize=20, fontweight="bold")
                plt.ylabel("annualised returns", fontsize=20, fontweight="bold")
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontsize(15)
                ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
                ax.legend(loc="best", prop={"size": 16})
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.write("𝄖𝄖𝄖" * 17)
            if self.graphit == True:
                graph_b()
                
            return (max_sharpe_allocation_df, min_vol_allocation_df)



        def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate):
            max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
            sdp, rp = portfolio_annualised_performance(max_sharpe["x"], mean_returns, cov_matrix)
            max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index=PT.columns, columns=["allocation"])
            max_sharpe_allocation["allocation"] = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
            t_lst_0000 = []
            temp_lst_0000 = list(max_sharpe_allocation["allocation"])
            for t in temp_lst_0000:
                if t < self.max_allocations:
                    t_lst_0000.append(t)
                else:
                    t_lst_0000.append(self.max_allocations)
            max_sharpe_allocation["allocation"] = t_lst_0000
            max_sharpe_allocation = max_sharpe_allocation.T
            
            temp = pd.DataFrame(max_sharpe_allocation.T).reset_index()
            temp.columns = ["tickers", "allocation"]
            temp = temp.sort_values("allocation", ascending=False)
            temp_allocation = []
            temp_ticker = list(temp["tickers"])
            temp_allocation = list(temp["allocation"])
            max_sharpe_allocation_df = pd.DataFrame(zip(temp_ticker, temp_allocation), columns=["ticker", "allocation"]).sort_values("allocation", ascending=False)
            t_lst = []
            for t in list(max_sharpe_allocation_df["allocation"]):
                if t > self.max_allocations:
                    t_lst.append(self.max_allocations)
                else:
                    t_lst.append(t)
            max_sharpe_allocation_df["allocation"] = t_lst
            rank = []
            [
                rank.append(x)
                for x in range(1, len(max_sharpe_allocation_df["ticker"]) + 1)
            ]
            max_sharpe_allocation_df["rank"] = rank
            max_sharpe_allocation_df = max_sharpe_allocation_df.set_index("rank")
            max_sharpe_allocation_df = max_sharpe_allocation_df[max_sharpe_allocation_df["allocation"] != 0]
            max_sharpe_allocation_df = max_sharpe_allocation_df[max_sharpe_allocation_df["allocation"] >= 1.0]
                        
            
            min_vol = min_variance(mean_returns, cov_matrix)
            sdp_min, rp_min = portfolio_annualised_performance(min_vol["x"], mean_returns, cov_matrix)
            min_vol_allocation = pd.DataFrame(min_vol.x, index=PT.columns, columns=["allocation"])
            min_vol_allocation["allocation"] = [round(i * 100, 2) for i in min_vol_allocation.allocation]
            t_lst_000 = []
            temp_lst_000 = list(min_vol_allocation["allocation"])
            for t in temp_lst_000:
                if t < self.max_allocations:
                    t_lst_000.append(t)
                else:
                    t_lst_000.append(self.max_allocations)
            min_vol_allocation["allocation"] = t_lst_000
            min_vol_allocation = min_vol_allocation.T
            
            temp = pd.DataFrame(min_vol_allocation.T).reset_index()
            temp.columns = ["tickers", "allocation"]
            temp = temp.sort_values("allocation", ascending=False)
            temp_allocation = []
            temp_ticker = list(temp["tickers"])
            temp_allocation = list(temp["allocation"])
            min_vol_allocation_df = pd.DataFrame(zip(temp_ticker, temp_allocation), columns=["ticker", "allocation"]).sort_values("allocation", ascending=False)
            t_lst = []
            for t in list(min_vol_allocation_df["allocation"]):
                if t > self.max_allocations:
                    t_lst.append(self.max_allocations)
                else:
                    t_lst.append(t)
            min_vol_allocation_df["allocation"] = t_lst
            rank = []
            [rank.append(x) for x in range(1, len(min_vol_allocation_df["ticker"]) + 1)]
            min_vol_allocation_df["rank"] = rank
            min_vol_allocation_df = min_vol_allocation_df.set_index("rank")
            min_vol_allocation_df = min_vol_allocation_df[min_vol_allocation_df["allocation"] != 0]            
            
            an_vol = np.std(PT.pct_change()) * np.sqrt(252)
            an_rt = mean_returns * 252            


            def storage_c():
                st.write("▶ __Maximum Sharpe Ratio Portfolio Allocation__")
                st.text(f"► Total Stocks Allocated: [{len(max_sharpe_allocation_df.ticker)} / {self.port_count}]")
                st.text(f"► Annualised Return: {round(rp * 100, 2)}%")
                st.text(f"► Annualised Volatility: {round(sdp * 100, 2)}%")
                st.text(f"► Sharpe Ratio : {round((rp / sdp) * 100, 2)}%")
                st.text(f"► Total Allocation: {round(max_sharpe_allocation_df['allocation'].sum(),2)}")
                st.dataframe(max_sharpe_allocation_df)

                st.write("▶ __Minimum Volatility Portfolio Allocation__")
                st.text(f"► Total Stocks Allocated: [{len(min_vol_allocation_df.ticker)} / {self.port_count}]")
                st.text(f"► Annualised Return: {round(rp_min * 100, 2)}%")
                st.text(f"► Annualised Volatility: {round(sdp_min * 100, 2)}%")
                st.text(f"► Sharpe Ratio : {round((rp_min / sdp_min) * 100, 2)}%")
                st.text(f"► Total Allocation: {round(min_vol_allocation_df['allocation'].sum(),2)}")
                st.dataframe(min_vol_allocation_df)
                
                print("▶ __Maximum Sharpe Ratio Portfolio Allocation__")
                print(f"► Total Stocks Allocated: [{len(max_sharpe_allocation_df.ticker)} / {self.port_count}]")
                print(f"► Annualised Return: {round(rp * 100, 2)}%")
                print(f"► Annualised Volatility: {round(sdp * 100, 2)}%")
                print(f"► Sharpe Ratio : {round((rp / sdp) * 100, 2)}%")
                print(f"► Total Allocation: {round(max_sharpe_allocation_df['allocation'].sum(),2)}")
                print(max_sharpe_allocation_df)

                print("▶ __Minimum Volatility Portfolio Allocation__")
                print(f"► Total Stocks Allocated: [{len(min_vol_allocation_df.ticker)} / {self.port_count}]")
                print(f"► Annualised Return: {round(rp_min * 100, 2)}%")
                print(f"► Annualised Volatility: {round(sdp_min * 100, 2)}%")
                print(f"► Sharpe Ratio : {round((rp_min / sdp_min) * 100, 2)}%")
                print(f"► Total Allocation: {round(min_vol_allocation_df['allocation'].sum(),2)}")
                print(min_vol_allocation_df)                
            storage_c()

            def graph_c():
                st.subheader("◾ · Individual Stock Returns and Volatility\n")
                for i, txt in enumerate(PT.columns):
                    st.text(f"{txt}: annuaised return {round(an_rt[i],2)} - annualised volatility: {round(an_vol[i],2)}")
                                    
                fig, ax = plt.subplots()
                ax.scatter(an_vol, an_rt, marker="o", s=200)
                for i, txt in enumerate(PT.columns):
                    ax.annotate(txt, (an_vol[i], an_rt[i]), xytext=(10, 0), textcoords="offset points")
                ax.scatter(sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe ratio")
                ax.scatter(sdp_min, rp_min, marker="*", color="g", s=500, label="Minimum volatility")
                target = np.linspace(rp_min, 0.34, 50)
                efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
                ax.plot([p["fun"] for p in efficient_portfolios], target, linestyle="-.", color="black", label="efficient frontier")
                ax.set_title("Portfolio Optimization with Individual Stocks", fontsize=30, fontweight="bold")
                ax.set_xlabel("annualised volatility", fontsize=20, fontweight="bold")
                ax.set_ylabel("annualised returns", fontsize=20, fontweight="bold")
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontsize(15)
                ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
                ax.legend(loc="best", prop={"size": 16})
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            if self.graphit == True:
                graph_c()
                
            return (max_sharpe_allocation_df, min_vol_allocation_df)



        st.subheader(f"◾ · Optimal Efficient Frontier ~ [Method · 1]")
        st.write("◻ Random Number Wt/Position Simulation")        
        (max_sharpe_df_1, min_vol_df_1) = display_simulated_ef_with_random(mean_returns, cov_matrix, self.num_portfolios, self.rfr)


        st.subheader("◾ · Optimal Efficient Frontier ~ [Method · 2]")
        st.write("◻ A Random Number Of Portfolios & Position Weights")
        (max_sharpe_df_2, min_vol_df_2) = display_calculated_ef_with_random(mean_returns, cov_matrix, self.num_portfolios, self.rfr)


        st.subheader("◾ · Optimal Efficient Frontier ~ [Method · 3]")
        st.write("◻ Calculated Efficient Frontier With Selected Position Weights")
        print("◾ · Optimal Efficient Frontier ~ [Method · 3]")
        print("◻ Calculated Efficient Frontier With Selected Position Weights")        
        (max_sharpe_df_3, min_vol_df_3) = display_ef_with_selected(mean_returns, cov_matrix, self.rfr)


        saver_lst_1 = [max_sharpe_df_3, min_vol_df_3]           # max_sharpe_df_1, min_vol_df_1, roll_out_list_a, max_sharpe_df_2, min_vol_df_2,
        namer_lst_1 = ["max_sharpe_df_3", "min_vol_df_3"]   # "max_sharpe_df_1", "min_vol_df_1", "roll_out_list_a", "max_sharpe_df_2", "min_vol_df_2",

        if self.return_files == True:
            for r in range(len(saver_lst_1)):
                fd = pd.DataFrame(saver_lst_1[r])
                fd.to_pickle(f"reports/portfolio/{self.m1}/{self.d1}/{namer_lst_1[r]}.pkl")
        return max_sharpe_df_3, min_vol_df_3
