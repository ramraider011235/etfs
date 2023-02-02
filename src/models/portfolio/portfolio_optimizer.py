import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.util.remote import Remote

from pymoo.core.problem import Problem
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoode.algorithms import GDE3
from pymoode.survival import RankAndCrowding

from yahooquery import Ticker
import datetime as dt
import pandas as pd
from os.path import exists
from pathlib import Path
import pickle5 as pickle
from dateutil.relativedelta import relativedelta
import numpy as np
np.random.seed(42)




class Portfolio_Optimizer_2(object):
    
    
    def __init__(self, time_date, rfr, num_sims, data, max_allocations, graphit=True):
        self.graphit = graphit
        self.max_allocations = max_allocations
        self.data = data
        self.num_sims = num_sims
        self.rfr = rfr
        self.cherry_pick = time_date
        start_date_101 = dt.date(int(str(time_date)[:4]), int(str(time_date)[5:7]), int(str(time_date)[8:]))
        self.years_ago = str(start_date_101 - relativedelta(years=1, days=0))[:10]    
        self.tickers0 = list(data.columns)
    
    
    def run_mod(self, N=252):
        st.write('------', '\n', '------')
        st.header("__❱ Portfolio · Optimization [EF]:__")
        st.write('-------------')
                
        dataset = self.data.copy()

        #Daily log returns assumed to follow a normal distribution
        log_returns = (dataset / dataset.shift(1)).apply(np.log).dropna()
        # mean_log_returns = (log_returns).mean(axis=0) * N
        
        #Covariance matrix of sum of N normally distributed elements
        cov_log_matrix = log_returns.cov() * N

        #Simple returns
        simple_returns = (dataset / dataset.shift(1) - 1.0).dropna()
        mean_simple_returns = simple_returns.mean(axis=0)
        cov_simple_matrix = simple_returns.cov()

        #Obtain variance of individual assets
        individual_var = pd.Series({key: cov_simple_matrix.loc[key, key] for key in cov_log_matrix.index})
        individual_sigma = individual_var.apply(np.sqrt)    
        
        
        if self.graphit == True:
            fig, ax = plt.subplots(figsize=[6, 5], dpi=100)
            ax.scatter(100 * individual_sigma, 100 * mean_simple_returns, color="navy", label="Individual Assets")
            ax.set_xlabel(r"Mean daily $\sigma$ [%]")
            ax.set_ylabel("Mean daily returns [%]")
            fig.tight_layout()
            plt.show()        
            
            
        #Declare mu and sigma
        mu = mean_simple_returns.values
        sigma = cov_simple_matrix.values        
        
        
        class PortfolioProblem(Problem):
            def __init__(self, mu, sigma, risk_free_rate=(self.rfr/N), **kwargs):
                super().__init__(n_var=mu.shape[0], n_obj=2, xl=0.0, xu=1.0, n_ieq_constr=1, **kwargs)
                self.mu = mu
                self.sigma = sigma
                self.risk_free_rate = risk_free_rate

            def _evaluate(self, X, out, *args, **kwargs):
                #Remember each line in X corresponds to an individual, with w in the columns
                exp_return = X.dot(self.mu).reshape([-1, 1])
                exp_sigma = np.sqrt(np.sum(X * X.dot(self.sigma), axis=1, keepdims=True))
                sharpe = (exp_return - self.risk_free_rate) / exp_sigma
                out["F"] = np.column_stack([exp_sigma, -exp_return])
                out["G"] = np.sum(X, axis=1, keepdims=True) - 1
                out["sharpe"] = sharpe
                
                
        class Normalizer(Mutation):
            def _do(self, problem, X, **kwargs):
                X = X / np.sum(X, axis=1, keepdims=True)
                return X


        class PortfolioRepair(Repair):
            def _do(self, problem, X, **kwargs):
                X[X < 1e-3] = 0
                return X / X.sum(axis=1, keepdims=True)
            
            
        problem = PortfolioProblem(mu, sigma, 0.02/252)
        normalizer = Normalizer()

        np.random.seed(12)
        X0 = np.random.rand(100, len(mu))
        X0 = normalizer._do(problem, X0)

        res = minimize(
            problem,
            GDE3(
                100, 
                CR=0.9,
                survival=RankAndCrowding(crowding_func="pcd"),
                pm=normalizer,
                sampling=X0
            ),
            ("n_gen", 250),
            seed=12
        )                
        res_sms = minimize(
            problem,
            SMSEMOA(
                100,
                repair=PortfolioRepair(),
                sampling=X0
            ),
            ("n_gen", 250),
            seed=12
        )          
        res = res_sms 
        
        
        def evaluate_as_function(X, mu, sigma):
            risk_free_allocation = 1 - X.sum(axis=1)
            exp_return = X.dot(mu).reshape([-1, 1])
            exp_sigma = np.sqrt(np.sum(X * X.dot(sigma), axis=1, keepdims=True))
            return np.column_stack([exp_sigma, exp_return])



        X_mc = np.random.rand(self.num_sims, len(mu))
        X_mc = X_mc / np.sum(X_mc, axis=1, keepdims=True)
        F_mc = evaluate_as_function(X_mc, mu, sigma)
        argmax_sharpe = res.opt.get("sharpe").argmax()
        best_sharpe = res.opt.get("sharpe").max()   
        
        
        if self.graphit == True:
            fig, ax = plt.subplots(figsize=[6, 5], dpi=100)
            fig.patch.set_facecolor('white')
            ax.scatter(res.F[:, 0] * 100, -res.F[:, 1] * 100, color="navy", label="GDE3")
            ax.scatter(100 * individual_sigma, 100 * mean_simple_returns, color="firebrick", label="Individual Assets", zorder=2)
            ax.scatter(F_mc[:, 0] * 100, F_mc[:, 1] * 100, color="grey", alpha=0.2, label="Monte Carlo", zorder=0)
            ax.scatter(res.F[argmax_sharpe, 0] * 100, -res.F[argmax_sharpe, 1] * 100, color="darkgoldenrod", label="Best Sharpe", zorder=4)
            ax.plot([0, 100 * res.F[:, 0].max()],
                    [100 * problem.risk_free_rate, 100 * problem.risk_free_rate + 100 * best_sharpe * res.F[:, 0].max()],
                    color="black", linestyle="--", zorder=3, label="Risk-Free Tangency Line")
            ax.set_xlim([0, 7.0])
            ax.set_ylim([-0.5, None])
            ax.legend()
            ax.set_xlabel(r"Mean daily $\sigma$ [%]")
            ax.set_ylabel("Mean daily returns [%]")
            fig.tight_layout()
            plt.show()             
        
        
        argmax_sharpe = res.opt.get("sharpe").argmax()
        best_sharpe = res.opt.get("sharpe").max()          
        
        if self.graphit == True:        
            fig, ax = plt.subplots(figsize=[6, 5], dpi=100)
            fig.patch.set_facecolor('white')
            ax.scatter(res_sms.F[:, 0] * 100, -res_sms.F[:, 1] * 100, color="navy", label="SMS-EMOA")
            ax.scatter(100 * individual_sigma, 100 * mean_simple_returns, color="firebrick", label="Individual Assets", zorder=2)
            ax.scatter(F_mc[:, 0] * 100, F_mc[:, 1] * 100, color="grey", alpha=0.2, label="Monte Carlo", zorder=0)
            ax.scatter(res.F[argmax_sharpe, 0] * 100, -res.F[argmax_sharpe, 1] * 100, color="darkgoldenrod", label="Best Sharpe", zorder=4)
            ax.plot([0, 100 * res.F[:, 0].max()],
                    [100 * problem.risk_free_rate, 100 * problem.risk_free_rate + 100 * best_sharpe * res.F[:, 0].max()],
                    color="black", linestyle="--", zorder=3, label="Risk-Free Tangency Line")
            ax.set_xlim([0, 7.0])
            ax.set_ylim([-0.5, None])
            ax.legend()
            ax.set_xlabel(r"Mean daily $\sigma$ [%]")
            ax.set_ylabel("Mean daily returns [%]")
            fig.tight_layout()
            plt.show()                  
                    
                    
        tickers = np.array(dataset.columns)
        
        
        allocation1 = res.X[argmax_sharpe, :]
        order1 = np.flip(np.argsort(allocation1))
        best_sharpe_df = pd.DataFrame.from_dict(dict(zip(tickers[order1], allocation1[order1])), orient='index').reset_index()
        best_sharpe_df.columns = ['ticker', 'allocation']
        best_sharpe_df['allocation'] = best_sharpe_df['allocation'] * 100
        best_sharpe_df = best_sharpe_df.round(2)
        best_sharpe_df = best_sharpe_df[best_sharpe_df['allocation'] >= 1.0]              
        st.subheader("__Best sharpe:__")
        st.write(f"◾ Total Allocation:__【{round(best_sharpe_df['allocation'].sum(), 2)}%】__")
        # 【
        # 】
        # for j in order1:
        #     _t = tickers[j]
        #     _a = allocation1[j] * 100
        #     if _a > 0.9:
        #         st.write(f"{_t}: {_a:.2f}%")        
        st.dataframe(best_sharpe_df)
        
        
        lowest_variance = np.argmin(res.F[:, 0])
        allocation2 = res.X[lowest_variance, :]
        order2 = np.flip(np.argsort(allocation2))
        lowest_variance_df = pd.DataFrame.from_dict(dict(zip(tickers[order2], allocation2[order2])), orient='index').reset_index()
        lowest_variance_df.columns = ['ticker', 'allocation']
        lowest_variance_df['allocation'] = lowest_variance_df['allocation'] * 100
        lowest_variance_df = lowest_variance_df.round(2)
        lowest_variance_df = lowest_variance_df[lowest_variance_df['allocation'] >= 1.0]        
        st.subheader("__Lowest variance:__")
        st.write(f"◾ Total Allocation:__【{round(lowest_variance_df['allocation'].sum(), 2)}%】__")
        # for j in order2:
        #     _t = tickers[j]
        #     _a = allocation2[j] * 100
        #     if _a > 0.9:
        #         st.write(f"{_t}: {_a:.2f}%")        
        st.dataframe(lowest_variance_df)
        
        
        best_returns = np.argmin(res.F[:, 1])
        allocation3 = res.X[best_returns, :]
        order3 = np.flip(np.argsort(allocation3))
        best_returns_df = pd.DataFrame.from_dict(dict(zip(tickers[order3], allocation3[order3])), orient='index').reset_index()
        best_returns_df.columns = ['ticker', 'allocation']
        best_returns_df['allocation'] = best_returns_df['allocation'] * 100
        best_returns_df = best_returns_df.round(2)
        best_returns_df = best_returns_df[best_returns_df['allocation'] >= 1.0]    
        st.subheader("__Best returns:__")
        st.write(f"◾ Total Allocation:__【{round(best_returns_df['allocation'].sum(), 2)}%】__")
        # for j in order3:
        #     _t = tickers[j]
        #     _a = allocation3[j] * 100
        #     if _a > 0.9:
        #         st.write(f"{_t}: {_a:.2f}%")        
        st.dataframe(best_returns_df)
        
        
        t_lst_00 = []
        temp_lst_00 = list(best_sharpe_df["allocation"])
        for t in temp_lst_00:
            if t < self.max_allocations:
                t_lst_00.append(t)
            else:
                t_lst_00.append(self.max_allocations)
        best_sharpe_df["allocation"] = t_lst_00    
        
        
        t_lst_01 = []
        temp_lst_01 = list(lowest_variance_df["allocation"])
        for t in temp_lst_01:
            if t < self.max_allocations:
                t_lst_01.append(t)
            else:
                t_lst_01.append(self.max_allocations)
        lowest_variance_df["allocation"] = t_lst_01    
        
        
        # t_lst_02 = []
        # temp_lst_02 = list(best_returns_df["allocation"])
        # for t in temp_lst_02:
        #     if t < self.max_allocations:
        #         t_lst_02.append(t)
        #     else:
        #         t_lst_02.append(self.max_allocations)
        # best_returns_df["allocation"] = t_lst_02                        
        
        
        return best_sharpe_df, lowest_variance_df, best_returns_df
        
        
        
        
        
        
# if __name__ == '__main__':
#     day1 = input("date:")
#     month1 = str(day1)[:7]
#     year1 = str(day1)[:4]
#     # bulk_data_file1 = (f"/home/gdp/hot_box/i4m/data/recommenders/{year1}/{month1}/{day1}/recommender_05_return_dataFrame.pkl")
#     # data0 = pd.read_pickle(bulk_data_file1)    
#     tickers = ['AEHR', 'ROCC', 'SM', 'ARRY', 'NOG', 'MTDR', 'APA', 'AXSM', 'PHX', 'TMDX', 'ENPH', 'BTU', 'LBRT', 'ERF', 'CIVI', 'PI', 'CALX', 'CHRD', 'PARR', 'FTI', 'SGML', 'FANG']
    
#     best_sharpe_df, lowest_variance_df, best_returns_df = Portfolio_Optimizer_2(day1, tickers, False).run_mod()