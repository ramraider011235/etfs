import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

np.random.seed(42)
plt.style.use("ggplot")
path = Path.cwd()

import src.tools.functions as f0



class The_PCA_Analysis(object):
    
    
    def __init__(self, report_date, save_final=True):
        self.report_date = report_date
        self.save_final = save_final

        self.saveAdvisor = Path(f"data/advisor/pca/{str(self.report_date)[:7]}/{self.report_date}/")
        if not self.saveAdvisor.exists():
            self.saveAdvisor.mkdir(parents=True)
            

    def build_pca(self, data, x_factor, graph_it=True):
        st.write('------', '\n', '------')
        st.header("__❱ Principal · Component · Analysis:__")
        st.write('-------------')
        
        self.tickers = list(data.columns)
        self.graph_it = graph_it
        self.prices = pd.DataFrame(data).dropna()
        self.rs = pd.DataFrame(self.prices.apply(np.log).diff(1)).iloc[1:]
        self.x_factor = int(round(float(len(self.tickers) * x_factor)))
        self.x_factor_3rds = int(round(float(len(self.tickers) * 0.33)))
        self.x_factor_qtrs = int(round(float(len(self.tickers) * 0.25)))

        
        if self.graph_it:
            try:
                fig, ax = plt.subplots()
                ax = self.rs.plot(legend=0,grid=True,title=f"Daily Returns",)
                plt.tight_layout()
                st.pyplot()
                plt.close(fig)
            except Exception:
                pass
        
        if self.graph_it:
            try:
                fig, ax = plt.subplots()
                abc = (self .rs.cumsum().apply(np.exp))
                abc.plot(legend=0, grid=True, title=f"Cumulative Returns",)
                plt.tight_layout()
                st.pyplot()
                plt.close(fig)
            except Exception:
                pass


        num1 = 1
        pca = PCA(n_components=num1, svd_solver='full').fit(self.rs.fillna(0.0))
        pc1 = pd.Series(index=self.rs.columns, data=pca.components_[0])       

        # pc1.to_csv('/home/gdp/i4m/pca1.csv')
        # fig, ax = plt.subplots()
        # pc1.plot(
        #     xticks=[],
        #     grid=True,
        #     title=f"First Principal Component",
        # )
        # plt.tight_layout()
        # st.pyplot(fig)


        weights = abs(pc1) / sum(abs(pc1))
        myrs = (weights * self.rs).sum(1)

        # if self.graph_it:
        # myrs_cumsum = myrs.cumsum().apply(np.exp)
        #     fig, ax = plt.subplots()
        #     myrs_cumsum.plot(grid=True,title=f"Cumulative Daily Returns of 1st Principal Component Stock",)
        #     st.pyplot()


        try:
            self.prices = yf.download(["^GSPC"], period="1y")["Adj Close"]
        except Exception:
            self.prices = yf.download(["SPY"], period="1y")["Adj Close"]
            
        rs_df = pd.concat([myrs, self.prices.apply(np.log).diff(1)], 1)
        rs_df.columns = ["PCA Portfolio", "SP500_Index"]
        
        # if self.graph_it:
        #     fig, ax = plt.subplots()
        #     rs_df.dropna().cumsum().apply(np.exp).plot(subplots=True, grid=True, linewidth=3)
        #     plt.tight_layout()
        #     st.pyplot()

        # if self.graph_it:
            # fig, ax = plt.subplots(2, 1)
            # pc1.nlargest(10).plot.bar(
            #     ax=ax[0],
            #     color="blue",
            #     grid=True,
            #     title="Stocks with Highest PCA Score (-OR- Least Negative) PCA Weights",
            #     )
            # pc1.nsmallest(10).plot.bar(
            #     ax=ax[1],
            #     color="green",
            #     grid=True,
            #     title="Stocks with Lowest PCA Score (-OR- Most Negative) PCA Weights",
            #     )
            # plt.tight_layout()
            # st.pyplot()

        
        myrs = self.rs[pc1.nlargest(self.x_factor).index].mean(1)
        myrs1 = myrs.cumsum().apply(np.exp)
        largest_ret = myrs1.iloc[-1]

        # if self.graph_it:
        #     fig, ax = plt.subplots()
        #     p101 = self.prices[: self.report_date].apply(np.log).diff(1).cumsum().apply(np.exp)
        #     myrs1.plot(
        #         grid=True,
        #         linewidth=3,
        #         title=f"PCA Selection ({self.x_factor} Most Impactful) vs S&P500 Index",
        #         )
        #     p101.plot(grid=True, linewidth=3)
        #     plt.legend(["PCA Selection", "SP500_Index"])
        #     plt.tight_layout()
        #     st.pyplot()


        myrs = self.rs[pc1.nsmallest(self.x_factor).index].mean(1)
        myrs2 = myrs.cumsum().apply(np.exp)
        smallest_ret = myrs2.iloc[-1]
        
        # if self.graph_it:
        #     p102 = self.prices[: self.report_date].apply(np.log).diff(1).cumsum().apply(np.exp)        
        #     fig, ax = plt.subplots()
        #     myrs2.plot(
        #         grid=True,
        #         linewidth=3,
        #         title=f"PCA Selection ({self.x_factor} Least Impactful) vs S&P500 Index",
        #     )
        #     p102.plot(grid=True, linewidth=3)
        #     plt.legend(["PCA Selection", "SP500_Index"])
        #     plt.tight_layout()
        #     st.pyplot()

     
        spy500_ret = (self.prices["2021":].apply(np.log).diff(1).cumsum().apply(np.exp).iloc[-1])

        # if len(self.tickers) > 10:
        #     ws = [-1,] * 5 + [1,] * 5
        #     myrs = (self.rs[list(pc1.nsmallest(5).index) + list(pc1.nlargest(5).index)] * ws).mean(1)
        #     fig, ax = plt.subplots()
        #     myrs.cumsum().apply(np.exp).plot(
        #         grid=True,
        #         linewidth=3,
        #         title=f"PCA Portfolio (5 Most & 5 Least Impactful) vs The Round 5 Stocks",
        #     )
        #     self.prices["2020":].apply(np.log).diff(1).cumsum().apply(np.exp).plot(grid=True, linewidth=3)
        #     plt.legend(["PCA Selection", "SP500_Index"])
        #     plt.tight_layout()
        #     st.pyplot(fig)

        #     st.write(f"◾ __Principal Components From Ticker List__")
        #     st.write(f"◻ LARGEST PCA VALUES == [{round(largest_ret,2)}]")
        #     st.write(f"◻ SMALLEST PCA VALUES == [{round(smallest_ret,2)}]")
        #     st.write(f"◻ SPY500 VALUES == [{round(spy500_ret,2)}]")
        #     if largest_ret > smallest_ret:
        #         return self.rs[pc1.nlargest(self.x_factor).index]
        #     else:
        #         return self.rs[pc1.nsmallest(self.x_factor).index]



        # myrs_top = self.rs[pc1.nlargest(self.x_factor_3rds).index].mean(1)
        # myrs_top2 = myrs_top.cumsum().apply(np.exp)
        # top_3rd_ret = myrs_top2.iloc[-1]


        # myrs_middle = self.rs[pc1.nlargest(int(self.x_factor_3rds * 2)).index].mean(1)
        # myrs_middle2 = myrs_middle.cumsum().apply(np.exp)
        # middle_3rd_ret = myrs_middle2.iloc[-1]

        # myrs_bottom = self.rs[pc1.nsmallest(self.x_factor_3rds).index].mean(1)
        # myrs_bottom2 = myrs_bottom.cumsum().apply(np.exp)
        # bottom_3rd_ret = myrs_bottom2.iloc[-1]                      


        largest_ret_lst = self.rs[pc1.nlargest(self.x_factor).index]
        # top_3rd_ret_lst = self.rs[pc1.nlargest(self.x_factor_3rds).index]
        # middle_3rd_ret_lst = self.rs[pc1.nlargest(self.x_factor_3rds).index]
        # bottom_3rd_ret_lst = self.rs[pc1.nlargest(self.x_factor_3rds).index]
        smallest_ret_lst = self.rs[pc1.nsmallest(self.x_factor).index]    

        # st.subheader(f"◾ __Principal Components From Ticker List__")

        # st.write(f"__◾ LARGEST PCA VALUES = 【{round(largest_ret,2)}】-【{len(largest_ret_lst.columns)}】__")
        # st.text(f"◾ {list(largest_ret_lst.columns)}")

        # st.write(f"__◾ Top 3rd PCA VALUES = 【{round(top_3rd_ret,2)}】-【{len(top_3rd_ret_lst.columns)}】__")
        # st.text(f"◾ {list(top_3rd_ret_lst.columns)}")

        # st.write(f"__◾ Middle 3rd PCA VALUES = 【{round(middle_3rd_ret,2)}】-【{len(middle_3rd_ret_lst.columns)}】__")
        # st.text(f"◾ {list(middle_3rd_ret_lst.columns)}")

        # st.write(f"__◾ Bottom 3rd PCA VALUES = 【{round(bottom_3rd_ret,2)}】-【{len(bottom_3rd_ret_lst.columns)}】__")        
        # st.text(f"◾ {list(bottom_3rd_ret_lst.columns)}")

        # st.write(f"__◾ SMALLEST PCA VALUES = 【{round(smallest_ret,2)}】-【{len(smallest_ret_lst.columns)}】__")
        # st.text(f"◾ {list(smallest_ret_lst.columns)}")

        # st.write(f"__◾ SPY500 VALUES = 【{round(spy500_ret,2)}】__")        


        # if (largest_ret > smallest_ret) and (largest_ret > top_3rd_ret) and (largest_ret > middle_3rd_ret) and (largest_ret > bottom_3rd_ret):
        #     return largest_ret_lst

        # elif (smallest_ret > largest_ret) and (smallest_ret > top_3rd_ret) and (smallest_ret > middle_3rd_ret) and (smallest_ret > bottom_3rd_ret):
        #     return smallest_ret_lst          

        # elif (top_3rd_ret > smallest_ret) and (top_3rd_ret > largest_ret) and (top_3rd_ret > middle_3rd_ret) and (top_3rd_ret > bottom_3rd_ret):
        #     return top_3rd_ret_lst

        # elif (middle_3rd_ret > smallest_ret) and (middle_3rd_ret > top_3rd_ret) and (middle_3rd_ret > largest_ret) and (middle_3rd_ret > bottom_3rd_ret):
        #     return middle_3rd_ret_lst

        # elif (bottom_3rd_ret > smallest_ret) and (bottom_3rd_ret > top_3rd_ret) and (bottom_3rd_ret > middle_3rd_ret) and (bottom_3rd_ret > largest_ret):
        #     return bottom_3rd_ret_lst

        # else:
        #     return smallest_ret_lst        


        # returns = self.rs[pc1.nlargest(len(self.tickers)).index].dropna()
        # df_cum_daily_returns = (1 + returns).cumprod() - 1
        # df_cum_daily_returns = df_cum_daily_returns.dropna().reset_index()
        # cum_return_entire_period = df_cum_daily_returns.iloc[:, 1:].tail(1)
        # fd = pd.DataFrame(cum_return_entire_period * 100).round(2).T.reset_index()
        # fd.columns = ['ticker', 'cum_return']

        # quantile_scores = fd.quantile([0.2, 0.4, 0.6, 0.8])

        # quantile_0_20_df = fd[fd['cum_return'] < quantile_scores['cum_return'].iloc[0]]
        # quantile_20_40_df = fd[(fd['cum_return'] > quantile_scores['cum_return'].iloc[0]) & (fd['cum_return'] < quantile_scores.cum_return.iloc[1])]
        # quantile_40_60_df = fd[(fd['cum_return'] > quantile_scores['cum_return'].iloc[1]) & (fd['cum_return'] < quantile_scores.cum_return.iloc[2])]
        # quantile_60_80_df = fd[(fd['cum_return'] > quantile_scores['cum_return'].iloc[2]) & (fd['cum_return'] < quantile_scores.cum_return.iloc[3])]
        # quantile_80_100_df = fd[fd['cum_return'] > quantile_scores['cum_return'].iloc[-1]]

        # quantile_0_20_tickers = list(quantile_0_20_df['ticker'])
        # quantile_20_40_tickers = list(quantile_20_40_df['ticker'])
        # quantile_40_60_tickers = list(quantile_40_60_df['ticker'])
        # quantile_60_80_tickers = list(quantile_60_80_df['ticker'])
        # quantile_80_100_tickers = list(quantile_80_100_df['ticker'])

        # quantile_0_20_df = quantile_0_20_df.set_index('ticker')
        # quantile_20_40_df = quantile_20_40_df.set_index('ticker')
        # quantile_40_60_df = quantile_40_60_df.set_index('ticker')
        # quantile_60_80_df = quantile_60_80_df.set_index('ticker')
        # quantile_80_100_df = quantile_80_100_df.set_index('ticker')        

        # quantile_0_20_ret = quantile_0_20_df.mean(1).cumsum().apply(np.exp).iloc[-1]
        # quantile_20_40_ret = quantile_20_40_df.mean(1).cumsum().apply(np.exp).iloc[-1]
        # quantile_40_60_ret = quantile_40_60_df.mean(1).cumsum().apply(np.exp).iloc[-1]
        # quantile_60_80_ret = quantile_60_80_df.mean(1).cumsum().apply(np.exp).iloc[-1]
        # quantile_80_100_ret = quantile_80_100_df.mean(1).cumsum().apply(np.exp).iloc[-1]


        # st.subheader(f"◾ __Principal Components From Ticker List__")

        # st.write(f"__◾ LARGEST PCA VALUES = 【{round(largest_ret,2)}】-【{len(largest_ret_lst.columns)}】__")
        # st.text(f"◾ {list(largest_ret_lst.columns)}")

        # st.write(f"__◾ quantile_0_20_ret PCA VALUES = 【{round(quantile_0_20_ret,2)}】-【{len(quantile_0_20_tickers)}】__")
        # st.text(f"◾ {list(quantile_0_20_tickers)}")

        # st.write(f"__◾ quantile_20_40_ret PCA VALUES = 【{round(quantile_20_40_ret,2)}】-【{len(quantile_20_40_tickers)}】__")
        # st.text(f"◾ {list(quantile_20_40_tickers)}")

        # st.write(f"__◾ quantile_40_60_ret PCA VALUES = 【{round(quantile_40_60_ret,2)}】-【{len(quantile_40_60_tickers)}】__")
        # st.text(f"◾ {list(quantile_40_60_tickers)}")        

        # st.write(f"__◾ quantile_60_80_ret PCA VALUES = 【{round(quantile_60_80_ret,2)}】-【{len(quantile_60_80_tickers)}】__")        
        # st.text(f"◾ {list(quantile_60_80_tickers)}")

        # st.write(f"__◾ quantile_80_100_ret PCA VALUES = 【{round(quantile_80_100_ret,2)}】-【{len(quantile_80_100_tickers)}】__")        
        # st.text(f"◾ {list(quantile_80_100_tickers)}")        

        # st.write(f"__◾ SMALLEST PCA VALUES = 【{round(smallest_ret,2)}】-【{len(smallest_ret_lst.columns)}】__")
        # st.text(f"◾ {list(smallest_ret_lst)}")

        # st.write(f"__◾ SPY500 VALUES = 【{round(spy500_ret,2)}】__")


        # if (largest_ret > smallest_ret) and (largest_ret > quantile_0_20_ret) and (largest_ret > quantile_20_40_ret) and (largest_ret > quantile_40_60_ret) and (largest_ret > quantile_60_80_ret) and (largest_ret > quantile_80_100_ret):
        #     return largest_ret_lst

        # elif (smallest_ret > largest_ret) and (smallest_ret > quantile_0_20_ret) and (smallest_ret > quantile_20_40_ret) and (smallest_ret > quantile_40_60_ret) and (smallest_ret > quantile_60_80_ret) and (smallest_ret > quantile_80_100_ret):
        #     return smallest_ret_lst          

        # elif (quantile_0_20_ret > smallest_ret) and (quantile_0_20_ret > largest_ret) and (quantile_0_20_ret > quantile_20_40_ret) and (quantile_0_20_ret > quantile_40_60_ret) and (quantile_0_20_ret > quantile_60_80_ret) and (quantile_0_20_ret > quantile_80_100_ret):
        #     return quantile_0_20_df

        # elif (quantile_20_40_ret > smallest_ret) and (quantile_20_40_ret > quantile_0_20_ret) and (quantile_20_40_ret > largest_ret) and (quantile_20_40_ret > quantile_40_60_ret) and (quantile_20_40_ret > quantile_60_80_ret) and (quantile_20_40_ret > quantile_80_100_ret):
        #     return quantile_20_40_df

        # elif (quantile_40_60_ret > smallest_ret) and (quantile_40_60_ret > quantile_0_20_ret) and (quantile_40_60_ret > quantile_20_40_ret) and (quantile_40_60_ret > largest_ret) and (quantile_40_60_ret > quantile_60_80_ret) and (quantile_40_60_ret > quantile_80_100_ret):
        #     return quantile_40_60_df

        # elif (quantile_60_80_ret > smallest_ret) and (quantile_60_80_ret > quantile_0_20_ret) and (quantile_60_80_ret > quantile_20_40_ret) and (quantile_60_80_ret > largest_ret) and (quantile_60_80_ret > quantile_40_60_ret) and (quantile_60_80_ret > quantile_80_100_ret):
        #     return quantile_60_80_df      

        # elif (quantile_80_100_ret > smallest_ret) and (quantile_80_100_ret > quantile_0_20_ret) and (quantile_80_100_ret > quantile_20_40_ret) and (quantile_80_100_ret > largest_ret) and (quantile_80_100_ret > quantile_40_60_ret) and (quantile_80_100_ret > quantile_60_80_ret):
        #     return quantile_80_100_df                        

        # else:
        #     return smallest_ret_lst






        myrs_top = self.rs[pc1.nlargest(self.x_factor_qtrs).index]
        t_lst = list(myrs_top)        
        myrs_top = myrs_top.mean(1)
        myrs_top2 = myrs_top.cumsum().apply(np.exp)
        top_25_ret = myrs_top2.iloc[-1]

        myrs_50 = self.rs[pc1.nlargest(int(self.x_factor_qtrs * 2)).index]
        for col in myrs_50.columns:
            if col in t_lst:
                del myrs_50[col]      
        myrs_50 = myrs_50.mean(1)  
        myrs_502 = myrs_50.cumsum().apply(np.exp)
        top_50_ret = myrs_502.iloc[-1]

        myrs_25_2 = self.rs[pc1.nsmallest(self.x_factor_qtrs).index]
        t_lst = list(myrs_25_2)
        myrs_25_2 = myrs_25_2.mean(1)
        myrs_bottom2 = myrs_25_2.cumsum().apply(np.exp)
        bottom_25_ret = myrs_bottom2.iloc[-1]          

        myrs_75 = self.rs[pc1.nsmallest(int(self.x_factor_qtrs * 2)).index]
        for col in myrs_75.columns:
            if col in t_lst:
                del myrs_75[col]      
        myrs_75 = myrs_75.mean(1)  
        myrs_752 = myrs_75.cumsum().apply(np.exp)
        bottom_50_ret = myrs_752.iloc[-1]        

          


        top_25_ret_lst = self.rs[pc1.nlargest(self.x_factor_qtrs).index]
        t_lst = list(top_25_ret_lst)

        top_50_ret_lst = self.rs[pc1.nlargest(self.x_factor_qtrs * 2).index]
        for col in top_50_ret_lst.columns:
            if col in t_lst:
                del top_50_ret_lst[col]

        bottom_25_ret_lst = self.rs[pc1.nsmallest(self.x_factor_qtrs).index]
        t_lst = list(bottom_25_ret_lst)

        bottom_50_ret_lst = self.rs[pc1.nsmallest(self.x_factor_qtrs * 2).index]
        for col in bottom_50_ret_lst.columns:
            if col in t_lst:
                del bottom_50_ret_lst[col]        



        st.subheader(f"◾ __Principal Components From Ticker List__")

        st.write(f"__◾ LARGEST PCA VALUES = 【{round(largest_ret,2)}】-【{len(largest_ret_lst.columns)}】__")
        st.text(f"◾ {list(largest_ret_lst.columns)}")

        st.write(f"__◾ Top 25 PCA VALUES = 【{round(top_25_ret,2)}】-【{len(top_25_ret_lst.columns)}】__")
        st.text(f"◾ {list(top_25_ret_lst.columns)}")

        st.write(f"__◾ Middle 50 PCA VALUES = 【{round(top_50_ret,2)}】-【{len(top_50_ret_lst.columns)}】__")
        st.text(f"◾ {list(top_50_ret_lst.columns)}")

        st.write(f"__◾ Middle 75 PCA VALUES = 【{round(bottom_50_ret,2)}】-【{len(bottom_50_ret_lst.columns)}】__")
        st.text(f"◾ {list(bottom_50_ret_lst.columns)}")        

        st.write(f"__◾ Bottom 25 PCA VALUES = 【{round(bottom_25_ret,2)}】-【{len(bottom_25_ret_lst.columns)}】__")        
        st.text(f"◾ {list(bottom_25_ret_lst.columns)}")

        st.write(f"__◾ SMALLEST PCA VALUES = 【{round(smallest_ret,2)}】-【{len(smallest_ret_lst.columns)}】__")
        st.text(f"◾ {list(smallest_ret_lst.columns)}")

        st.write(f"__◾ SPY500 VALUES = 【{round(spy500_ret,2)}】__")





        if (largest_ret > smallest_ret) and (largest_ret > top_25_ret) and (largest_ret > top_50_ret) and (largest_ret > bottom_50_ret) and (largest_ret > bottom_25_ret):
            return largest_ret_lst

        elif (smallest_ret > largest_ret) and (smallest_ret > top_25_ret) and (smallest_ret > top_50_ret) and (smallest_ret > bottom_50_ret) and (smallest_ret > bottom_25_ret):
            return smallest_ret_lst          

        elif (top_25_ret > smallest_ret) and (top_25_ret > largest_ret) and (top_25_ret > top_50_ret) and (top_25_ret > bottom_50_ret) and (top_25_ret > bottom_25_ret):
            return top_25_ret_lst

        elif (top_50_ret > smallest_ret) and (top_50_ret > top_25_ret) and (top_50_ret > largest_ret) and (top_50_ret > bottom_50_ret) and (top_50_ret > bottom_25_ret):
            return top_50_ret_lst

        elif (bottom_50_ret > smallest_ret) and (bottom_50_ret > top_25_ret) and (bottom_50_ret > top_50_ret) and (bottom_50_ret > largest_ret) and (bottom_50_ret > bottom_25_ret):
            return bottom_50_ret_lst

        elif (bottom_25_ret > smallest_ret) and (bottom_25_ret > top_25_ret) and (bottom_25_ret > top_50_ret) and (bottom_25_ret > largest_ret) and (bottom_25_ret > bottom_50_ret):
            return bottom_25_ret_lst            

        else:
            return smallest_ret_lst



        # if (top_3rd_ret > middle_3rd_ret) and (top_3rd_ret > bottom_3rd_ret):
        #     return top_3rd_ret_lst

        # elif (middle_3rd_ret > top_3rd_ret) and (middle_3rd_ret > bottom_3rd_ret):
        #     return middle_3rd_ret_lst

        # elif (bottom_3rd_ret > top_3rd_ret) and (bottom_3rd_ret > middle_3rd_ret):
        #     return bottom_3rd_ret_lst        

        # else:
        #     return top_3rd_ret
