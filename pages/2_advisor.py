import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
from yahooquery import Ticker
from os.path import exists
from pathlib import Path
import pickle5 as pickle
from dateutil.relativedelta import relativedelta
import datetime as dt
import numpy as np
np.random.seed(42)

import src.tools.functions as f0
from src.data.get_history import Get_Stock_History
from src.models.portfolio import proof as p1
from src.models.portfolio.ml_predict import ML_Classifier_Predictor
from src.models.portfolio.proof_port import The_Portfolio_Optimizer as p2
from src.models.portfolio.web_pca import The_PCA_Analysis as pca
from src.models.portfolio.web_monteCarloCholesky import MonteCarloCholesky as mcc
from src.models.portfolio.random_forest import The_Random_Forest as rf1


# _________________________________________________________________________________________________________________________________________
# |                                                                                                                                      |
# | · · · · · · · · · · · · · · · · · · · · · · · · · · · · · [[ ADVISOR ]] · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·  |
# |______________________________________________________________________________________________________________________________________|


class Advisor(object):


    def __init__(self):
        st.sidebar.write('<style>div.st-bf{flex-direction:row;} div.st-ag{font-weight:bold;padding-left:1px;}</style>', unsafe_allow_html=True)
        st.write('<style>div.st-bf{flex-direction:row;} div.st-ag{font-weight:bold;padding-left:1px;}</style>', unsafe_allow_html=True)
        self.mode_select = "Model"



    def model1(self):       
        st.title('__𝄖𝄖𝄗𝄗𝄘𝄘𝄙𝄙𝄚𝄚 Model 𝄚𝄚𝄙𝄙𝄘𝄘𝄗𝄗𝄖𝄖__')
        st.write('----------', '\n', '----------')            
    
        
        cherry_pick = '2023-02-01'  
        
        self.cherry_pick = cherry_pick
        self.start1 = str(cherry_pick)[:10]
        self.day1 = str(cherry_pick)[8:]
        self.month1 = str(cherry_pick)[:7]
        self.month2 = str(cherry_pick)[5:7]
        self.year1 = str(cherry_pick)[:4]
        self.ender_date = str(dt.datetime.now())[:10]
        start_date_101 = dt.date(int(str(cherry_pick)[:4]), int(str(cherry_pick)[5:7]), int(str(cherry_pick)[8:]))
        self.years_ago = str(start_date_101 - relativedelta(years=1, days=0))[:10]

        self.saveReport = Path(f"reports/portfolio/{self.month1}/{self.start1}/")
        self.saveRec = Path(f"data/recommenders/{self.year1}/{self.month1}/{self.start1}/")
        self.saveRaw = Path(f"data/raw/{self.month1}/{self.start1}/")
        self.saveScreeners = Path(f"data/screeners/{self.month1}/{self.start1}/")
        self.saveTickers = Path(f"data/tickers/{self.month1}/{self.start1}/")                
        
        investment = 20000.0
        num_sims_mpt = 10000
        num_sims_mcc = int(round(num_sims_mpt / 3,0))
        max_wt = 34.0
        rfr, treasury_date = 0.0339, '2023-02-01'

        run_list = ['max_sharpe', 'equal_wt', 'mcc', 'min_volatility']

        graphit_0 = "Yes"
        self.graph1 = f0.true_false(graphit_0)

        file_saver = 'Yes'
        self.save1 = f0.true_false(file_saver)   

        optimize_method = 'markowitz'


# ______________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · · · · · · · · [[ Model ]] · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·|


        cols = st.sidebar.columns(3)
        with cols[1]:
            run_proof_button = st.button(" Run ")
        st.sidebar.write('-'*12)


        if run_proof_button:            
            self.saveRec = Path(f"data/recommenders/{self.year1}/{self.month1}/{self.start1}/")
            self.saveRaw = Path(f"data/raw/{self.month1}/{self.start1}/")
            self.saveScreeners = Path(f"data/screeners/{self.month1}/{self.start1}/")
            self.saveTickers = Path(f"data/tickers/{self.month1}/{self.start1}/")
            self.saveReport = Path(f"reports/portfolio/{self.month1}/{self.start1}/")

            self.advisor1 = Path(f"data/advisor/build/{self.month1}/{self.start1}/")
            if not self.advisor1.exists():
                self.advisor1.mkdir(parents=True)

            self.port_res = Path(f"reports/port_results/{self.month1}/{self.start1}/")
            if not self.port_res.exists():
                self.port_res.mkdir(parents=True)


# _______________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · [[ HYPERPARAMETER · FEATURIZATION ]] · · · · · · · · · · · · · · · · · · · · · · · · ·|


            data = pd.read_pickle(self.saveRec / f"recommender_05_return_dataFrame.pkl")
            
            data = data.sort_values('my_score', ascending=False)
            port_tics = list(data["ticker"])      
            
            st.write(f"◾ Investment: __【{f'${investment:,.2f}'}】__")
            st.write(f"◾ Model Simulations (MPT) = __【{f'{num_sims_mpt:,}'}】__")
            st.write(f"◾ Model Simulations (MCC) = __【{f'{num_sims_mcc:,}'}】__")
            st.write(f"◾ Risk·Free·Rate (10·yr·T-bill) = __【{round(rfr * 100, 2)}%  ~ {treasury_date}】__")
            st.write(f"◾ Mean my_score = __【{round(data['my_score'].mean(), 2)}】__")        
            st.write(f"__◾ Total Tickers In Model = 【{len(port_tics)}】__")    

            st.dataframe(data.copy().round(2))

# ______________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · · · · [[ DATA · COLLECTION ]] · · · · · · · · · · · · · · · · · · · · · · · · · · · ·|
            

            df_train_data, df_test_data = Get_Stock_History(day_0=self.start1).function1(port_tics)
                        

# _____________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · · · · · · [[ PCA & MPT ]] · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·|
            
            
            df_pca = df_train_data.copy()

            st.markdown(len(df_pca.columns))


            if optimize_method == 'markowitz':
                (sharpe1, vol1) = p2(self.start1, self.save1, self.graph1).optimize(investment, num_sims_mpt, max_wt, df_pca, rfr)
        


# _____________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · · · · [[ MAXIMUM · SHARPE ]] · · · · · · · · · · · · · · · · · · · · · · · · · · · ·|


            def maximum_sharpe_portfolio():
                max_sharpe_df_b = pd.DataFrame(sharpe1).reset_index()
                max_sharpe_df_3 = pd.DataFrame(max_sharpe_df_b.copy())

                st.write('-------', '\n', '------')
                st.header("▶ MAXIMUM SHARPE RATIO")
                data = pd.DataFrame(df_test_data.filter(max_sharpe_df_3['ticker']).copy())          
                if max_sharpe_df_3['allocation'].sum() > 99:
                    p1.Model_Concept(self.start1, investment, 'maximum_sharpe_ratio', self.save1, self.graph1).setup(max_sharpe_df_3, data)
                else:
                    a = max_sharpe_df_3['allocation'].sum()
                    new_l = []
                    for i in max_sharpe_df_3['allocation']:
                        new_l.append(round((i * 100) / a, 0))
                    max_sharpe_df_3['allocation'] = new_l
                    p1.Model_Concept(self.start1, investment, 'maximum_sharpe_ratio', self.save1, self.graph1).setup(max_sharpe_df_3, data)
                return

# _______________________________________________________________________________________________________________________________________
# | · · · · · · · · · · · · · · · · · · · · · · · · · [[ MINIMUM · VOLATILITY ]] · · · · · · · · · · · · · · · · · · · · · · · · · · · ·|


            def min_volatility_portfolio():
                min_vol_df = pd.DataFrame(vol1.copy()).reset_index()
                try:
                    del min_vol_df['rank']
                except Exception as e:
                    pass


                st.write('------', '\n', '-----')
                st.header("▶ MINIMUM VOLATILITY")
                data = pd.DataFrame(df_test_data.filter(min_vol_df['ticker']).copy())
                if min_vol_df['allocation'].sum() > 99:
                    p1.Model_Concept(self.start1, investment, 'minimum_volatility_portfolio', self.save1, self.graph1).setup(min_vol_df, data)
                    return
                else:
                    a = min_vol_df['allocation'].sum()
                    new_l = []
                    for i in min_vol_df['allocation']:
                        new_l.append(round((i * 100) / a, 0))
                    min_vol_df['allocation'] = new_l                
                    p1.Model_Concept(self.start1, investment, 'minimum_volatility_portfolio', self.save1, self.graph1).setup(min_vol_df, data)
                    return


# ___________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · · · [[ EQUALLY · WEIGHTED ]] · · · · · · · · · · · · · · · · · · · · · · · · · · ·|


            def equally_weighted_portfolio():
                equal_wt_df_a = pd.DataFrame(sharpe1).reset_index()
                min_vol_df_a = pd.DataFrame(vol1).reset_index()
                equal_wt_df_a = equal_wt_df_a.append(min_vol_df_a)

                try:
                    del equal_wt_df_a['rank']
                except Exception as e:
                    pass

                equal_wt_df = pd.DataFrame(equal_wt_df_a).reset_index()

                st.write('------', '\n', '------')
                st.header("▶ EQUALLY WEIGHTED")      
                equal_wt_df = equal_wt_df.round()          
                equal_wt_df["allocation"] = 100 / len(equal_wt_df["ticker"])
                data = pd.DataFrame(df_test_data.filter(equal_wt_df["ticker"]))
                p1.Model_Concept(self.start1, investment, 'equally_weighted', self.save1, self.graph1).setup(equal_wt_df, data)    
                return
                

# _____________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · · [[ MONTE · CARLO · CHOLESKY ]] · · · · · · · · · · · · · · · · · · · · · · · · · ·|


            def monte_carlo_cholesky_portfolio():
                max_sharpe_df_a = pd.DataFrame(sharpe1).reset_index()
                min_vol_df_a = pd.DataFrame(vol1).reset_index()
                max_sharpe_df_a = max_sharpe_df_a.append(min_vol_df_a)

                monte_carlo_cholesky = pd.DataFrame(max_sharpe_df_a.copy()).reset_index()                    
                    
                st.write('------', '\n', '------')
                st.header("▶ MONTE CARLO CHOLESKY")
                data_train = pd.DataFrame(df_train_data.filter(monte_carlo_cholesky["ticker"]))
                mcc_fin_df = mcc(self.start1).mcc_sharpe(rfr, data_train, num_sims_mcc, self.graph1)
                data = pd.DataFrame(df_test_data.filter(mcc_fin_df["ticker"]))
                p1.Model_Concept(self.start1, investment, 'monte_carlo_cholesky', self.save1, self.graph1).setup(mcc_fin_df, data)
                return


# ______________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · · · · [[ CREATE · PORTFOLIO ]] · · · · · · · · · · · · · · · · · · · · · · · · · · · ·|            


            if 'max_sharpe' in run_list:
                maximum_sharpe_portfolio()

            if 'min_volatility' in run_list:
                min_volatility_portfolio()                              

            if 'equal_wt' in run_list:
                equally_weighted_portfolio()
                
            if 'mcc' in run_list:
                monte_carlo_cholesky_portfolio()

              
    st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)    
 

            
            
            
if __name__ == '__main__':
    Advisor().model1()