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
from src.models.hyperparameter.hyperparameters import Featurization
from src.models.portfolio import proof as p1
from src.models.portfolio.ml_predict import ML_Classifier_Predictor
from src.models.portfolio.proof_port import The_Portfolio_Optimizer as p2
from src.models.portfolio.portfolio_optimizer import Portfolio_Optimizer_2 as p22
from src.models.portfolio.web_pca import The_PCA_Analysis as pca
from src.models.portfolio.web_monteCarloCholesky import MonteCarloCholesky as mcc
from src.models.strategy.indicators import Indicator_Ike as ii
from src.models.portfolio.random_forest import The_Random_Forest as rf1
from src.models.clean import Clean


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
        
        
        # cols = st.sidebar.columns(1)
        # with cols[0]:
        #     cherry_pick = str(
        #         st.date_input(
        #             label="Select Model Date",
        #             value=dt.date(2023, 1, 3),
        #             min_value=dt.date(2022, 3, 24),
        #             max_value=dt.date(2023, 3, 31),
        #         )
        #     )[:10]      
        
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


        sma_ema_choices = ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'SAR', 'BB', 'MACD', 'RSI']
        list_runs = ['max_sharpe', 'best', 'equal_wt', 'mcc', 'min_volatility']
        classifier_list = ['PCA', 'PCA_RF', 'RF_PCA', 'RF', 'None']
        crap_lst = []
        sheen_lst = []     
        
        investment = 20000.0
        num_sims_mpt = 10000
        num_sims_mcc = int(round(num_sims_mpt / 3,0))
        max_wt = 34.0
        rfr, treasury_date = Clean(self.start1).clean_main()



        cols = st.sidebar.columns(1)
        with cols[0]:
            run_list = st.multiselect('Portfolio Options:', options=list_runs, default=list_runs, key='opt1')
                
        
        cols = st.sidebar.columns(2)
        with cols[0]:
            graphit_0 = st.selectbox("Graph:", ("Yes", "No"), index=1)
            self.graph1 = f0.true_false(graphit_0)

            with cols[1]:
                file_saver = st.selectbox("Save:", ("Yes", "No"), index=0)
                self.save1 = f0.true_false(file_saver)            


        optimize_method = 'efficient_frontier'


        cols = st.sidebar.columns(2)
        with cols[0]:
            classifier_select = st.selectbox(label='Classifier Method:', options=classifier_list, index=0) 

            with cols[1]:                            
                if classifier_select == 'PCA':
                    pca_factor = float(st.number_input('PCA-Factor:', 0.01, 1.00, value=0.1))   
                            
                elif classifier_select == 'RF_PCA':
                    pca_factor = float(st.number_input('PCA-Factor:', 0.01, 1.00, value=0.1))
                        
                elif classifier_select == 'PCA_RF':
                    pca_factor = float(st.number_input('PCA-Factor:', 0.01, 1.00, value=0.1))

                elif classifier_select == 'RF':
                    print('k')                     
                    
                elif classifier_select == 'None':
                    print('k') 

                else:
                    print('k')


        cols = st.sidebar.columns(2)
        with cols[0]:
            use_indicator = st.selectbox(label='Technical Strategy:', options=('Yes','No'), index=1)

            if use_indicator == 'Yes':
                with cols[1]:
                    crossover_1 = st.selectbox(label="Strategy:", options=sma_ema_choices, index=0)


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
            st.text(list(data['ticker']))
            st.markdown(len(data['ticker']))

# ______________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · · · · [[ DATA · COLLECTION ]] · · · · · · · · · · · · · · · · · · · · · · · · · · · ·|
            

            df_train_data, df_test_data = Get_Stock_History(day_0=self.start1).function1(port_tics)
                        

# _____________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · · · · · · [[ PCA & MPT ]] · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·|
            
            
            if classifier_select != 'None':
                
                if classifier_select == 'PCA':                
                    chicken_dinna = pca(self.start1).build_pca(df_train_data, pca_factor, self.graph1)
                    df_pca = pd.DataFrame(df_train_data.filter(chicken_dinna).copy())

                elif classifier_select == 'RF':
                    chicken_dinna = rf1(self.graph1, self.start1).run_mod(df_train_data, self.years_ago)
                    df_pca = pd.DataFrame(df_train_data.filter(chicken_dinna).copy())

                elif classifier_select == 'RF_PCA':   
                    rf_res = rf1(self.graph1, self.start1).run_mod(df_train_data, self.years_ago)
                    rf_res_df = df_train_data.filter(rf_res).copy()
                    chicken_dinna = pca(self.start1).build_pca(rf_res_df, pca_factor, self.graph1)
                    df_pca = pd.DataFrame(df_train_data.filter(chicken_dinna).copy())
                                    
                elif classifier_select == 'PCA_RF':
                    pca_res = pca(self.start1).build_pca(df_train_data, pca_factor, self.graph1)
                    pca_res_df = df_train_data.filter(pca_res).copy()
                    chicken_dinna = rf1(self.graph1, self.start1).run_mod(pca_res_df, self.years_ago)
                    df_pca = pd.DataFrame(df_train_data.filter(chicken_dinna).copy())
                    
            else:
                df_pca = df_train_data.copy()

            st.markdown(len(df_pca.columns))


            
            if optimize_method == 'markowitz':
                (sharpe1, vol1) = p2(self.start1, self.save1, self.graph1).optimize(investment, num_sims_mpt, max_wt, df_pca, rfr)

            if optimize_method == 'efficient_frontier':
                sharpe1, vol1, best1 = p22(self.start1, rfr, num_sims_mpt, df_pca, max_wt, self.graph1).run_mod()
                               
            
# _______________________________________________________________________________________________________________________________________
# | · · · · · · · · · · · · · · · · · · · · · · · · · · · [[ STRATEGY ]] · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·|

        
            def sharper(df_0):       
                df1 = pd.DataFrame(df_0)
                port_tics_0 = sorted(list(df1["ticker"]))
                port_tics_1 = []
                cc, ccc = 0.0, len(port_tics_0)                                     

                for p in port_tics_0:
                    if p in crap_lst:
                        df1 = df1.drop(df1[df1["ticker"] == p].index)
                    elif p in sheen_lst:
                        pass
                    elif p not in crap_lst and p not in sheen_lst:
                        port_tics_1.append(p)

                if len(port_tics_1) >= 1:
                    st.write('---', '\n', '---')
                    st.header("__❱ STRATEGY:__")
                    st.write('----------------')
                                                     

                    for p in port_tics_1:
                        try:
                            try:
                                temp = pd.DataFrame(pd.read_pickle(self.advisor1 / f"{p}_hist_{self.ender_date}.pkl"))
                            except Exception as e:
                                temp = pd.DataFrame(Ticker('RYTM').history(period='1y')).reset_index().set_index('date').round(2)
                                del temp['symbol']     
                                temp.to_pickle(self.advisor1 / f"{p}_hist_{self.ender_date}.pkl")

                            temp.index = pd.to_datetime(temp.index)
                            data = temp.loc[:self.start1]                       
                            cc += 1                                           


                            if crossover_1 == 'SMA':
                                try:
                                    x = ii(p, self.start1, cc, ccc, self.graph1).kingpin(crossover_1, data)
                                    if x == p:
                                        sheen_lst.append(p)
                                    else:
                                        crap_lst.append(p)
                                        df1 = df1.drop(df1[df1["ticker"] == p].index)
                                except Exception as e:
                                    crap_lst.append(p)
                                    df1 = df1.drop(df1[df1["ticker"] == p].index)
                                    st.write(f'FAILURE = {p}')
                                    print(e)


                            if crossover_1 == 'EMA':
                                try:
                                    x = ii(p, self.start1, cc, ccc, self.graph1).kingpin(crossover_1, data)
                                    if x == p:
                                        sheen_lst.append(p)
                                    else:
                                        crap_lst.append(p)
                                        df1 = df1.drop(df1[df1["ticker"] == p].index)
                                except Exception:
                                    crap_lst.append(p)
                                    df1 = df1.drop(df1[df1["ticker"] == p].index)
                                    st.write(f'FAILURE = {p}')


                            if crossover_1 == 'WMA':
                                try:
                                    x = ii(p, self.start1, cc, ccc, self.graph1).kingpin(crossover_1, data)
                                    if x == p:
                                        sheen_lst.append(p)
                                    else:
                                        crap_lst.append(p)
                                        df1 = df1.drop(df1[df1["ticker"] == p].index)
                                except Exception:
                                    st.write(f'FAILURE = {p}')
                                    crap_lst.append(p)
                                    df1 = df1.drop(df1[df1["ticker"] == p].index)                                            
                                    

                            if crossover_1 == 'DEMA':
                                try:                                        
                                    x = ii(p, self.start1, cc, ccc, self.graph1).kingpin(crossover_1, data)
                                    if x == p:
                                        sheen_lst.append(p)
                                    else:
                                        crap_lst.append(p)
                                        df1 = df1.drop(df1[df1["ticker"] == p].index)
                                except Exception:
                                    st.write(f'FAILURE = {p}')
                                    crap_lst.append(p)
                                    df1 = df1.drop(df1[df1["ticker"] == p].index)                                            


                            if crossover_1 == 'TEMA':
                                try:                                        
                                    x = ii(p, self.start1, cc, ccc, self.graph1).kingpin(crossover_1, data)
                                    if x == p:
                                        sheen_lst.append(p)
                                    else:
                                        crap_lst.append(p)
                                        df1 = df1.drop(df1[df1["ticker"] == p].index)
                                except Exception:
                                    st.write(f'FAILURE = {p}')    
                                    crap_lst.append(p)
                                    df1 = df1.drop(df1[df1["ticker"] == p].index)                                                 
                                    

                            if crossover_1 == 'TRIMA':
                                try:                                        
                                    x = ii(p, self.start1, cc, ccc, self.graph1).kingpin(crossover_1, data)
                                    if x == p:
                                        sheen_lst.append(p)
                                    else:
                                        crap_lst.append(p)
                                        df1 = df1.drop(df1[df1["ticker"] == p].index)
                                except Exception:
                                    st.write(f'FAILURE = {p}')    
                                    crap_lst.append(p)
                                    df1 = df1.drop(df1[df1["ticker"] == p].index)                                                 
                                

                            if crossover_1 == 'SAR':
                                try:                                        
                                    x = ii(p, self.start1, cc, ccc, self.graph1).kingpin(crossover_1, data)
                                    if x == p:
                                        sheen_lst.append(p)
                                    else:
                                        crap_lst.append(p)
                                        df1 = df1.drop(df1[df1["ticker"] == p].index)
                                except Exception:
                                    st.write(f'FAILURE = {p}')    
                                    crap_lst.append(p)
                                    df1 = df1.drop(df1[df1["ticker"] == p].index)                                                 


                            if crossover_1 == "BB":
                                try:
                                    x = ii(p, self.start1, cc, ccc, self.graph1).kingpin(crossover_1, data)
                                    if x == p:
                                        sheen_lst.append(p)
                                    else:
                                        crap_lst.append(p)
                                        df1 = df1.drop(df1[df1["ticker"] == p].index)
                                except Exception:
                                    st.write(f'FAILURE = {p}')    
                                    crap_lst.append(p)
                                    df1 = df1.drop(df1[df1["ticker"] == p].index)  


                            if crossover_1 == "MACD":
                                try:
                                    x = ii(p, self.start1, cc, ccc, self.graph1).kingpin(crossover_1, data)
                                    if x == p:
                                        sheen_lst.append(p)
                                    else:
                                        crap_lst.append(p)
                                        df1 = df1.drop(df1[df1["ticker"] == p].index)
                                except Exception:
                                    st.write(f'FAILURE = {p}')    
                                    crap_lst.append(p)
                                    df1 = df1.drop(df1[df1["ticker"] == p].index)                                              


                            if crossover_1 == "RSI":
                                try:
                                    x = ii(p, self.start1, cc, ccc, self.graph1).kingpin(crossover_1)
                                    if x == p:
                                        sheen_lst.append(p)
                                    else:
                                        crap_lst.append(p)
                                        df1 = df1.drop(df1[df1["ticker"] == p].index)
                                except Exception:
                                    st.write(f'FAILURE = {p}')    
                                    crap_lst.append(p)
                                    df1 = df1.drop(df1[df1["ticker"] == p].index)                                                
                                        
                        except:
                            crap_lst.append(p)
                            df1 = df1.drop(df1[df1["ticker"] == p].index)
                            st.write(f'FAILURE = {p}')

                return df1


# _____________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · · · · [[ MAXIMUM · SHARPE ]] · · · · · · · · · · · · · · · · · · · · · · · · · · · ·|


            def maximum_sharpe_portfolio():
                max_sharpe_df_b = pd.DataFrame(sharpe1).reset_index()
                if use_indicator == 'Yes':
                    max_sharpe_df_3 = pd.DataFrame(sharper(max_sharpe_df_b))
                else:
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
                if use_indicator == 'Yes':
                    min_vol_df_a = pd.DataFrame(vol1).reset_index()
                    try:
                        del min_vol_df_a['rank']
                    except Exception as e:
                        pass
                    min_vol_df = sharper(min_vol_df_a)
                else: 
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
#   | · · · · · · · · · · · · · · · · · · · · · · · · [[ BEST · WEIGHTED ]] · · · · · · · · · · · · · · · · · · · · · · · · · · ·|


            def best_weighted_portfolio():
                try:
                    best_wt_df_a = pd.DataFrame(best1).reset_index()
                    try:
                        del best_wt_df_a['rank']
                    except Exception as e:
                        pass
                    if best_wt_df_a.empty == False:
                        if use_indicator == 'Yes':                    
                            best_wt_df = pd.DataFrame(sharper(best_wt_df_a))
                        else:
                            best_wt_df = pd.DataFrame(best_wt_df_a.copy())
                            
                        
                        st.write('------', '\n', '------')
                        st.header("▶ BEST WEIGHTED")
                        data = pd.DataFrame(df_test_data.filter(list(best_wt_df["ticker"])))
                        if best_wt_df['allocation'].sum() >= 98:
                            p1.Model_Concept(self.start1, investment, 'best_weighted', self.save1, self.graph1).setup(best_wt_df, data)    
                        else:
                            a = best_wt_df['allocation'].sum()
                            new_l = []
                            for i in best_wt_df['allocation']:
                                new_l.append(round((i * 100) / a, 0))
                            best_wt_df['allocation'] = new_l                
                            p1.Model_Concept(self.start1, investment, 'best_weighted', self.save1, self.graph1).setup(best_wt_df, data)    
                        return
                except Exception as e:
                    print(e)


# ___________________________________________________________________________________________________________________________________
#   | · · · · · · · · · · · · · · · · · · · · · · · · [[ EQUALLY · WEIGHTED ]] · · · · · · · · · · · · · · · · · · · · · · · · · · ·|


            def equally_weighted_portfolio():
                equal_wt_df_a = pd.DataFrame(sharpe1).reset_index()
                try:
                    del equal_wt_df_a['rank']
                except Exception as e:
                    pass
                if use_indicator == 'Yes':                    
                    equal_wt_df = sharper(equal_wt_df_a)
                else:
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
                if use_indicator == 'Yes':
                    monte_carlo_cholesky = sharper(max_sharpe_df_a)
                else:
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
                
            if 'best' in run_list:
                best_weighted_portfolio()                              

            if 'equal_wt' in run_list:
                equally_weighted_portfolio()
                
            if 'mcc' in run_list:
                monte_carlo_cholesky_portfolio()

              
    st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)    
 

            
            
            
if __name__ == '__main__':
    Advisor().model1()