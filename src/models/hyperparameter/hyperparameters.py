import streamlit as st
import datetime as dt
import pandas as pd
from os.path import exists
from pathlib import Path
import pickle5 as pickle

from src.tools import functions as f0
from src.models.portfolio.ml_predict import ML_Classifier_Predictor





class Featurization(object):


    def __init__(self, start_date, filter_0, filter_1):
        st.header(f"__❱ Hyperparameter · Optimization:__")
        st.write('-------------')

        self.start_date = start_date
        self.month_0 = str(self.start_date)[:7]
        self.year_0 = str(self.start_date)[:4]        
        self.filter_0 = filter_0
        self.filter_1 = filter_1
        self.saveRec = Path(f"data/recommenders/{self.year_0}/{self.month_0}/{self.start_date}/")


    def function_1(self):
        data = pd.DataFrame(f0.get_final_df(date1=self.start_date, data_plot='A'))
        st.write(f"__◾ Total Tickers Approved = 【{len(data['ticker'])}】__")

        if self.filter_0 == 'Range':
            thee_score_a, thee_score_b, my_mean, variance, sd = f0.get_stats1(data, self.filter_1, 'Range')
            data = data[(data['my_score'] >= thee_score_a) & (data['my_score'] <= thee_score_b)]
            st.write(f"\
                ◾ Mean Composite Score =__【{round(my_mean, 2)}】__ \n\n\
                ◾ Variance =__【{round(variance, 2)}】__ \n\n\
                ◾ Std =__【{round(sd, 2)}】__ \n\n\
                ◾ Adjusted Lower MY_SCORE =__【{thee_score_a}】__ \n\n\
                ◾ Adjusted Upper MY_SCORE =__【{thee_score_b}】__ "
                )
            
        if self.filter_0 == "Minimum":
            thee_score, my_mean, variance, sd = f0.get_stats1(data, self.filter_1, 'Minimum')
            data = data[data['my_score'] >= thee_score]
            st.write(f"\
                ◾ Mean Composite Score =__【{round(my_mean, 2)}】__ \n\n\
                ◾ Variance =__【{round(variance, 2)}】__ \n\n\
                ◾ Std =__【{round(sd, 2)}】__ \n\n\
                ◾ Adjusted Lower MY_SCORE =__【{round(thee_score, 2)}】__"
                )
            
        if self.filter_0 == 'None':
            pass

        return data


    def function_2(self, data, hyperparameter_method, hyperparameter_save, hyperparameter_rsi):
        if hyperparameter_method == 'Yes':
            
            if hyperparameter_save == 'Yes':

                if exists(self.saveRec / f"recommender_hyperperameter.pkl"):
                    open1 = (str(self.saveRec) + "/recommender_hyperperameter.pkl")
                    with open(open1, "rb") as fh:
                        data = pd.DataFrame(pickle.load(fh))   
                else:            
                    port_tics = list(data['ticker'])
                    port_tics = ML_Classifier_Predictor(graph=False, date=self.start_date).main(port_tics)
                    data = pd.DataFrame(data[data['ticker'].isin(port_tics)])
                    data.to_pickle(self.saveRec / f"recommender_hyperperameter.pkl")
                        
            else:            
                port_tics = list(data['ticker'])
                port_tics = ML_Classifier_Predictor(graph=False, date=self.start_date).main(port_tics)
                data = pd.DataFrame(data[data['ticker'].isin(port_tics)])
                data.to_pickle(self.saveRec / f"recommender_hyperperameter.pkl")       

        if hyperparameter_rsi == 'Yes':
            data = data[data['relative_strength_index_14'] >= 47]
            data = data[data['relative_strength_index_14'] <= 71]
            # data = data[data['relative_strength_index_14'] <= 70.0]
            # data = data[data['relative_strength_index_14'] >= 47.0]

        return data               


    def function_3(self, data, investment, num_sims_mpt, num_sims_mcc, rfr, treasury_date):
        data = pd.DataFrame(data)
        data = data.sort_values('my_score', ascending=False)
        port_tics = list(data["ticker"])      
        
        st.write(f"◾ Investment: __【{f'${investment:,.2f}'}】__")
        st.write(f"◾ Model Simulations (MPT) = __【{f'{num_sims_mpt:,}'}】__")
        st.write(f"◾ Model Simulations (MCC) = __【{f'{num_sims_mcc:,}'}】__")
        st.write(f"◾ Risk·Free·Rate (10·yr·T-bill) = __【{round(rfr * 100, 2)}%  ~ {treasury_date}】__")
        st.write(f"◾ Mean my_score = __【{round(data['my_score'].mean(), 2)}】__")        
        st.write(f"__◾ Total Tickers In Model = 【{len(port_tics)}】__")    

        st.dataframe(data.copy().set_index('company').round(2))
        st.text(list(data['ticker']))
        st.markdown(len(data['ticker']))
        
        return data, port_tics