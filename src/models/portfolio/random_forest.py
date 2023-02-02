import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

pd.set_option("display.max_rows", 50)
pd.set_option('display.max_columns', None)
plt.style.use(["seaborn-darkgrid", "seaborn-poster"])
plt.rcParams["figure.figsize"] = [13, 6.5]
plt.style.use("seaborn")
sm, med, lg = 10, 15, 25
plt.rc("font", size=sm)                       # controls default text sizes
plt.rc("axes", titlesize=med)                 # fontsize of the axes title
plt.rc("axes", labelsize=med)                 # fontsize of the x & y labels
plt.rc("xtick", labelsize=sm)                 # fontsize of the tick labels
plt.rc("ytick", labelsize=sm)                 # fontsize of the tick labels
plt.rc("legend", fontsize=sm)                 # legend fontsize
plt.rc("figure", titlesize=lg)                # fontsize of the figure title
plt.rc("axes", linewidth=2)                   # linewidth of plot lines
plt.rcParams["figure.figsize"] = [15, 6.5]
plt.rcParams["figure.dpi"] = 134



class The_Random_Forest(object):


    def __init__(self, graph1, day1, investment=200000.0):
        self.graph1 = graph1
        self.investment = investment
        self.day1 = day1

        month1 = str(day1)[:7]
        year1 = str(day1)[:4]
        
        try:
            self.bulk_data_file = Path(f"/home/gdp/hot_box/i4m/data/random_forest/{year1}/{month1}/{day1}/")
            if not self.bulk_data_file.exists():
                self.bulk_data_file.mkdir(parents=True)        
        except Exception as e:
            self.bulk_data_file = Path(f"/Users/gdp/hot_box/i4m/data/random_forest/{year1}/{month1}/{day1}/")
            if not self.bulk_data_file.exists():
                self.bulk_data_file.mkdir(parents=True)                    


    def collect_data(self, data_0, years_ago):
        spy_hist = yf.download('^GSPC', start=years_ago, end=self.day1)
        data_0['SP_500'] = spy_hist['Close']
        return data_0


    def clean_data(self, data_0, years_ago):        
        df_hist = pd.DataFrame(self.collect_data(data_0, years_ago))
        return df_hist.round(2)


    def configure_shares(self, df, years_ago):
        fd = pd.DataFrame(df).dropna()
        shares_lst = []
        cols_lst = fd.columns
        col_num = float(len(fd.columns)-1)
        x_sum = round(self.investment / col_num)
        for i in fd.iloc[0]:
            shares_lst.append(round(x_sum/i))    
        for k, v in enumerate(shares_lst):
            fd[cols_lst[k]] = (fd[cols_lst[k]] * v)    
        fd['sums'] = fd.sum(axis=1)
        spy_hist = yf.download('^GSPC', start=years_ago, end=self.day1)
        fd['SP_500'] = spy_hist['Close']            
        fd['SP_500'] = self.investment / fd['SP_500']        
        return fd.round(2)


    def configure_returns(self, fd):
        fd = pd.DataFrame(fd)
        df_daily_returns = fd.pct_change().iloc[1:]
        df_daily_returns['vs_sp500'] = df_daily_returns['sums'] > df_daily_returns['SP_500']
        del df_daily_returns['SP_500']
        del df_daily_returns['sums']            
        return df_daily_returns 


    def random_forest_setup(self, df0):
        df = pd.DataFrame(df0).dropna(thresh=1, axis='columns')
        st.write(f"Total Tickers: {len(df.columns)}")

        # Make a numpy array called y containing T/F values
        y = df.pop('vs_sp500').values       

        # Make 2D numpy array containing feature-data (everything except the labels)
        X = df.values

        # Use sklearn's train_test_split to create train and test set. 
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Use sklearn's RandomForestClassifier to build model of data
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        # RANDOM FOREST PREDICT RESULTS
        y_predict = rf.predict(X_test)        

        # SCORE
        score_1 = round((rf.score(X_test, y_test)) * 100, 2)
        st.write("score:", score_1)

        # PRECISION
        precision_1 = round((precision_score(y_test, y_predict)) * 100, 2)
        st.write("precision:", precision_1)

        # RECALL
        recall_1 = round((recall_score(y_test, y_predict)) * 100, 2)       
        st.write("recall:", recall_1)

        # BUILD RandomForestClassifier - ( OUT_OF_BAG = TRUE )
        rf = RandomForestClassifier(n_estimators=30, oob_score=True)
        rf.fit(X_train, y_train)

        # ACCURACY SCORE        
        accuracy_score_1 = round((rf.score(X_test, y_test)) * 100, 2)
        st.write("\n accuracy score:", accuracy_score_1)

        # FINAL CONFIGURE OF METRICS
        n = len(rf.feature_importances_)
        importances = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        features = list(df.columns[indices])
        rank_list = []
        for f in range(n):
            print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))
            rank_list.append(features[f])

        if self.graph1 == True:                
            fig, ax = plt.subplots()
            ax.bar(range(n), importances[indices], yerr=std[indices], color="r", align="center")
            ax.set_xticks(range(n))
            ax.set_xticklabels(features, rotation = 69)
            ax.set_xlim([-1, n])
            ax.set_xlabel("Freatures")
            ax.set_ylabel("Importance")
            ax.set_title("Feature Importances")
            st.pyplot(fig)

        st.write('𝄖'*13)
        return rank_list, score_1 #accuracy_score_1



    def run_mod(self, data_X, years_ago):     
        st.write('------', '\n', '-------')
        st.header("__❱ Random · Forest:__")
        st.write('-------------')
                    
        data_0 = pd.DataFrame(data_X)   
        data_0_0 = pd.DataFrame(data_0.copy())
        data_1_0 = self.configure_shares(data_0_0, years_ago)
        data_2_0 = self.configure_returns(data_1_0)
        rank_list_0, accuracy_score_0 = self.random_forest_setup(data_2_0)
        len_0 = len(rank_list_0)

        rf_mark_1 = int(len_0 * 0.69)
        rf_mark_2 = int(len_0 * 0.345)
        rf_mark_3 = int(len_0 * 0.1725)

        s1 = rank_list_0[:rf_mark_1]
        s2 = rank_list_0[:rf_mark_2]       
        s3 = rank_list_0[:rf_mark_3]        

        data_0_1 = data_0.copy().filter(s1)
        data_1_1 = self.configure_shares(data_0_1, years_ago)
        data_2_1 = self.configure_returns(data_1_1)
        rank_list_1, accuracy_score_1 = self.random_forest_setup(data_2_1)     


        if rf_mark_2 >= 10:
            data_0_2 = data_0.copy().filter(s2)
            data_1_2 = self.configure_shares(data_0_2, years_ago)
            data_2_2 = self.configure_returns(data_1_2)
            rank_list_2, accuracy_score_2 = self.random_forest_setup(data_2_2)     
        else:
            accuracy_score_2 = 0.01            


        if rf_mark_3 >= 10:
            data_0_3 = data_0.copy().filter(s3)
            data_1_3 = self.configure_shares(data_0_3, years_ago)
            data_2_3 = self.configure_returns(data_1_3)
            rank_list_3, accuracy_score_3 = self.random_forest_setup(data_2_3)
        else:
            accuracy_score_3 = 0.01



        if (accuracy_score_0 > accuracy_score_1) and (accuracy_score_0 > accuracy_score_2) and (accuracy_score_0 > accuracy_score_3):
            short_rank_list = rank_list_0

        elif (accuracy_score_1 > accuracy_score_0) and (accuracy_score_1 > accuracy_score_2) and (accuracy_score_1 > accuracy_score_3):
            short_rank_list = rank_list_1

        elif (accuracy_score_2 > accuracy_score_1) and (accuracy_score_2 > accuracy_score_0) and (accuracy_score_2 > accuracy_score_3):
            short_rank_list = rank_list_2

        elif (accuracy_score_3 > accuracy_score_1) and (accuracy_score_3 > accuracy_score_2) and (accuracy_score_3 > accuracy_score_0):
            short_rank_list = rank_list_3
            
        else:
            short_rank_list = rank_list_1



        st.markdown(short_rank_list)
        return short_rank_list