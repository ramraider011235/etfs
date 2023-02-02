from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import seaborn as sns
import numpy as np

import pandas as pd
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
np.random.seed(42)
pd.set_option('display.max_columns', None)  # or 1000
plt.style.use('ggplot')
sm, med, lg = 10, 15, 20
plt.rc("font", size=sm)                                                        # controls default text sizes
plt.rc("axes", titlesize=med)                                                  # fontsize of the axes title
plt.rc("axes", labelsize=med)                                                  # fontsize of the x & y labels
plt.rc("xtick", labelsize=sm)                                                  # fontsize of the tick labels
plt.rc("ytick", labelsize=sm)                                                  # fontsize of the tick labels
plt.rc("legend", fontsize=sm)                                                  # legend fontsize
plt.rc("figure", titlesize=lg)                                                 # fontsize of the figure title
plt.rc("axes", linewidth=2)                                                    # linewidth of plot lines
plt.rcParams["figure.figsize"] = [9, 7]
plt.rcParams["figure.dpi"] = 100



class ML_Classifier_Predictor(object):


    def __init__(self, graph, date):
        self.graph = graph
        self.graph_type = 'A'
        self.date = date


    def s1(self, stock):
        loc1 = f"/home/gdp/hot_box/i4m/data/finviz/tickers/{stock}_finviz_hist.csv"
        x = pd.DataFrame(pd.read_csv(loc1)).sort_values('date', ascending=False).set_index('date')
        x.index = pd.to_datetime(x.index)
        x = pd.DataFrame(x[:self.date].copy()).reset_index()

        num1 = 15  # 21, 32, 42, int(len(x) - 1)
        data = pd.DataFrame(x[list(x.columns)[6:]].copy()).round(4).fillna(0.0).iloc[-num1:]
        data = data.rename(columns={"up_or_down": "target_A_up_or_down"})
        data['target_B_trailing_start_day'] = data['price'] > data['price'].iloc[1]
        data['target_C_trailing_10_day'] = data['price'] > data['price'].shift(10)
        data = data.rename(columns={"target_A_up_or_down": "target"})
        return data


    def s2(self, data, stock, c, num2):
        data = pd.DataFrame(data)
        df = data.copy().dropna()

        y = df.pop('target')
        X = df.values
        X_train, X_test, y_train, y_test = train_test_split(X, y)         # Split data [training/test]

        model = RandomForestClassifier(
            criterion='entropy',
            oob_score=True,
            random_state=42,
            verbose=1,
            warm_start=True,
            class_weight='balanced_subsample',
        )
        # model = RandomForestRegressor(
        #     criterion='poisson', 
        #     # warm_start=True
        # )
        model.fit(X_train, y_train)
        
        if self.graph == True:
            accuracy = model.score(X_test, y_test)
            pred_proba = model.predict_proba(X_test)
            st.write(pred_proba)
            st.write(f"[{c}/{num2}] - ", stock, " - Accuracy: ", accuracy)
                                                                                              
        num_days = 10
        df = data.copy().dropna()
        df.pop('target')
        new_data = df.values                                                 # Set Data = OG Data

        predictions = []
        for i in range(num_days):
            pred = model.predict(new_data)[0]
            predictions.append(pred)
            new_data[0] = new_data[1]
            new_data[1] = new_data[2]
            new_data[2] = pred
            
        if self.graph == True:  
            st.text(f"{stock} Predictions: {predictions} \n")

        if True in predictions:
            return True       
        else:
            return False


    def s3(self, data):
        data = pd.DataFrame(data)
        df = pd.DataFrame(data.copy()).dropna()
        n = len(df.columns)
        
        y = df.pop('target')
        X = df.values
        (X_train, X_test, y_train, y_test) = train_test_split(X, y)                      # Split data [training/test]

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        
        importances = rf.feature_importances_[:n]
        feature_importance = np.array(importances)
        indices = np.argsort(importances)[::-1]
        feature_names = list(df.columns[indices])
        
        data={
            'feature_names':feature_names, 
            'feature_importance':feature_importance
        }
        
        fi_df = pd.DataFrame(data)
        fi_df.sort_values(
            by=['feature_importance'], 
            ascending=False, 
            inplace=True
        )
        
        fi_df = fi_df[fi_df['feature_importance'] > 0.0]
        st.dataframe(fi_df)

        fig, ax = plt.subplots(figsize=(20,10))
        sns.barplot(
            x=fi_df['feature_importance'], 
            y=fi_df['feature_names'],
            )
        plt.title('RANDOM FOREST FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
        st.plotly_chart(fig, use_container_width=True)



    def s4(self, df):
        df = pd.DataFrame(df)
        n = len(df.columns)-1

        y = df.pop('target')
        X = df.values
        X_train, X_test, y_train, y_test = train_test_split(X, y)                        # Split data [training/test]

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        importances = rf.feature_importances_[:n]
        feature_importance = np.array(importances)
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        feature_names = list(df.columns[indices])
        score_keeper = 0.0
        
        data={
            'feature_names':feature_names, 
            'feature_importance':feature_importance, 
            'std':std
        }

        fi_df = pd.DataFrame(data)
        fi_df.sort_values(
            by=['feature_importance'], 
            ascending=False, 
            inplace=True
        )

        fi_df = fi_df[fi_df['feature_importance'] > 0.0]
        feature_importance2 = np.array(fi_df['feature_importance'])
        feature_names2 = np.array(fi_df['feature_names'])
        std2 = np.array(fi_df['std'])

        st.write("> Feature ranking:", f"\n{'_'*25}")
        for f in range(n):
            if feature_importance[indices[f]] > 0.0:
                st.write(
                    "%d. %s (%f)" % (f + 1, feature_names[f], importances[indices[f]])   
                )
                score_keeper += importances[indices[f]]
        st.write(
            '\n', 
            f">>> Total Variance Accounted For [{round((score_keeper*100),4)} %]", 
            '\n'
        )

        fig, ax = plt.subplots(figsize=(20,10))
        n1 = len(feature_importance2)       
        ax.bar(
            range(n1),
            height=feature_importance2,
            yerr=std2, 
            color="r",
            align="center",
        )
        ax.set_xticks(range(n1))
        ax.set_xticklabels(feature_names2, rotation = 90)
        ax.set_xlim([-1, n1])
        ax.set_xlabel("importance")
        ax.set_title("Feature Importances")
        plt.legend(["importance", "std"], loc="lower right")
        st.plotly_chart(fig, use_container_width=True)



    def main(self, tickers=None):
        c = 0
        res_lst = []
        num2 = len(tickers)
        
        for ticker in tickers:
            c += 1
            
            try:
                data = self.s1(ticker)
                res = self.s2(data, ticker, c, num2)
                
                if res == True:
                    res_lst.append(ticker)

                if self.graph == 'Yes':

                    if self.graph_type == 'A':
                        self.s3(data)
                        
                    if self.graph_type == 'B':
                        self.s4(data)     

            except Exception as e:
                st.write(f"ERROR!!!!! - {ticker} - {e}")

        return res_lst