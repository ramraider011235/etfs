import datetime as dt
from pathlib import Path
import pandas as pd
import streamlit as st

import src.tools.functions as f0

pd.set_option('display.max_columns', None)



class Builder_1(object):
    

    def __init__(self, day1):
        month1 = str(day1)[:7]
        year1 = str(day1)[:4]
        now = dt.date.today()
        now = now.strftime('%m-%d-%Y')
        yesterday = dt.date.today() - dt.timedelta(days = 3)
        yesterday = yesterday.strftime('%m-%d-%Y')

        self.saveRec = Path(f"/home/gdp/hot_box/i4m/data/recommenders/{year1}/{month1}/{day1}/")
        if not self.saveRec.exists():
            self.saveRec.mkdir(parents=True)            

        self.save_finviz = Path(f"/home/gdp/hot_box/i4m/data/finviz/{month1}/{day1}/finviz.csv")
        if not self.save_finviz.exists():
            self.save_finviz.mkdir(parents=True)

   
    def source_data_1(self):
        data_finviz = pd.read_csv(self.save_finviz).round(4).fillna(0.001)
        data_0 = pd.DataFrame(f0.clean_sort(data_finviz))
        data_0.to_pickle(self.saveRec / "a1_finviz.pkl")
        print("\n[0] Bulk Data:")
        print(f"---> Total Stock Count: {data_0.shape} \n")        
        return data_0


    def source_data_2(self, data_0, a='y', b='y', c='y', d=3.0, e=0.0, f=0.0, g=0.0):
        data = pd.DataFrame(data_0)
        print(f"[1] Analyst Recom & Target Price")


        if a == 'y':
            data_1a1 = data[data['analyst_recom'] != 0.001]
            data_1a2 = data_1a1[data_1a1['target_price'] != 0.001]
            data = pd.DataFrame(data_1a2.copy())
            print(f"---> Stocks With a Stated Analyst Recom: {data_1a1.shape}") 
            print(f"---> Stocks With A target_price: {data_1a2.shape}")            

        if b == 'y':
            data = data[data['target_price'] > data['price']]
            print(f"---> Target Price > Current Price: {data.shape}")
            
        if data.empty == False:
            data = data[data['analyst_recom'] <= d]
            print(f"---> Average Analyst Recom =< {d}: {data.shape}")  

        if c == 'y':     
            data_b1 = data[data['eps_growth_quarter_over_quarter'] >= e]
            data_b2 = data_b1[data_b1['sales_growth_quarter_over_quarter'] >= f]
            data_b3 = data_b2[data_b2['operating_margin'] >= g]
            data = pd.DataFrame(data_b3.copy())
            print(f"\n[1b] EPS_Growth_Q2Q, Sales_Growth_Q2Q, Operating_Margin:")
            print(f"---> eps_growth_quarter_over_quarter >= {e}%: {data_b1.shape}")
            print(f"---> sales_growth_quarter_over_quarter >= {f}%: {data_b2.shape}")               
            print(f"---> operating_margin >= {g}%: {data_b3.shape}\n")            
                           

        data.to_pickle(self.saveRec / "recommender_01_return_dataFrame.pkl")
        return data


    def run_mod_1(self, a, b, c, d, e, f, g):
        data_0 = self.source_data_1()
        data_1 = self.source_data_2(data_0, a, b, c, d, e, f, g)
        return data_0, data_1
