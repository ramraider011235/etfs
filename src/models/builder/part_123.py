import streamlit as st
import datetime as dt
from pathlib import Path
import pandas as pd

import functions as f0

pd.set_option('display.max_columns', None)



class Part_123(object):
    
    def __init__(self, day1):
        month1 = str(day1)[:7]
        year1 = str(day1)[:4]
        now = dt.date.today()
        now = now.strftime('%m-%d-%Y')
        yesterday = dt.date.today() - dt.timedelta(days = 3)
        yesterday = yesterday.strftime('%m-%d-%Y')

        self.saveRaw = Path(f"/home/gdp/i4m/data/raw/{month1}/{day1}/")    
        self.sentiment = Path(f"/home/gdp/i4m/data/sentiment/sentiment/{year1}/{month1}/{day1}/")    
        self.single_news = Path(f"/home/gdp/i4m/data/sentiment/single_news/{year1}/{month1}/{day1}/")    
        self.saveRec = Path(f"/home/gdp/i4m/data/recommenders/{year1}/{month1}/{day1}/")
        self.finviz_save = Path(f"/home/gdp/i4m/data/finviz/{month1}/{day1}/finviz.csv")
        
        if not self.saveRaw.exists():
            self.saveRaw.mkdir(parents=True)
    
        if not self.sentiment.exists():
            self.sentiment.mkdir(parents=True)   
                   
        if not self.single_news.exists():
            self.single_news.mkdir(parents=True)           
        
        if not self.saveRec.exists():
            self.saveRec.mkdir(parents=True)



    def source_data_1(self):
        data = pd.read_csv(self.finviz_save).round(4).fillna(0.0)
        data = f0.clean_sort(data)
        data.to_pickle(self.saveRec / "a1_finviz.pkl")
        st.write(f"[1] Bulk Data Total Stock Count: {data.shape}")
        st.dataframe(data)
        return data


    def analyst_recom_2(self):
        data = self.source_data_1()
        data2 = data[data['analyst_recom'] != 0.0]
        data3 = data2[data2['analyst_recom'] <= 2.5]
        
        st.write(f"[2] Stocks With an Average Analyst Recom: {data2.shape}")
        st.dataframe(data2)    
        st.write(f"[3] Stocks With an Average Analyst Recom == of BUY(2) or STRONG-BUY(1) == (x<2.5): {data3.shape}")   
        st.dataframe(data3)
        return data3


    def technicals_minervini_a(self, data, c=3):
        d1 = data[(data['price'] > data['sma_20'])]
        d2 = data[(data['price'] > data['sma_50'])]
        d3 = data[(data['price'] > data['sma_200'])]
        d4 = data[(data['sma_20'] > data['sma_50'])]
        d5 = data[(data['sma_50'] > data['sma_200'])]
        d6 = data[(data['price'] / data['low_50_day']) > 1.3]
        d7 = data[(data['price'] / data['high_50_day']) > 0.7]   
        script_lst = [
            f"Price > sma_20: {d1.shape}",
            f"Price > sma_50: {d2.shape}",
            f"Price > sma_200: {d3.shape}",
            f"sma_20 > sma_50: {d4.shape}",
            f"sma_50 > sma_200: {d5.shape}",
            f"(price / low_50_day) > 1.3: {d6.shape}",
            f"(price / high_50_day) > 0.7: {d7.shape}",
        ]         
        for i in script_lst:
            c += 1
            st.write(f"[{int(c)}] " + i)
        return data
    
    
    def technicals_minervini_b(self, data, c=3):
        # c += 1
        # data = data[(data['price'] > data['sma_20'])]
        # st.write(f"[{int(c)}] Price > sma_20: {data.shape}")
        
        # c += 1
        # data = data[(data['price'] > data['sma_50'])]
        # st.write(f"[{int(c)}] Price > sma_50: {data.shape}")
        
        # c += 1   
        # data = data[(data['price'] > data['sma_200'])]
        # st.write(f"[{int(c)}] Price > sma_200: {data.shape}")
        
        c += 1   
        data = data[(data['sma_20'] > data['sma_50'])]
        st.write(f"[{int(c)}] sma_20 > sma_50: {data.shape}")
        
        # c += 1   
        # data = data[(data['sma_50'] > data['sma_200'])]
        # st.write(f"[{int(c)}] sma_50 > sma_200: {data.shape}")
        
        # c += 1
        # data = data[(data['price'] / data['low_50_day']) > 1.3]
        # st.write(f"[{int(c)}] (price / low_50_day) > 1.3: {data.shape}")         
        
        # c += 1
        # data = data[(data['price'] / data['high_50_day']) > 0.7]
        # st.write(f"[{int(c)}] (price / high_50_day) > 0.7: {data.shape}", "\n")         
        
        return data    
    
    
    
    
if __name__ == '__main__':
    day = '2022-06-16'
    df = Part_123(day).analyst_recom_2()