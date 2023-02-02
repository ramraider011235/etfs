import pandas as pd
from pathlib import Path

import src.tools.lists as l0


class Builder_5(object):


    def __init__(self, day1):
        month1 = str(day1)[:7]
        year1 = str(day1)[:4]
        self.saveRec = Path(f"/home/gdp/hot_box/i4m/data/recommenders/{year1}/{month1}/{day1}/")
        if not self.saveRec.exists():
            self.saveRec.mkdir(parents=True)  
        self.rec_final_01 = pd.DataFrame(pd.read_pickle(self.saveRec / "recommender_01_return_dataFrame.pkl"))
        self.rec_final_02 = pd.DataFrame(pd.read_pickle(self.saveRec / "recommender_02_return_dataFrame.pkl"))
        self.rec_final_03 = pd.DataFrame(pd.read_pickle(self.saveRec / "recommender_03_return_dataFrame.pkl"))
        self.rec_final_04 = pd.DataFrame(pd.read_pickle(self.saveRec / "recommender_04_return_dataFrame.pkl"))


    def run_mod_5(self):
        df = pd.DataFrame(self.rec_final_04)
        col_1 = df.pop('company')
        col_2 = df.pop('ticker')        
        col_3 = df.pop('my_score')
        col_4 = df.pop('sentiment_score')
        col_5 = df.pop('rs_rating')
        col_7 = df.pop('analyst_recom')                    
        col_8 = df.pop('returns_multiple')
        col_9 = df.pop('price')
        col_10 = df.pop('target_price')
        df.insert(0, 'target_price', col_10)   
        df.insert(0, 'price', col_9)   
        df.insert(0, 'returns_multiple', col_8)           
        df.insert(0, 'analyst_recom', col_7)                
        df.insert(0, 'rs_rating', col_5)           
        df.insert(0, 'sentiment_score', col_4)     
        df.insert(0, 'my_score', col_3)  
        df.insert(0, 'ticker', col_2)
        df.insert(0, 'company', col_1)
        df_5 = pd.DataFrame(df).reset_index().round(2).sort_values('my_score', ascending=False).fillna(0.0)
        del df_5['index']
        df_5.to_pickle(self.saveRec / "recommender_05_return_dataFrame.pkl")
        print(df_5.shape)
        return df_5             
