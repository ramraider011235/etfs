import pandas as pd
from pathlib import Path

import src.tools.lists as l0
import src.tools.functions as f0


class Builder_4(object):


    def __init__(self, day1):
        month1 = str(day1)[:7]
        year1 = str(day1)[:4]
        self.saveRec = Path(f"/home/gdp/hot_box/i4m/data/recommenders/{year1}/{month1}/{day1}/")
        if not self.saveRec.exists():
            self.saveRec.mkdir(parents=True)  
        self.rec_final_01 = pd.DataFrame(pd.read_pickle(self.saveRec / "recommender_01_return_dataFrame.pkl"))
        self.rec_final_02 = pd.DataFrame(pd.read_pickle(self.saveRec / "recommender_02_return_dataFrame.pkl"))
        self.rec_final_03 = pd.DataFrame(pd.read_pickle(self.saveRec / "recommender_03_return_dataFrame.pkl"))


    def fix_rec_01(self):
        self.rec_final_01.columns = [x.lower() for x in self.rec_final_01.columns]
        d1 = l0.analyst_recom_config()
        adj__analyst_lst = []
        for i in self.rec_final_01["analyst_recom"]:
            for key, val in d1.items():
                if i == key:
                    adj__analyst_lst.append(val)
        self.rec_final_01["analyst_recom"] = adj__analyst_lst            
        return self.rec_final_01


    def fix_rec_02(self):
        self.rec_final_02.columns = [x.lower() for x in self.rec_final_02.columns]
        del self.rec_final_02['sma_50']
        del self.rec_final_02['sma_200']
        del self.rec_final_02['low_52_week']
        del self.rec_final_02['high_52_week']
        self.rec_final_02 = self.rec_final_02.round(2)
        return self.rec_final_02
    

    def fix_rec_03(self):
        self.rec_final_03.columns = [x.lower() for x in self.rec_final_03.columns]
        return self.rec_final_03   
    

    def merge_dataframes(self, rec_01_0, rec_02_0, rec_03_0):
        rec_01 = pd.DataFrame(rec_01_0[rec_01_0["ticker"].isin(list(rec_03_0["ticker"]))])
        rec_02 = pd.DataFrame(rec_02_0[rec_02_0["ticker"].isin(list(rec_03_0["ticker"]))])
        rec_03 = pd.DataFrame(rec_03_0.copy())
        a = pd.DataFrame(rec_01.merge(rec_02, how="inner", on="ticker"))
        b = a.merge(rec_03, how="inner", on="ticker")   
        final_df = pd.DataFrame(b.copy())
        return final_df
        
    
    def run_mod_4(self):
        rec_final_01_1 = self.fix_rec_01()
        rec_final_02_1 = self.fix_rec_02()
        rec_final_03_1 = self.fix_rec_03()
        final_df = pd.DataFrame(self.merge_dataframes(rec_final_01_1, rec_final_02_1, rec_final_03_1))
        final_df["my_score"] = (((final_df["analyst_recom"]) + (final_df["rs_rating"]) + (final_df["sentiment_score"])) / 3)
        final_df.to_pickle(self.saveRec / "recommender_04_return_dataFrame.pkl")
        print(f"[4] Recommender Stage #04 - [Total Passed == {len(final_df['ticker'])}]")
        print(final_df.shape)
        return final_df
