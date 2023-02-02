import datetime as dt
import pandas as pd
from os.path import exists
from .ust import save_xml, read_rates, available_years




class Clean(object):
    
    
    def __init__(self, day1):
        self.day1 = str(day1)[:10]
        self.now = str(dt.datetime.now())[:10]
    
    
    def clean_prep(self):
        """
            - save UST yield rates to local folder for selected years
        """
        for year in available_years():
            save_xml(year, folder="/home/gdp/hot_box/i4m/data/xml", overwrite=True)    
            
            
    def clean_main(self):
        """
            - run later - force update last year (overwrites existing file)
            - read UST yield rates as pandas dataframe
            - save as single CSV file
        """
        if exists("/home/gdp/hot_box/i4m/data/xml/rates.csv"):
            data = pd.read_csv("/home/gdp/hot_box/i4m/data/xml/rates.csv", parse_dates=["date"]).set_index("date")
            

            if pd.to_datetime(self.day1) in list(data.index):
                data = data[data.index == self.day1]                
                return (data['BC_10YEAR'].iloc[-1] / 100), str(data.index[-1])[:10]
            
            else:
                save_xml(2023, folder="/home/gdp/hot_box/i4m/data/xml", overwrite=True)
                df = read_rates(start_year=2022, end_year=2023, folder="/home/gdp/hot_box/i4m/data/xml")            
                df.to_csv("/home/gdp/hot_box/i4m/data/xml/rates.csv")
                df = pd.read_csv("/home/gdp/hot_box/i4m/data/xml/rates.csv", parse_dates=["date"]).set_index("date")
                df = df[df.index == self.day1]
                return (df['BC_10YEAR'].iloc[-1] / 100), str(df.index[-1])[:10]
            
        else:
            save_xml(2023, folder="/home/gdp/hot_box/i4m/data/xml", overwrite=True)
            df = read_rates(start_year=2022, end_year=2023, folder="/home/gdp/hot_box/i4m/data/xml")            
            df.to_csv("/home/gdp/hot_box/i4m/data/xml/rates.csv")
            df = df[df.index == self.day1]
            return (df['BC_10YEAR'].iloc[-1] / 100), str(df.index[-1])[:10]