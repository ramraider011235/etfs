import pandas as pd
import numpy as np
from itertools import product
import talib as ta
import streamlit as st



class Optimal_Double_SMA(object):
    
    
    def __init__(self, tic):
        self.tic = tic


    def grab_data(self, data, moving_avg1):
        try:
            raw = pd.DataFrame(data[f"{self.tic}"].copy())
        except Exception:
            raw = pd.DataFrame(data['adjclose'].copy())
            raw.columns = [x.lower() for x in raw.columns]
            raw.columns = [self.tic]
        
        results = pd.DataFrame()     
        sma1 = range(2, 61, 1)
        sma2 = range(2, 122, 2)        
           
        for SMA1, SMA2 in product(sma1, sma2):
            data1 = pd.DataFrame(raw.copy())
            data1.dropna(inplace=True)
            data1["Returns"] = np.log(data1[self.tic] / data1[self.tic].shift(1))
            
            if moving_avg1 == "SMA":
                # Create a short & long SMA (Simple Moving Average)
                data1[f"{moving_avg1}_short"] = ta.SMA(data1[self.tic], SMA1)            
                data1[f"{moving_avg1}_long"] = ta.SMA(data1[self.tic], SMA2)
            
            elif moving_avg1 == "EMA":
                # Create a short & long SMA EMA (Exponential Moving Average)
                data1[f"{moving_avg1}_short"] = ta.EMA(data1[self.tic], SMA1)            
                data1[f"{moving_avg1}_long"] = ta.EMA(data1[self.tic], SMA2)                
                            
            elif moving_avg1 == "WMA":
                # Create a short & long SMA simple WMA (Weighted Moving Average)                     
                data1[f"{moving_avg1}_short"] = ta.WMA(data1[self.tic], SMA1)
                data1[f"{moving_avg1}_long"] = ta.WMA(data1[self.tic], SMA2)
                
            elif moving_avg1 == "DEMA":
                # Create a short & long SMA EMA (Exponential Moving Average)
                data1[f"{moving_avg1}_short"] = ta.DEMA(data1[self.tic], SMA1)            
                data1[f"{moving_avg1}_long"] = ta.DEMA(data1[self.tic], SMA2)                
                            
            elif moving_avg1 == "TEMA":
                # Create a short & long SMA simple WMA (Triple Exponential Moving Average)                     
                data1[f"{moving_avg1}_short"] = ta.TEMA(data1[self.tic], SMA1)
                data1[f"{moving_avg1}_long"] = ta.TEMA(data1[self.tic], SMA2) 
                
            elif moving_avg1 == "TRIMA":
                # Create a short & long SMA simple WMA (Triangular Moving Average)                     
                data1[f"{moving_avg1}_short"] = ta.TRIMA(data1[self.tic], SMA1)
                data1[f"{moving_avg1}_long"] = ta.TRIMA(data1[self.tic], SMA2)                                                                           
                
            
            data1["Position"] = np.where(data1[f"{moving_avg1}_short"] > data1[f"{moving_avg1}_long"], 1, -1)
            data1["Strategy"] = data1["Position"].shift(1) * data1["Returns"]
            data1.dropna(inplace=True)
            perf = np.exp(data1[["Returns", "Strategy"]].sum())
            results = results.append(
                pd.DataFrame(
                    {
                        f"{moving_avg1}_short": SMA1,
                        f"{moving_avg1}_long": SMA2,
                        "MARKET(%)": perf["Returns"],
                        "STRATEGY(%)": perf["Strategy"],
                        "OUT": (perf["Strategy"] - perf["Returns"]),
                    },
                    index=[0],
                ),
                ignore_index=True,
            )
            
        results = results.loc[results[f"{moving_avg1}_short"] < results[f"{moving_avg1}_long"]]
        results = (results.sort_values("OUT", ascending=False).reset_index(drop=True).head(10))

        st.write("__________")
        st.dataframe(results.iloc[:1])
        
        S = results[f"{moving_avg1}_short"][0]
        L = results[f"{moving_avg1}_long"][0]
        return S, L