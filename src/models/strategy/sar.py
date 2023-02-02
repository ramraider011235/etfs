import streamlit as st
import pandas as pd
import yfinance as yf
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


            
class PSAR:
    
    
  def __init__(self, init_af=0.02, max_af=0.2, af_step=0.02):
    self.max_af = max_af
    self.init_af = init_af
    self.af = init_af
    self.af_step = af_step
    self.extreme_point = None
    self.high_price_trend = []
    self.low_price_trend = []
    self.high_price_window = deque(maxlen=2)
    self.low_price_window = deque(maxlen=2)

    # Lists to track results
    self.psar_list = []
    self.af_list = []
    self.ep_list = []
    self.high_list = []
    self.low_list = []
    self.trend_list = []
    self._num_days = 0


  def calcPSAR(self, high, low):
    if self._num_days >= 3:
      psar = self._calcPSAR()
    else:
      psar = self._initPSARVals(high, low)
    psar = self._updateCurrentVals(psar, high, low)
    self._num_days += 1
    return psar


  def _initPSARVals(self, high, low):
    if len(self.low_price_window) <= 1:
      self.trend = None
      self.extreme_point = high
      return None
    if self.high_price_window[0] < self.high_price_window[1]:
      self.trend = 1
      psar = min(self.low_price_window)
      self.extreme_point = max(self.high_price_window)
    else: 
      self.trend = 0
      psar = max(self.high_price_window)
      self.extreme_point = min(self.low_price_window)
    return psar


  def _calcPSAR(self):
    prev_psar = self.psar_list[-1]
    if self.trend == 1: # Up
      psar = prev_psar + self.af * (self.extreme_point - prev_psar)
      psar = min(psar, min(self.low_price_window))
    else:
      psar = prev_psar - self.af * (prev_psar - self.extreme_point)
      psar = max(psar, max(self.high_price_window))
    return psar


  def _updateCurrentVals(self, psar, high, low):
    if self.trend == 1:
      self.high_price_trend.append(high)
    elif self.trend == 0:
      self.low_price_trend.append(low)
    psar = self._trendReversal(psar, high, low)
    self.psar_list.append(psar)
    self.af_list.append(self.af)
    self.ep_list.append(self.extreme_point)
    self.high_list.append(high)
    self.low_list.append(low)
    self.high_price_window.append(high)
    self.low_price_window.append(low)
    self.trend_list.append(self.trend)
    return psar


  def _trendReversal(self, psar, high, low):
    reversal = False

    if self.trend == 1 and psar > low:
      self.trend = 0
      psar = max(self.high_price_trend)
      self.extreme_point = low
      reversal = True

    elif self.trend == 0 and psar < high:
      self.trend = 1
      psar = min(self.low_price_trend)
      self.extreme_point = high
      reversal = True

    if reversal:
      self.af = self.init_af
      self.high_price_trend.clear()
      self.low_price_trend.clear()

    else:

        if high > self.extreme_point and self.trend == 1:
          self.af = min(self.af + self.af_step, self.max_af)
          self.extreme_point = high

        elif low < self.extreme_point and self.trend == 0:
          self.af = min(self.af + self.af_step, self.max_af)
          self.extreme_point = low

    return psar



class SAR_Build():
    

    def __init__(self, ticker, day1, graph_it=False, cc=1, ccc=1):
        self.ticker = ticker
        self.d1 = str(day1)[:10]
        self.graph = graph_it
        self.cc = cc
        self.ccc = ccc
        
        
    def build1(self, data):
        indic = PSAR()
        # data = data.iloc[-30:]
        data['PSAR'] = data.apply(lambda x: indic.calcPSAR(x['high'], x['low']), axis=1)
        data['EP'] = indic.ep_list
        data['Trend'] = indic.trend_list
        data['AF'] = indic.af_list

        indic._calcPSAR()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        psar_bull = data.loc[data['Trend']==1]['PSAR']
        psar_bear = data.loc[data['Trend']==0]['PSAR']
        buy_sigs = data.loc[data['Trend'].diff()==1]['adjclose']
        short_sigs = data.loc[data['Trend'].diff()==-1]['adjclose']

        self.graph = True
        if self.graph == True:
            # data = data.iloc[-30:]
            # buy_sigs = buy_sigs.iloc[-30:]
            # short_sigs = short_sigs.iloc[-30:]
            # psar_bull = psar_bull.iloc[-30:]
            # psar_bear = psar_bear.iloc[-30:]
            
            fig, ax = plt.subplots(figsize=(20,7))
            plt.plot(
              data['adjclose'], 
              label='adjclose', 
              linewidth=3, 
              zorder=0, 
              color='k'
            )
            plt.scatter(
                buy_sigs.index,
                buy_sigs, 
                color='green',#colors[2], 
                label='Buy', 
                marker='^', 
                s=250
            )
            plt.scatter(
                short_sigs.index, 
                short_sigs, 
                color='red', 
                label='Short', 
                marker='v', 
                s=250
            )
            plt.scatter(
              psar_bull.index, 
              psar_bull, 
              color=colors[0], 
              label='Up Trend', 
              s=10
            )
            plt.scatter(
              psar_bear.index, 
              psar_bear, 
              color=colors[3], 
              label='Down Trend', 
              s=10
            )
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.title(f'{self.ticker} Price and Parabolic SAR')
            plt.legend()
            plt.grid(True)
            st.pyplot(fig)


        n_l = []
        for i in range(len(data.index)):
            if data.index[i] in buy_sigs:
                n_l.append('Buy')
            elif data.index[i] in short_sigs:
                n_l.append('Sell')
            else:
                n_l.append('Hold')
        n_l2 = []
        for i in range(len(data.index)):
            if data.index[i] in psar_bull:
                n_l2.append('BULL')
            elif data.index[i] in psar_bear:
                n_l2.append('BEAR')
            else:
                n_l2.append('-')
                
        data['action'] = n_l
        data['Bear_Bull'] = n_l2      

        # st.write(data['Bear_Bull'][-1])
        # st.write(data['Bear_Bull'][-1] == 'BULL')


        if data['action'][-1] == 'Buy':
            st.metric(
                label=f"No.【{self.cc} / {self.ccc}】",
                value=f"{self.ticker}",
                delta=f"{data['Bear_Bull'][-1]} [{data['action'][-1]}]",
            )  
            return self.ticker
        
        elif data['action'][-1] == 'Sell':
            st.metric(
                label=f"No.【{self.cc} / {self.ccc}】",
                value=f"{self.ticker}",
                delta=f"- {data['Bear_Bull'][-1]} [{data['action'][-1]}]",
            )  
            return 

        elif data['action'][-1] == 'Hold':

            if data['Bear_Bull'][-1] == 'BULL':
                st.metric(
                    label=f"No.【{self.cc} / {self.ccc}】",
                    value=f"{self.ticker}",
                    delta=f"{data['Bear_Bull'][-1]} [{data['action'][-1]}]",
                ) 
                return self.ticker

            if data['Bear_Bull'][-1] == 'BEAR':
                st.metric(
                    label=f"No.【{self.cc} / {self.ccc}】",
                    value=f"{self.ticker}",
                    delta=f"- {data['Bear_Bull'][-1]} [{data['action'][-1]}]",
                )        
                return 
                  
        else:
            st.metric(
                label=f"No.【{self.cc} / {self.ccc}】",
                value=f"{self.ticker}",
                delta=f"{data['Bear_Bull'][-1]} [{data['action'][-1]}]",
            )  
            return #self.ticker


        # if data['Bear_Bull'][-1] == 'BULL':
        #     st.metric(
        #         label=f"No.【{self.cc} / {self.ccc}】",
        #         value=f"{self.ticker}",
        #         delta=f"{data['Bear_Bull'][-1]}",
        #     )
        #     st.dataframe(data.iloc[-1:])
        #     st.write(f"{'__________'}")
        #     return self.ticker
        
        # elif data['Bear_Bull'][-1] == 'BEAR':
        #     st.metric(
        #         label=f"No.【{self.cc} / {self.ccc}】",
        #         value=f"{self.ticker}",
        #         delta=f"- {data['Bear_Bull'][-1]}",
        #     )  
        #     st.dataframe(data.iloc[-1:])
        #     st.write(f"{'__________'}")
        #     return 
          
        # else:
        #     st.metric(
        #         label=f"No.【{self.cc} / {self.ccc}】",
        #         value=f"{self.ticker}",
        #         delta=f"- {data['Bear_Bull'][-1]}",
        #     )  
        #     st.dataframe(data.iloc[-1:])
        #     st.write(f"{'__________'}")
        #     return               
              