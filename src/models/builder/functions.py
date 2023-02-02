from numpy import var
import yfinance as yf
import pandas as pd
from yahooquery import Ticker
import streamlit as st
from datetime import date, datetime
from pathlib import Path
from os.path import exists
import statistics as stats



from numpy import var
import yfinance as yf
import pandas as pd
from yahooquery import Ticker
import streamlit as st
from datetime import date, datetime
from pathlib import Path
from os.path import exists
import statistics as stats



def recommended_stocks(name_lst, report_date="2021-07-13"):
    edate = datetime.now()
    report_date = st.sidebar.date_input(
        label="> recommender date:",
        value=date(2021, 7, 14),
        min_value=date(2021, 7, 14),
        max_value=edate,
        key="date to run proof",
        help="Select a date in the range between 2021.07.15 - 2021.08.26. \
            This date will be the date the recommender model was run and we \
                will use the resulting tickers for our proof",
        )
    saveReport_port_results = Path(f"reports/port_results/{str(report_date)[:7]}/{str(report_date)[:10]}/")
    r_stocks = list(pd.read_csv(saveReport_port_results / f"{name_lst}.csv")["ticker"])
    return r_stocks, report_date


def recommended_stocks_2(name_lst, report_date):
    r_stocks = list(pd.read_csv(f"reports/port_results/{str(report_date)[:7]}/{str(report_date)[:10]}/{name_lst}.csv")["ticker"])
    st.write(f" - Below Are The Selected Stocks - total stocks = [{len(r_stocks)}]")
    st.text(r_stocks)
    st.sidebar.write(" *" * 25)
    return r_stocks



def display_as_percent(val):
    return str(round(val * 100, 1)) + "%"


def company_longName(ticker):
    d = Ticker(ticker).quote_type
    return list(d.values())[0]["longName"]


def clean69(data, target1):
    d_lst_1, d_lst_2 = data[target1], data['price']
    aaa_lst = []
    for r in range(len(d_lst_1)):
        r1 = 1 + d_lst_1[r]
        x = d_lst_2[r] / r1
        aaa_lst.append(x)
    return aaa_lst        
    
    
def col_clean_69(data, a_lst):
    try:
        for a in a_lst:
            val = clean69(data, a)
            del data[a]
            data[a] = val
        return data
    except Exception:
        pass       
    
    
def clean(list1):
    temp = []
    for i in list1:
        if type(i) == int:
            temp.append(float(i))
        if type(i) == float:
            temp.append(i)
        elif type(i) == str:
            temp.append(float(i[:-1]))
    return temp


def clean_percent(list1):
    temp = []
    for i in list1:
        if type(i) == int:
            x = float(i)
            temp.append(x/100)
        if type(i) == float:
            temp.append(i/100)
        elif type(i) == str:
            x = float(i[:-1])
            temp.append(x/100)
    return temp


def clean_columns(data):
    data.columns = [x.lower() for x in data.columns]
    data.columns = [x.replace(" ", "_") for x in data.columns]
    data.columns = [x.replace("(", "") for x in data.columns]
    data.columns = [x.replace(")", "") for x in data.columns]
    data.columns = [x.replace("/", "") for x in data.columns]
    data.columns = [x.replace("-", "_") for x in data.columns]
    data.columns = [x.replace(".", "") for x in data.columns]
    return data


def col_clean_num_per(d1, cols_lst, percent_list, number_list, string_list):
    try:
        data = d1.copy()
    
        for x in data[percent_list]:           
            val1 = clean_percent(data[x])
            del data[x]
            data[x] = val1   
                
        for x in data[number_list]:           
            val1 = clean(data[x])
            del data[x]
            data[x] = val1
                
            # if x in string_list:
            #     val1 = clean_percent(data[x])
            #     del data[x]
            #     data[x] = val1
                
            # else:
            #     pass   
            
    except Exception:
        pass   
              
    return data    


def clean_sort(d0):    
    data = d0.copy()
   
    string_list = [
        'Ticker', 'Company', 'Sector', 'Industry', 'Country'
        ]
    
    percent_list = [
        'Dividend Yield', 'Payout Ratio', 'Insider Ownership', 'Insider Transactions', 'Institutional Ownership', 'Institutional Transactions', 
        'Float Short', 'Return on Assets', 'Return on Equity', 'Return on Investment', 'Gross Margin', 'Operating Margin', 'Profit Margin', 
        'Performance (Half Year)', 'Performance (Year)', 'Performance (YTD)', 
        'Performance (Week)', 'Performance (Month)', 'Performance (Quarter)', 
        'Volatility (Week)', 'Volatility (Month)', '52-Week Low', 'Gap',
        '20-Day Simple Moving Average', '50-Day Simple Moving Average', '200-Day Simple Moving Average', '50-Day High', '50-Day Low', '52-Week High',
        'EPS growth this year', 'EPS growth next year', 'EPS growth past 5 years', 'EPS growth next 5 years', 'Sales growth past 5 years', 
        'EPS growth quarter over quarter', 'Sales growth quarter over quarter',
    ]
    
    number_list = [
        'No.', 'Market Cap', 'P/E', 'Forward P/E', 'PEG', 'P/S', 'P/B', 'P/Cash', 'P/Free Cash Flow', 'EPS (ttm)',   
        'Current Ratio', 'Quick Ratio', 'LT Debt/Equity', 'Total Debt/Equity', 'Shares Outstanding', 'Shares Float', 'Short Ratio', 'Beta',
        'Average True Range', 'Relative Strength Index (14)', 'Analyst Recom', 'Average Volume', 'Relative Volume', 'Price', 'Volume', 'Target Price'    
    ]

    a_lst = [
        '20_day_simple_moving_average',  '50_day_simple_moving_average', '200_day_simple_moving_average',  '50_day_high',  '50_day_low', '52_week_high',  '52_week_low'
    ]        
           
    
    cols_lst = list(data.columns)
    data = clean_bad_columns_1(data)
    data = col_clean_num_per(data, cols_lst, percent_list, number_list, string_list)            
    data = clean_columns(data)
    data = col_clean_69(data, a_lst)
    data = clean_bad_columns_2(data)
    
    return data 




    
def clean_bad_columns_1(data):
    try:
        del data['Earnings Date']
    except Exception:
        pass
    try:
        del data['Change']
    except Exception:
        pass
    try:
        del data['Change from Open']
    except Exception:
        pass
    try:
        del data['IPO Date']
    except Exception:
        pass
    try:
        del data['After-Hours Close']
    except Exception:
        pass
    try:
        del data['After-Hours Change']    
    except Exception:
        pass
    return data
    
    
def clean_bad_columns_2(data):
    try:
        del data["earnings_date"]
    except Exception:
        pass
    
    try:        
        del data["ipo_date"]
    except Exception:
        pass 
           
    try:
        del data["index"]
    except Exception:
        pass
    
    try:
        del data["rank"]
    except Exception:
        pass
    
    try:
        del data["afterhours_close"]
    except Exception:
        pass
    
    try:
        del data["change"]
    except Exception:
        pass
    
    try:
        del data["afterhours_change"]
    except Exception:
        pass     
      
    try:
        del data["change_from_open"]
    except Exception:
        pass        
                                
    try:
        data = data.rename(columns={"symbol": "ticker"})
    except Exception:
        pass
    
    try:
        data = data.rename(columns={"52_week_low": "low_52_week"})
    except Exception:
        pass
    
    try:
        data = data.rename(columns={"52_week_high": "high_52_week"})
    except Exception:
        pass
    
    try:
        data = data.rename(columns={"50_day_high": "high_50_day"})
    except Exception:
        pass
    
    try:
        data = data.rename(columns={"50_day_low": "low_50_day"})
    except Exception:
        pass
    
    try:
        data = data.rename(columns={"20_day_simple_moving_average": "sma_20"})
    except Exception:
        pass
    
    try:
        data = data.rename(columns={"50_day_simple_moving_average": "sma_50"})
    except Exception:
        pass
    
    try:
        data = data.rename(columns={"200_day_simple_moving_average": "sma_200"})
    except Exception:
        pass    

    try:
        data["over_50day_low"] = data["current_price"] / data["low_50_day"]
    except Exception:
        pass
    return data    



def get_all_statistics(df1, line):
    n = len(df1)
    var1 = 'my_score'
    
    # MEAN
    my_mean = round(df1[f"{var1}"].sum() / n, 2)
    df1['deviation'] = df1[f"{var1}"] - my_mean
    df1['deviation_sq'] = df1['deviation'] ** 2

    # SQUARED DEVIATION
    deviation_sq_sum = round(df1['deviation_sq'].sum(),2)

    # VARIANCE
    variance = round(deviation_sq_sum / n,2)

    # STANDARD DEVIATION
    sd = round(variance ** 0.5,2)

    # STANDARD DEVIATION - [0.5]
    a_std_from_mean_a = round(my_mean - (sd * 0.5),2)
    a_std_from_mean_b = round(my_mean + (sd * 0.5),2)
    
    # STANDARD DEVIATION - [1]
    b_std_from_mean_a = round(my_mean - (sd * 1.0),2)
    b_std_from_mean_b = round(my_mean + (sd * 1.0),2)

    # STANDARD DEVIATION - [1.5]
    c_std_from_mean_a = round(my_mean - (sd * 1.5),2)
    c_std_from_mean_b = round(my_mean + (sd * 1.5),2)

    # STANDARD DEVIATION - [2]
    d_std_from_mean_a = round(my_mean - (sd * 2.0),2)
    d_std_from_mean_b = round(my_mean + (sd * 2.0),2)
    
    # STANDARD DEVIATION - [2.5]
    e_std_from_mean_a = round(my_mean - (sd * 2.5),2)
    e_std_from_mean_b = round(my_mean + (sd * 2.5),2)
    
    # STANDARD DEVIATION - [3]
    f_std_from_mean_a = round(my_mean - (sd * 3.0),2)
    f_std_from_mean_b = round(my_mean + (sd * 3.0),2)
    
    fin_lst = [
        f_std_from_mean_a,
        e_std_from_mean_a,
        d_std_from_mean_a,
        c_std_from_mean_a,
        b_std_from_mean_a,
        a_std_from_mean_a,
        my_mean,
        a_std_from_mean_b, 
        b_std_from_mean_b,
        c_std_from_mean_b,
        d_std_from_mean_b,
        e_std_from_mean_b,
        f_std_from_mean_b,
        ]
    
    return fin_lst


def get_stats1(df1, choice_1, style):
    # MEAN - my_mean = round(df1['my_score'].sum() / len(df1),4)
    my_mean = df1['my_score'].mean()
    
    # DEVIATION
    df1['deviation'] = df1['my_score'] - my_mean
    
    # DEVIATION**2
    df1['deviation_sq'] = df1['deviation'] ** 2
    
    # VARIANCE - Variance = round((df1['deviation_sq'].sum()) / len(df1),4)
    variance = stats.variance(df1['my_score'])
    
    # STANDARD DEVIATION - # sd = round(variance ** 0.5    ,4)
    sd = stats.stdev(df1['my_score'])
    
    if style == 'Minimum':
        res007 = round(my_mean + (sd * choice_1), 4)
        return res007, my_mean, variance, sd
    
    elif style == 'Range':    
        res006 = round(my_mean + (sd * choice_1[0]),2)
        res007 = round(my_mean + (sd * choice_1[1]),2)
        return res006, res007, my_mean, variance, sd
