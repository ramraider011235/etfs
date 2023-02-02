from yahooquery import Ticker
import statistics as stats
import streamlit as st
import pandas as pd
from os.path import exists
from pathlib import Path
import pickle5 as pickle
import numpy as np





a = [
    # 'no', 
    'ticker', 'company', 'sector', 'industry', 'country', 
    'insider_ownership', 'insider_transactions',
    'institutional_ownership', 'institutional_transactions',     
    'market_cap', 'shares_outstanding', 'shares_float', 'float_short',     
    'volume', 'average_volume', 'relative_volume',  'short_ratio', 'beta',    
    'dividend_yield', 'payout_ratio',  'return_on_assets', 'return_on_equity', 'return_on_investment',    
    'gross_margin', 'operating_margin', 'profit_margin',    
    'pe', 'forward_pe', 'peg', 'ps', 'pb', 'pcash', 'pfree_cash_flow', 'eps_ttm',        
    'current_ratio', 'quick_ratio', 'lt_debtequity', 'total_debtequity', 'gap',        
    'performance_week', 'performance_month', 'performance_quarter', 'performance_half_year', 'performance_year', 'performance_ytd',    
    'eps_growth_quarter_over_quarter', 'eps_growth_this_year', 'eps_growth_next_year', 'eps_growth_past_5_years', 'eps_growth_next_5_years',     
    'sales_growth_quarter_over_quarter', 'sales_growth_past_5_years',             
    'volatility_week', 'volatility_month', 
    'relative_strength_index_14', 'analyst_recom',     
    'price', 'target_price',
    'sma_20_day', 'sma_20_day_2', 'sma_50_day', 'sma_50_day_2', 'sma_200_day', 'sma_200_day_2',         
    'high_52_week', 'high_52_week_2', 'low_52_week', 'low_52_week_2', 'average_true_range', 'high_50_day', 'high_50_day_2', 'low_50_day', 'low_50_day_2',        
    # 'fwd_price_1mo', 'fwd_price'
]    
string_list = [
    'Ticker', 
    'Company', 
    'Sector', 
    'Industry', 
    'Country',
    'Shortable',
    'Index',
    'Optionable',        
    ]
percent_list = [
    'Dividend Yield', 
    'Payout Ratio', 
    'Insider Ownership', 
    'Insider Transactions', 
    'Institutional Ownership', 
    'Institutional Transactions', 
    'Float Short', 
    'Return on Assets', 
    'Return on Equity', 
    'Return on Investment', 
    'Gross Margin', 
    'Operating Margin', 
    'Profit Margin', 
    'Performance (Half Year)', 
    'Performance (Year)', 
    'Performance (YTD)', 
    'Volatility (Week)', 
    'Volatility (Month)', 
    '52-Week Low', 
    'Gap',
    '20-Day Simple Moving Average', 
    '50-Day Simple Moving Average', 
    '200-Day Simple Moving Average', 
    '50-Day High', 
    '50-Day Low', 
    '52-Week High',
    'EPS growth this year', 
    'EPS growth next year', 
    'EPS growth past 5 years', 
    'EPS growth next 5 years', 
    'Sales growth past 5 years', 
    'EPS growth quarter over quarter', 
    'Sales growth quarter over quarter', 
    'Performance (Week)', 
    'Performance (Month)', 
    'Performance (Quarter)'
    ]
number_list = [
    'No.', 
    'Market Cap', 
    'P/E', 
    'Forward P/E', 
    'PEG', 
    'P/S', 
    'P/B', 
    'P/Cash', 
    'P/Free Cash Flow', 
    'EPS (ttm)',   
    'Current Ratio', 
    'Quick Ratio', 
    'LT Debt/Equity', 
    'Total Debt/Equity', 
    'Shares Outstanding', 
    'Shares Float', 
    'Short Ratio',
    'Beta','Average True Range', 
    'Relative Strength Index (14)',
    'Analyst Recom', 'Average Volume',
    'Relative Volume', 
    'Price', 
    'Volume', 
    'Target Price',
    'Employees',
    'Book/sh',
    'EPS next Q',
    'Sales',
    'Dividend',
    'Prev Close',
    'Cash/sh',
    'Income',
    'Short Interest',
    ]
a_lst = [
    '20_day_simple_moving_average',  
    '50_day_simple_moving_average',
    '200_day_simple_moving_average', 
    '50_day_high', 
    '50_day_low', 
    '52_week_high', 
    '52_week_low'
    ]

string_cols = [
    'Ticker',                             #  1 
    'Company',                            #  2 
    'Sector',                             #  3 
    'Industry',                           #  4 
    'Country',                            #  5 
    'Index',                              #  6 
    'Optionable',                         #  68
    'Shortable',                          #  69
]    
num_cols = [
'No.',                                    #  0   
'Market Cap',                             #  7   
'P/E',                                    #  8   
'Forward P/E',                            #  9   
'PEG',                                    #  10  
'P/S',                                    #  11  
'P/B',                                    #  12  
'P/Cash',                                 #  13  
'P/Free Cash Flow',                       #  14  
'Book/sh',                                #  15  
'Cash/sh',                                #  16  
'Dividend',                               #  17  
'EPS (ttm)',                              #  20  
'EPS next Q',                             #  21  
'Sales',                                  #  29  
'Income',                                 #  30  
'Shares Outstanding',                     #  31  
'Shares Float',                           #  32  
'Short Ratio',                            #  38  
'Short Interest',                         #  39  
'Current Ratio',                          #  43  
'Quick Ratio',                            #  44  
'LT Debt/Equity',                         #  45  
'Total Debt/Equity',                      #  46  
'Beta',                                   #  56  
'Average True Range',                     #  57  
'Relative Strength Index (14)',           #  67  
'Employees',                              #  70  
'Analyst Recom',                          #  73  
'Average Volume',                         #  74  
'Relative Volume',                        #  75  
'Volume',                                 #  76  
'Target Price',                           #  77  
'Prev Close',                             #  78  
'Price',                                  #  79  
'After-Hours Close',                      #  81  
]    
percent_cols = [
    'Dividend Yield',                     #  18  
    'Payout Ratio',                       #  19  
    'EPS growth this year',               #  22  
    'EPS growth next year',               #  23  
    'EPS growth past 5 years',            #  24  
    'EPS growth next 5 years',            #  25  
    'Sales growth past 5 years',          #  26  
    'Sales growth quarter over quarter',  #  27  
    'EPS growth quarter over quarter',    #  28  
    'Insider Ownership',                  #  33  
    'Insider Transactions',               #  34  
    'Institutional Ownership',            #  35  
    'Institutional Transactions',         #  36  
    'Float Short',                        #  37  
    'Return on Assets',                   #  40  
    'Return on Equity',                   #  41  
    'Return on Investment',               #  42  
    'Gross Margin',                       #  47  
    'Operating Margin',                   #  48  
    'Profit Margin',                      #  49  
    'Performance (Week)',                 #  50  
    'Performance (Month)',                #  51  
    'Performance (Quarter)',              #  52  
    'Performance (Half Year)',            #  53  
    'Performance (Year)',                 #  54  
    'Performance (YTD)',                  #  55  
    'Volatility (Week)',                  #  58  
    'Volatility (Month)',                 #  59  
    '20-Day Simple Moving Average',       #  60  
    '50-Day Simple Moving Average',       #  61  
    '200-Day Simple Moving Average',      #  62  
    '50-Day High',                        #  63  
    '50-Day Low',                         #  64  
    '52-Week High',                       #  65  
    '52-Week Low',                        #  66  
    # 'Change from Open',                   #  71  
    'Gap',                                #  72  
    # 'Change',                             #  80  
    # 'After-Hours Change',                 #  82  
]    




def display_as_percent(val):
    return str(round(val * 100, 1)) + "%"



def company_longName(ticker):
    try:
        d = Ticker(ticker).quote_type
        return list(d.values())[0]["longName"]
    except Exception as e:
        return ticker
    
    
    
def get_diff(a, b):
    return list(set(a) ^ set(b))    
    
    
    
def analyst_recom_config():
    ar = [
        5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 
        4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 
        3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 
        2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 
        1.0
        ]
    analyst_recom = list(np.arange(20.0, 101.0, 2).round())
    return dict(zip(ar, analyst_recom))       



def str_tic_lst(port_tics):
    string_tickers = ''
    for i in port_tics:
        if i != port_tics[-1]:
            string_tickers += (i+' ')
        else:
            string_tickers += (i)
    return string_tickers



def true_false(var1):
    if var1 == "Yes":
        return True
    elif var1 == "No":
        return False    



def exception_response(crossover_1, p, x, sheen_lst, crap_lst, df1):
    try:
        if x == p:
            sheen_lst.append(p)
            return sheen_lst, crap_lst, df1
        else:
            crap_lst.append(p)
            df1 = df1.drop(df1[df1["ticker"] == p].index)
            return sheen_lst, crap_lst, df1
    except Exception:    
        st.write(f'FAILURE = {crossover_1}')
        crap_lst.append(p)
        df1 = df1.drop(df1[df1["ticker"] == p].index)
        return sheen_lst, crap_lst, df1



def clean_data_columns(data):
    try:
        del data['Unnamed: 0']
    except Exception as e:
        pass    
    data.columns = [x.lower() for x in data.columns]
    data.columns = [x.replace(" ", "_") for x in data.columns]
    data.columns = [x.replace("(", "") for x in data.columns]
    data.columns = [x.replace(")", "") for x in data.columns]
    # data.columns = [x.replace("/", "") for x in data.columns]
    data.columns = [x.replace("-", "_") for x in data.columns]
    data.columns = [x.replace(".", "") for x in data.columns]
    return data        



def clean_data_1(data):
    short_list = ['Earnings Date', 'Change', 'Change from Open', 'IPO Date', 'After-Hours Close', 'After-Hours Change',]
    for s in short_list:
        try:
            del data[f"{s}"]
        except Exception:
            pass
    return data



def clean_data_2(d1, percent_list, number_list):
    
    def clean(list1):
        temp = []
        for i in list1:
            if type(i) == int:
                temp.append(float(i))
            elif type(i) == float:
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
            elif type(i) == float:
                temp.append(i/100)
            elif type(i) == str:
                x = float(i[:-1])
                temp.append(x/100)
        return temp
    
    data = d1.copy()
    # try:
    for x in data[percent_list]:           
        val1 = clean_percent(data[x])
        del data[x]
        data[x] = val1   

    for x in data[number_list]:           
        val1 = clean(data[x])
        del data[x]
        data[x] = val1
    # except Exception:
    #     pass    
    return data   



def clean_data_3(data):
    data.columns = [x.lower() for x in data.columns]
    data.columns = [x.replace(" ", "_") for x in data.columns]
    data.columns = [x.replace("(", "") for x in data.columns]
    data.columns = [x.replace(")", "") for x in data.columns]
    data.columns = [x.replace("/", "") for x in data.columns]
    data.columns = [x.replace("-", "_") for x in data.columns]
    data.columns = [x.replace(".", "") for x in data.columns]
    return data



def clean_data_4(data, a_lst):    
    
    def clean69(data, target1):
        d_lst_1, d_lst_2 = data[target1], data['price']
        aaa_lst = []
        for r in range(len(d_lst_1)):
            r1 = 1 + d_lst_1[r]
            x = d_lst_2[r] / r1
            aaa_lst.append(x)
        return aaa_lst  
    try:
        for a in a_lst:
            val = clean69(data, a)
            data[f"{a + '_2'}"] = val
        return data
    except Exception:
        pass    
    
    
    
def clean_data_5(data):    
    short_lst = ["earnings_date", "ipo_date", "rank", "index", "change", "afterhours_change", "change_from_open", "afterhours_close"]
    for s in short_lst:
        try:
            del data[f"{s}"]
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
        data = data.rename(columns={"52_week_low_2": "low_52_week_2"})
    except Exception:
        pass        
    
    try:
        data = data.rename(columns={"52_week_high": "high_52_week"})
    except Exception:
        pass    
    
    try:
        data = data.rename(columns={"52_week_high_2": "high_52_week_2"})
    except Exception:
        pass        
    
    try:
        data = data.rename(columns={"50_day_high": "high_50_day"})
    except Exception:
        pass    
    
    try:
        data = data.rename(columns={"50_day_high_2": "high_50_day_2"})
    except Exception:
        pass        
    
    try:
        data = data.rename(columns={"50_day_low": "low_50_day"})
    except Exception:
        pass    
    
    try:
        data = data.rename(columns={"50_day_low_2": "low_50_day_2"})
    except Exception:
        pass        
    
    try:
        data = data.rename(columns={"20_day_simple_moving_average": "sma_20_day"})
    except Exception:
        pass    
    
    try:
        data = data.rename(columns={"20_day_simple_moving_average_2": "sma_20_day_2"})
    except Exception:
        pass        
    
    try:
        data = data.rename(columns={"50_day_simple_moving_average": "sma_50_day"})
    except Exception:
        pass    
    
    try:
        data = data.rename(columns={"50_day_simple_moving_average_2": "sma_50_day_2"})
    except Exception:
        pass        
    
    try:
        data = data.rename(columns={"200_day_simple_moving_average": "sma_200_day"})
    except Exception:
        pass   
    
    try:
        data = data.rename(columns={"200_day_simple_moving_average_2": "sma_200_day_2"})
    except Exception:
        pass       
    
    try:
        data["over_50day_low"] = round(data["price"] / data["low_50_day"], 2)
    except Exception:
        pass
    
    return data        



def clean_sort(d0):        
    data = d0.copy()
    data = clean_data_1(data)
    data = clean_data_2(data, percent_list, number_list)            
    data = clean_data_3(data)
    data = clean_data_4(data, a_lst)
    data = clean_data_5(data)
    return data[a]



def clean_data_main(data):    
    data.style.format({
        'var1': '{:,.2f}'.format,
        'var2': '{:,.2f}'.format,
        'var3': '{:,.2%}'.format,
    })       

    data = data.fillna(0.0)
    for i in data[percent_cols]:
        try:
            temp_lst = []
            for x in data[i]:
                if type(x) == float:
                    temp_lst.append(x)
                else:
                    y = round(float(x[:-1]) / 100, 4)
                    temp_lst.append(y)        
            data[i] = temp_lst    
        except Exception as e:
            print(e)
    
    data.columns = [x.lower() for x in data.columns]
    data.columns = [x.replace(" ", "_") for x in data.columns]
    data.columns = [x.replace("(", "") for x in data.columns]
    data.columns = [x.replace(")", "") for x in data.columns]
    data.columns = [x.replace("-", "_") for x in data.columns]
    data.columns = [x.replace(".", "") for x in data.columns]     
    data = clean_data_5(data)
    
    try:
        fix_lst = [
            'sales',
            'income',
        ]
        for f in fix_lst:
            data[f] = data[f] * 1000
            
        rev_per_ee = []
        for k, v in enumerate(data['employees']):
            if v == 0.0:
                rev_per_ee.append(v)
            else:
                rev_per_ee.append(round(data['sales'][k] / data['employees'][k], 2))

        income_per_ee = []
        for k, v in enumerate(data['employees']):
            if v == 0.0:
                income_per_ee.append(v)
            else:
                income_per_ee.append(round(data['income'][k] / data['employees'][k], 2))        
                
        data['rev_per_ee'] = rev_per_ee
        data['income_per_ee'] = income_per_ee    
    except Exception as e:
        print(e)
    return data



def get_stats1(df1, choice_1, style):
    
    df1 = pd.DataFrame(df1.copy())
    
    my_mean = df1['my_score'].mean()                               # MEAN
    df1['deviation'] = df1['my_score'] - my_mean                   # DEVIATION
    df1['deviation_sq'] = df1['deviation'] ** 2                    # DEVIATION**2
    variance = round((df1['deviation_sq'].sum()) / len(df1),4)     # VARIANCE   >>>   variance = stats.variance(df1['my_score'])
    sd = round(variance ** 0.5, 4)                                 # STANDARD DEVIATION   >>>   sd = stats.stdev(df1['my_score'])
    
    print(choice_1)
    
    if style == 'Minimum':
        res007 = round(my_mean + (sd * choice_1), 4)
        return res007, my_mean, variance, sd
    
    elif style == 'Range':    
        res006 = round(my_mean + (sd * choice_1[0]),2)
        res007 = round(my_mean + (sd * choice_1[1]),2)
        return res006, res007, my_mean, variance, sd
    
    
    
def get_stats11(df1, choice_1, style):
    
    my_score_mean = df1['my_score'].mean()
    my_score_var = df1['my_score'].var()
    my_score_std = df1['my_score'].std()
    
    if style == 'Minimum':
        res007 = round(my_score_mean + (my_score_std * choice_1), 4)
        return res007, my_score_mean, my_score_var, my_score_std
    
    elif style == 'Range':    
        res006 = round(my_score_mean + (my_score_std * choice_1[0]),2)
        res007 = round(my_score_mean + (my_score_std * choice_1[1]),2)
        return res006, res007, my_score_mean, my_score_var, my_score_std    
    
    
    
    
def get_final_df(date1, data_plot):
    
    day1 = str(date1)[:10]
    month1 = str(day1)[:7]
    year1 = str(day1)[:4]    
    saveRec = Path(f"data/recommenders/{year1}/{month1}/{day1}/")
    
    if data_plot == 'A':
        if exists(saveRec / f"recommender_05_return_dataFrame.pkl"):
            path_1 = "recommender_05_return_dataFrame"
            
    if data_plot == 'B':
        if exists(saveRec / f"recommender_05_ticker_filter.pkl"):
            path_1 = "recommender_05_ticker_filter"

    open1 = (str(saveRec) + f"/{path_1}.pkl")
    with open(open1, "rb") as fh:
        data = pd.DataFrame(pickle.load(fh))

    try:
        del data['no']
    except Exception as e:
        pass
    
    try:
        del data['index']          
    except Exception as e:
        pass      
                 
    try:
        fd = pd.DataFrame(data)
        col_1 = fd.pop('company')
        col_2 = fd.pop('ticker')        
        col_3 = fd.pop('my_score')
        col_4 = fd.pop('sentiment_score')
        col_5 = fd.pop('rs_rating')
        col_6 = fd.pop('analyst_recom')       
        col_7 = fd.pop('relative_strength_index_14')
        col_8 = fd.pop('returns_multiple')
        col_9 = fd.pop('price')
        col_10 = fd.pop('target_price')
        fd.insert(0, 'target_price', col_10)   
        fd.insert(0, 'price', col_9)   
        fd.insert(0, 'returns_multiple', col_8)           
        fd.insert(0, 'relative_strength_index_14', col_7)
        fd.insert(0, 'analyst_recom', col_6)          
        fd.insert(0, 'rs_rating', col_5)           
        fd.insert(0, 'sentiment_score', col_4)     
        fd.insert(0, 'my_score', col_3)  
        fd.insert(0, 'ticker', col_2)
        fd.insert(0, 'company', col_1)
        
    except Exception as e:
        print(e)
        fd = pd.DataFrame(data.copy())
        col_1 = fd.pop('company')
        col_2 = fd.pop('ticker')        
        col_3 = fd.pop('my_score')
        col_4 = fd.pop('sentiment_score')
        col_5 = fd.pop('rs_rating')
        col_6 = fd.pop('analyst_recom')       
        col_8 = fd.pop('returns_multiple')
        col_10 = fd.pop('target_price')
        fd.insert(0, 'target_price', col_10)   
        fd.insert(0, 'returns_multiple', col_8)           
        fd.insert(0, 'analyst_recom', col_6)          
        fd.insert(0, 'rs_rating', col_5)           
        fd.insert(0, 'sentiment_score', col_4)     
        fd.insert(0, 'my_score', col_3)  
        fd.insert(0, 'ticker', col_2)
        fd.insert(0, 'company', col_1)  
        
    if data_plot == 'B':
        data = data[
            (data['current_price'] >= data['sma_150']) & 
            (data['sma_150'] >= data['sma_200']) & 
            (data['sma_200'] >= data['sma_200_20'])
            (data['current_price'] > (data['low_52_week_y'] * 1.13)) & 
            (data['current_price'] > (data['high_52_week_y'] * 0.61))
            (data['sma_20'] >= data['sma_50']) & 
            (data['sma_50'] >= data['sma_150']) & 
            (data['sma_50'] >= data['sma_200']) & 
            (data['current_price'] >= data['sma_20']) & 
            (data['current_price'] >= data['sma_50'])            
            ]        
        
    return data          
