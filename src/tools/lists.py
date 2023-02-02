import numpy as np
import pandas as pd



names_of_screeners = [
    "-",
    "Day_Gainer_Stocks",
    "Day_Loser_Stocks",
    "Most_Active_Stocks",
    "Trending_Tickers",
    "Most_Shorted_Stocks",
    "Undervalued_Large_Cap_Stocks",
    "Undervalued_Growth_Stocks",
    "Growth_Technology_Stocks",
    "Aggressive_Small_Cap_Stocks",
    "Small_Cap_Gainer_Stocks",
    "Top_Crypto_Securities",
    "Top_Mutual_Funds",
    "Portfolio_Anchor_Securities",
    "Solid_Large_Cap_Growth_Funds",
    "Solid_Mid_Cap_Growth_Funds",
    "Conservative_Foreign_Funds",
    "High_Yield_Bond_Funds",
]

stock_name_list = [
    "DOW_Symbols",
    "S&P100_Symbols",
    # "S&P400_Symbols",
    "S&P500_Symbols",
    # "S&P600_Symbols",
    # "NASDAQ_Symbols",
    # "Finviz_Symbols",
    # "Other_Symbols",
    # "Fool_Symbols",
    # "Oxford_Symbols",
    "Day_Gainer_Symbols",
    "Day_Losers_Symbols",
    "Day_Most_Active_Symbols",
    "Trending_Symbols",
    "MostShorted_Symbols",
    "Undervalued_Large_Cap_Symbols",
    "Undervalued_Growth_Symbols",
    "Growth_Technology_Symbols",
    "Aggressive_Small_Cap_Symbols",
    "Small_Cap_Gainers_Symbols",
    "Top_Crypto_Symbols",
    "Top_Mutual_Fund_Symbols",
    "Portfolio_Anchor_Symbols",
    "Solid_Growth_Funds_Symbols",
    "Solid_Mid_Cap_Growth_Funds_Symbols",
    "Conservative_Foreign_Funds_Symbols",
    "High_Yield_Bond_Symbols",
]

major_indicies = [
    "^GSPC", 
    "^MID", 
    "^OEX", 
    "^RUT"
    ]

major_index_names = [
    "S&P 500", 
    "S&P 400", 
    "S&P 100", 
    "Russell 2000"
    ]

general_pages = [
    "Home",
    "Screener",
    "Strategy",
    "Backtesting",
    "Forecasting",
    "Portfolio",
    "Analysis",
]

feature_strategy = [
    "-Select-Model-",
    "Moving-Average - SMA & EMA",
    "Optimal SMA",
    "OverBought & OverSold",
    "Support & Resistance Lines",
    "Strategy II",
    "Indicators",
]

namer_lst = [
    "max_sharpe_df_1",
    "min_vol_df_1",
    "max_sharpe_df_2",
    "min_vol_df_2",
    "max_sharpe_df_3",
    "min_vol_df_3",
]


def live_dates_model():
    #  |  'MONDAY'  |  'TUESDAY'  | 'WEDNESDAY' | 'THURSDAY'  | 'FRIDAY'    |
    july_2021 = [
                                    '2021-07-14', '2021-07-15', '2021-07-16',
        '2021-07-19', '2021-07-20', '2021-07-21', '2021-07-22', '2021-07-23',
        '2021-07-26', '2021-07-27',               '2021-07-29', '2021-07-30',
    ]
    august_2021 = [
        '2021-08-02', '2021-08-03', '2021-08-04', '2021-08-05', '2021-08-06',
        '2021-08-09', '2021-08-10',                             '2021-08-13', 
        '2021-08-16',                                           '2021-08-20',
        '2021-08-23',                             '2021-08-26', '2021-08-27',
                      '2021-08-31',
    ]    
    september_2021 = [
                                                  '2021-09-02', '2021-09-03',
                                    '2021-09-08', '2021-09-09', '2021-09-10',
        '2021-09-13', '2021-09-14', '2021-09-15', '2021-09-16', '2021-09-17',
                      '2021-09-21', '2021-09-22', '2021-09-23', '2021-09-24',
        '2021-09-27', '2021-09-28', 
    ]        
    october_2021 = [
                                                                '2021-10-01',
        '2021-10-04', '2021-10-05', '2021-10-06', '2021-10-07', '2021-10-08',
        '2021-10-11',               '2021-10-13', '2021-10-14', '2021-10-15',
        '2021-10-18', '2021-10-19', '2021-10-20', '2021-10-21', '2021-10-22', 
        '2021-10-25',               '2021-10-27', '2021-10-28', '2021-10-29', 
    ]
    november_2021 = [
        '2021-11-01', '2021-11-02', '2021-11-03',               '2021-11-05',
                                    '2021-11-10', '2021-11-11', '2021-11-12',
        '2021-11-15', '2021-11-16', '2021-11-17', '2021-11-18', '2021-11-19',
                      '2021-11-23', '2021-11-24', '2021-11-25', '2021-11-26', 
    ]
    december_2021 = [
                                                  '2021-12-02', '2021-12-03',
        '2021-12-06', '2021-12-07', '2021-12-08', '2021-12-09', '2021-12-10',
        '2021-12-13', '2021-12-14', '2021-12-15', '2021-12-16', '2021-12-17',
        '2021-12-20', '2021-12-21',
    ]
    january_2022 = [
        '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07',
        '2022-01-10', '2022-01-11', '2022-01-12', '2022-01-13',
                      '2022-01-18', '2022-01-19', '2022-01-20', '2022-01-21',
                                    '2022-01-26', '2022-01-27', '2022-01-28',
    ]
    february_2022 = [
                                                                '2022-02-04', 
        '2022-02-07',               '2022-02-09', '2022-02-10', '2022-02-11',
        '2022-02-14', '2022-02-15', '2022-02-16',               '2022-02-18',
                      '2022-02-22', '2022-02-23', '2022-02-24', '2022-02-25',
    ]    
    march_2022 = [
                                                                '2022-03-04',
        '2022-03-07', '2022-03-08', '2022-03-09', '2022-03-10', '2022-03-11',
        '2022-03-14', '2022-03-15',                             '2022-03-18',
        '2022-03-21', '2022-03-22', '2022-03-23', '2022-03-24', '2022-03-25',
                      '2022-03-29',               '2022-03-31',
    ]
    april_2022 = [               
                      '2022-04-05',               # ---SICK---, 
                      '2022-04-12',                             
                      '2022-04-19',               # ---SICK---,
                      '2022-04-26',               '2022-04-28', 
    ]
    may_2022 = [
                      '2022-05-03',               '2022-05-05',
                      '2022-05-10',                
                                                  '2022-05-19',
                                                  '2022-05-26', 
                      '2022-05-31',
    ]
    june_2022 = [
        '2022-06-06',                             '2022-06-09',
                      '2022-06-14',               '2022-06-16',
                      '2022-06-21',               '2022-06-23', '2022-06-24',
                      '2022-06-28',               '2022-06-30',
    ]
    july_2022 = [
                      '2022-07-05',               '2022-07-07', 
                                                  '2022-07-13', '2022-07-14',
                      '2022-07-19',               '2022-07-21',
                      '2022-07-26',               '2022-07-28',
    ]
    august_2022 = [
                      '2022-08-02',               '2022-08-04', '2022-08-05',
                      '2022-08-09',               '2022-08-11',
                      '2022-08-16',               '2022-08-18', '2022-08-19',
                      '2022-08-23',               '2022-08-25',
                      '2022-08-30',             
    ]
    september_2022 = [
                                                  '2022-09-01',
                      '2022-09-06',               '2022-09-08',
                      '2022-09-13',               '2022-09-15',
                      '2022-09-20',               '2022-09-22',
                      '2022-09-27',               '2022-09-29',
    ]    
    october_2022 = [
                      '2022-10-04', '2022-10-05', '2022-10-06',
                      '2022-10-11',               '2022-10-13',
                      
                      '2022-10-25',               '2022-10-27',
        '2022-10-31', 
    ]       
    november_2022 = [
                      '2022-11-01',                            '2022-11-04',
                      '2022-11-08',               '2022-11-10',
                      '2022-11-15', '2022-11-16', '2022-11-17',
                      '2022-11-22', '2022-11-23', '2022-11-24', '2022-11-25',
                      '2022-11-29', '2022-11-30',
    ]
    december_2022 = [
                                                  '2022-12-01', 
                      '2022-12-06', '2022-12-07', '2022-12-08', 
                      '2022-12-13', '2022-12-14', '2022-12-15', 
                      '2022-12-20', '2022-12-21', '2022-12-22', 
                      '2022-12-27', '2022-12-28', '2022-12-29', 
    ]    
    january_2023 = [
                      '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06',
                      '2023-01-10',               '2023-01-12',
    ]    
    live_months = [
        july_2021, august_2021, september_2021, october_2021, november_2021, december_2021,
        january_2022, february_2022, march_2022, april_2022, may_2022, june_2022, july_2022, august_2022, september_2022, october_2022, november_2022, december_2022,
        january_2023
        ]
    days = []
    for l in live_months:
        days += l
    return days[::-1]



def live_dates_updater():
    lst = [
                      '2022-12-13',               
                                    '2022-12-28', '2022-12-29',
                      '2023-01-03', '2023-01-04', '2023-01-05',
                      '2023-01-10',               '2023-01-12',
        
                      '2023-01-24',               '2023-01-26',
        '2023-01-30', '2023-01-31',
        ]
    return lst[::-1] 
    


def live_dates_model_2():
    #  |  'MONDAY'  |  'TUESDAY'  | 'WEDNESDAY' | 'THURSDAY'  | 'FRIDAY'    |
    march_2022 = [
                                                  '2022-03-24',
                      '2022-03-29',               '2022-03-31',
    ]
    april_2022 = [               
                      '2022-04-05',               # ---SICK---, 
                      '2022-04-12',                             
                      '2022-04-19',               # ---SICK---,
                      '2022-04-26',               '2022-04-28', 
    ]
    may_2022 = [
                      '2022-05-03',               '2022-05-05',
                      '2022-05-10',                
                                                  '2022-05-19',
                                                  '2022-05-26', 
                      '2022-05-31',
    ]
    june_2022 = [                                           
                                                  '2022-06-09',
                      '2022-06-14',               '2022-06-16',
                      '2022-06-21',               '2022-06-23', 
                      '2022-06-28',               '2022-06-30',
    ]
    july_2022 = [
                                    '2022-07-13', '2022-07-14',
                      '2022-07-19',               '2022-07-21',
                      '2022-07-26',               '2022-07-28',
    ]
    august_2022 = [
                      '2022-08-02',               '2022-08-04',
                                                  '2022-08-11',
                      '2022-08-16',               '2022-08-18', 
                      '2022-08-23',               '2022-08-25',
                      '2022-08-30',
    ]
    september_2022 = [
                                                  '2022-09-01',
                      '2022-09-06',               '2022-09-08',
                      '2022-09-13',               '2022-09-15',
                      '2022-09-27',               '2022-09-29',
    ]    
    october_2022 = [
                      '2022-10-04',               '2022-10-06',
                      '2022-10-11',               '2022-10-13',
                      '2022-10-18',
                      '2022-10-25',               '2022-10-27',
    ]       
    november_2022 = [
                      '2022-11-08',               '2022-11-10',
                      '2022-11-15', '2022-11-16', '2022-11-17',
                      '2022-11-22',               '2022-11-24', 
                      '2022-11-29', '2022-11-30',
    ]
    december_2022 = [
                                                  '2022-12-01', 
                      '2022-12-06', '2022-12-07', '2022-12-08', 
                      '2022-12-13', '2022-12-14', '2022-12-15', 
                      '2022-12-20', '2022-12-21', '2022-12-22', 
                                    '2022-12-28', '2022-12-29', 
    ]    
    january_2023 = [
                      '2023-01-03', '2023-01-04', '2023-01-05', 
                      '2023-01-10',               '2023-01-12',
    ]    
    live_months = [
        march_2022, april_2022, may_2022, june_2022, july_2022, august_2022, september_2022, october_2022, november_2022, december_2022,
        january_2023
        ]
    days = []
    for l in live_months:
        days += l
    return days[::-1]