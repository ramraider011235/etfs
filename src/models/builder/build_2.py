import pandas as pd
import yfinance as yf
from yahooquery import Ticker
from pathlib import Path
from os.path import exists
import datetime as dt
from dateutil.relativedelta import relativedelta



class Builder_2(object):
    
    
    def __init__(self, day1):
        self.day1 = day1
        month1 = str(day1)[:7]
        year1 = str(day1)[:4]
        self.saveRec = Path(f"/home/gdp/hot_box/i4m/data/recommenders/{year1}/{month1}/{day1}/")
        self.saveRaw = Path(f"/home/gdp/hot_box/i4m/data/raw/{month1}/{day1}/")
        if not self.saveRaw.exists():
            self.saveRaw.mkdir(parents=True)        


    def technicals_minervini(self):
        # if exists(self.saveRec / "recommender_02_return_dataFrame.pkl"):
        #     x = pd.read_pickle(self.saveRec / "recommender_02_return_dataFrame.pkl")
        #     return x
        # else:

        data = pd.read_pickle(self.saveRec / "recommender_01_return_dataFrame.pkl")
        rec_02_tickers = list(data["ticker"])      
        start_date_101 = dt.date(int(str(self.day1)[:4]), int(str(self.day1)[5:7]), int(str(self.day1)[8:]))
        years_ago = str(start_date_101 - relativedelta(years=1, days=69))[:10]            
        cols_lst = ["ticker", "rs_rating", "returns_multiple", "current_price", "sma_50", "sma_150", "sma_200", "sma_200_20", "low_52_week", "high_52_week"]
        exportList = pd.DataFrame(columns=cols_lst)
        

        # Index Returns
        index_name = '^GSPC'
        if exists(self.saveRaw / "sp500_index.pkl"):
            index_df = pd.DataFrame(pd.read_pickle(self.saveRaw / "sp500_index.pkl"))
            index_df["pct_change"] = index_df["Adj Close"].pct_change()
            index_return = (index_df["pct_change"] + 1).cumprod()[-1]
        elif not exists(self.saveRaw / "sp500_index.pkl"):
            index_df = pd.DataFrame(yf.download(index_name, start=years_ago, end=self.day1))
            index_df.to_pickle(self.saveRaw / "sp500_index.pkl")
            index_df["pct_change"] = index_df["Adj Close"].pct_change()
            index_return = (index_df["pct_change"] + 1).cumprod()[-1]


        def source_hist(ticker_list):
            bad_list = []
            for ticker in ticker_list:
                if exists(self.saveRaw / f"{ticker}.pkl"):
                    pass
                else:
                    bad_list.append(ticker)
            return bad_list    


        def import_history(port_tics1):                               
            tickers = Ticker(port_tics1, asynchronous=True)
            df3 = pd.DataFrame(tickers.history(start=years_ago, end=self.day1))
            for s in port_tics1:
                try:
                    df = pd.DataFrame(df3.T[s].T[['adjclose', 'high', 'low']][1:])
                    df.index = pd.to_datetime(df.index)
                    df.to_pickle(self.saveRaw / f"{s}.pkl")
                except:
                    print(f"failed ticker {s}")
            return         


        # Find top 50% performing stocks (relative to the S&P 500)
        bad_list = source_hist(rec_02_tickers)  
        if bad_list:
            import_history(bad_list)                         
        
        # Calculating returns relative to the market (returns multiple)
        returns_multiples = []
        for ticker in rec_02_tickers:
            try:
                df = pd.DataFrame(pd.read_pickle(self.saveRaw / f"{ticker}.pkl"))
                df["pct_change"] = df["adjclose"].pct_change()
                stock_return = (df["pct_change"] + 1).cumprod()[-1]
                returns_multiple = round((stock_return / index_return), 2)
                returns_multiples.extend([returns_multiple])
            except Exception:
                print(f"Bad Ticker: {ticker}")

        # Creating dataframe of only top 70%
        rs_df = pd.DataFrame(list(zip(rec_02_tickers, returns_multiples)),columns=["ticker", "returns_multiple"],)
        rs_df["rs_rating"] = rs_df["returns_multiple"].rank(pct=True) * 100
        rs_df = rs_df[rs_df["rs_rating"] >= rs_df["rs_rating"].quantile(0.3)]

        # Checking Minervini conditions of top 60% of stocks in given list
        rs_stocks = list(rs_df["ticker"])
        for stock in rs_stocks:
            try:     
                df = pd.DataFrame(pd.read_pickle(self.saveRaw / f"{stock}.pkl"))
                sma = [50, 150, 200]
                for x in sma:
                    df["SMA_" + str(x)] = round(df["adjclose"].rolling(window=x).mean(), 2)
                # Storing required values
                currentClose = df["adjclose"].iloc[-1]
                MA_50 = df["SMA_50"].iloc[-1]
                MA_150 = df["SMA_150"].iloc[-1]
                MA_200 = df["SMA_200"].iloc[-1]
                low_52_week = round(min(df["low"][-260:]), 2)
                high_52_week = round(max(df["high"][-260:]), 2)
                RS_Rating = round(rs_df[rs_df["ticker"] == stock].rs_rating.tolist()[0], 2)
                Returns_multiple = round(
                    rs_df[rs_df["ticker"] == stock].returns_multiple.tolist()[0], 2)
                try:
                    MA_200_20 = df["SMA_200"][-20]
                except Exception:
                    MA_200_20 = 0                   

            # Condition 1: Current Price > 150 SMA and > 200 SMA
                condition_1 = currentClose > MA_150 > MA_200
            # Condition 2: 150 SMA and > 200 SMA
                condition_2 = MA_150 > MA_200
            # Condition 3: 200 SMA trending up for at least 1 month
                condition_3 = MA_200 > MA_200_20
            # Condition 4: 50 SMA> 150 SMA and 50 SMA> 200 SMA
                condition_4 = MA_50 > MA_150 > MA_200
            # Condition 5: Current Price > 50 SMA
                condition_5 = currentClose > MA_50
            # Condition 6: Current Price is at least 30% above 52 week low
                condition_6 = currentClose >= (1.30 * low_52_week)
            # Condition 7: Current Price is within 25% of 52 week high
                condition_7 = currentClose >= (0.75 * high_52_week)            
                
            # If all conditions above are true, add Ticker to exportList
                if (condition_1 & condition_2 & condition_3 & condition_4 & condition_5 & condition_6 & condition_7):
                    exportList = exportList.append(
                        {
                            "ticker": stock,
                            "rs_rating": RS_Rating,
                            "returns_multiple": Returns_multiple,
                            "current_price": currentClose,
                            "sma_50": MA_50,
                            "sma_150": MA_150,
                            "sma_200": MA_200,
                            "sma_200_20": MA_200_20,
                            "low_52_week": low_52_week,
                            "high_52_week": high_52_week,
                        },
                        ignore_index=True,
                    ).sort_values(by="rs_rating", ascending=False)             
            except Exception:
                print(f"Bad Ticker: {stock}")



        print("\n[2] MINERVINI ")

        exportList_A = exportList.drop_duplicates(subset="ticker")
        print(f"   > PART-A:")
        print(f"     * Successful Stocks: [{exportList_A.shape}]")

        exportList_B = pd.DataFrame(exportList_A[exportList_A.rs_rating >= 69.0]).round(2)
        print(f"   > PART-B:")
        print(f"     * Successful Stock WHERE (rs_rating > 69.0): [{exportList_B.shape}] \n")        

        exportList_B.to_pickle(self.saveRec / "recommender_02_return_dataFrame.pkl")       
        return exportList_B
