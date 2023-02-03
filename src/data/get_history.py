import datetime as dt
import pandas as pd
from os.path import exists
from pathlib import Path
import yfinance as yf
from yahooquery import Ticker


class Get_Stock_History:


    def __init__(self, day_0='2023-01-19'):
        """
            Constructor method for MyClass
        """
        self.day_0 = day_0
        self.month_0 = str(self.day_0)[:7]
        self.year_0 = str(self.day_0)[:4]
        self.ender_date = str(dt.datetime.now())[:10]
        self.saveRaw = Path(f"/data/raw/{self.year_0}/{self.month_0}/{self.day_0}/")
        self.saveRec = Path(f"/data/recommenders/{self.year_0}/{self.month_0}/{self.day_0}/")
        self.advisor1 = Path(f"/data/advisor/build/{self.month_0}/{self.day_0}/")
        self.saveHist = Path(f"/data/hist/{self.year_0}/{self.month_0}/{self.day_0}/")       


    def function0(self):
        """
            Method 2 of MyClass
        """
        bulk_data = pd.read_pickle(self.saveRec / "recommender_05_return_dataFrame.pkl")
        return sorted(list(bulk_data['ticker']))


    def function1(self, ticker_list):
        for ticker in ticker_list:
            try:
                temp = pd.DataFrame(pd.read_pickle(self.saveRaw / f"{ticker}.pkl"))
                if not exists(self.advisor1 / f"{ticker}_hist_{self.ender_date}.pkl"):
                    temp.to_pickle(self.advisor1 / f"{ticker}_hist_{self.ender_date}.pkl")
            except:
                history_data = pd.DataFrame(Ticker(ticker).history(period='1y')).reset_index().set_index('date').round(2)
                del history_data['symbol']                

        if exists(self.saveHist / "all_stock_history_adjclose"):
            try:
                data = pd.DataFrame(pd.read_pickle(self.saveHist / "all_stock_history_adjclose")).dropna()
                data = data[ticker_list]
                data.index = pd.to_datetime(data.index)
                df_train_data = pd.DataFrame(data.loc[:self.day_0])
                df_test_data = pd.DataFrame(data.loc[self.day_0:])            
                return df_train_data, df_test_data            

            except:
                data_0 = yf.download(
                    tickers=ticker_list,
                    period='1y',
                    end=self.day_0,
                    interval='1d',
                    group_by='column',
                    auto_adjust=False,
                    actions=False,
                    rounding=True,
                    show_errors=False,
                    prepost=False,
                    proxy=None, 
                    timeout=None,
                    )['Adj Close']
                data = pd.DataFrame(data_0).dropna()
                data.to_pickle(self.saveHist / "all_stock_history_adjclose")
                data.index = pd.to_datetime(data.index)
                df_train_data = pd.DataFrame(data.loc[:self.day_0])
                df_test_data = pd.DataFrame(data.loc[self.day_0:])            
                return df_train_data, df_test_data    

        else:            
            data_0 = yf.download(
                tickers=ticker_list,
                period='1y',
                end=self.day_0,
                interval='1d',
                group_by='column',
                auto_adjust=False,
                actions=False,
                rounding=True,
                show_errors=False,
                prepost=False,
                proxy=None, 
                timeout=None,
                )['Adj Close']
            data = pd.DataFrame(data_0).dropna()
            data.to_pickle(self.saveHist / "all_stock_history_adjclose")
            data.index = pd.to_datetime(data.index)
            df_train_data = pd.DataFrame(data.loc[:self.day_0])
            df_test_data = pd.DataFrame(data.loc[self.day_0:])            
            return df_train_data, df_test_data


    def function2(self):
        """
            Method 1 of MyClass
        """
        pass


    def function3(self):
        """
            Method 2 of MyClass
        """
        pass




# if __name__ == '__main__':
#     """
#         usage
#     """
#     obj = Get_Stock_History(day_0='2023-01-19')
#     ticker_list = obj.function0()
#     history_df = obj.function1(ticker_list)
#     obj.function2()
#     obj.function3()