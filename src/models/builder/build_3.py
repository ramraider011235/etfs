from os.path import exists
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newspaper import Article, Config
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

pd.set_option('display.max_columns', None)
# nltk.download("vader_lexicon")
# nltk.download('punkt')

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10



class Builder_3(object):
    
    
    def __init__(self, day1):
        self.day1 = day1
        month1 = str(day1)[:7]
        year1 = str(day1)[:4]
        self.saveRec = Path(f"/home/gdp/hot_box/i4m/data/recommenders/{year1}/{month1}/{day1}/")

        self.sentiment = Path(f"/home/gdp/hot_box/i4m/data/sentiment/sentiment/{year1}/{month1}/{day1}/")
        if not self.sentiment.exists():
            self.sentiment.mkdir(parents=True)           
        
        self.single_news = Path(f"/home/gdp/hot_box/i4m/data/sentiment/single_news/{year1}/{month1}/{day1}/")   
        if not self.single_news.exists():
            self.single_news.mkdir(parents=True)        
            
        data_2 = pd.read_pickle(self.saveRec / "recommender_02_return_dataFrame.pkl")
        self.stocks_list = list(data_2['ticker'])        


    def sentiment_1(self, stocks):
        print(f"\nTotal Input Stocks: {len(stocks)} \n")
        finwiz_url = 'https://finviz.com/quote.ashx?t='
        new_stock_list = []
        news_tables = {}   
        for ticker in stocks:
            try:
                url = finwiz_url + ticker
                req = Request(url=url,headers={'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'}) 
                resp = urlopen(req)    
                html = BeautifulSoup(resp, features="lxml")
                news_table = html.find(id='news-table')
                news_tables[ticker] = news_table
                new_stock_list.append(ticker)
            except Exception:
                print(f'BAD TICKER: {ticker}')
        return news_tables, new_stock_list


    def sentiment_2(self, stocks, news_tables, parsed_news=[], n=21):
        new_stock_list = []
        for file_name, news_table in news_tables.items():
            rows = news_table.findAll("tr")
            rows = rows[:n]
            for row in rows:
                cols = row.findAll("td")
                try:
                    ticker = file_name.split('_')[0]
                    date = cols[0].text.split()[0]
                    time = cols[0].text.split()[1]
                    title = cols[1].get_text()
                    link = cols[1].a['href']
                    source = link.split("/")[2]          
                    if source == "feedproxy.google.com":
                        source = link.split("/")[4]
                    info_dict = {
                        "Ticker": ticker,
                        "Date": date, 
                        "Time": time,
                        "Title": title, 
                        "Source": source, 
                        "Link": link
                        }
                    parsed_news.append(info_dict)
                    new_stock_list.append(ticker)
                except Exception:
                    pass
        parsed_news_df = pd.DataFrame(parsed_news)
        parsed_news_df.columns = [x.lower() for x in parsed_news_df.columns]
        parsed_news_df['date'] = pd.to_datetime(parsed_news_df['date'])
        parsed_news_df = parsed_news_df[parsed_news_df['date'] >= pd.Timestamp('2022-03-01')]    
        parsed_news_df.to_pickle(self.single_news / f"sentiment_all_stock_news.pkl")
        tickers = list(set(parsed_news_df['ticker']))
        for ticker in tickers:
            stock_news_df = pd.DataFrame(parsed_news_df[parsed_news_df['ticker'] == ticker]).sort_values('date', ascending=False).iloc[:n]
            stock_news_df.to_pickle(self.single_news / f"df_single_news_{ticker}.pkl")
        return parsed_news_df, new_stock_list


    def sentiment_3(self, stocks):
        final_stocks = []
        c = 0.0
        for stock in stocks:
            c += 1
            if exists(self.single_news / f"df_single_news_full_{stock}.pkl"):
                final_stocks.append(stock)
                print(f"\n[ {int(c)} / {int(len(stocks))} ] - {stock} \n [X] - DONE - {stock}")
            else:
                print(f"\n[ {int(c)} / {int(len(stocks))} ] - {stock}")
                try:
                    df = pd.DataFrame(pd.read_pickle(self.single_news / f"df_single_news_{stock}.pkl"))
                    df.columns = [x.lower() for x in df.columns]
                    list =[]                                                                         # creating an empty list
                    for i in df.index:
                        dict = {}                                                                    # create empty dictionary to add articles
                        article = Article(df['link'][i], config=config)                              # providing the link
                        try:
                            article.download()                                                       # downloading the article 
                            article.parse()                                                          # parsing the article
                            article.nlp()                                                            # performing natural language processing
                        except:                                                                      # exception handling
                            print('error stock download')
                        dict['date']=df['date'][i]                                                   # storing results in dictionary from above
                        dict['source']=df['source'][i] 
                        dict['title']=article.title
                        dict['article']=article.text
                        dict['summary']=article.summary
                        dict['key_words']=article.keywords
                        dict['link']=df['link'][i]
                        list.append(dict)
                    check_empty = not any(list)
                    if check_empty == False:
                        try:
                            news_df=pd.DataFrame(list)                                               # creating dataframe
                            p1 = (self.single_news / f"df_single_news_full_{stock}.pkl")
                            news_df.to_pickle(p1)
                            final_stocks.append(stock)
                            print(f"[X] - DONE - {stock}")                                           # exception handling
                        except Exception:
                            print('error save')
                except Exception as e:                                                               # exception handling
                    print("Exception:" + str(e))
        return final_stocks          


    def sentiment_4(self, newS, stocks):
        for stock in stocks:               
            (
                dates, sources, titles, articles, summarys, key_words, links
            ) = (
                newS['date'], 
                newS['source'], 
                newS['title'], 
                newS['article'], 
                newS['summary'], 
                newS['key_words'], 
                newS['link']
            )
            parsed_news=[]
            for r in range(len(newS)):
                parsed_news.append([stock, dates[r], sources[r], titles[r], articles[r], summarys[r], key_words[r], links[r]])
        # Sentiment Analysis
            analyzer = SentimentIntensityAnalyzer()
            news = pd.DataFrame(parsed_news, columns=["ticker", "date", 'source', "title", 'article', 'summary', 'key_words', "link"]).dropna()       
            scores = news["summary"].apply(analyzer.polarity_scores).tolist()        
            df_scores = pd.DataFrame(scores)
            news = news.join(df_scores, rsuffix="_right")     
        # View Data
            news["date"] = pd.to_datetime(news['date']).dt.date
            unique_ticker = news["ticker"].unique().tolist()
            news_dict = {name: news.loc[news["ticker"] == name] for name in unique_ticker}
            values = []
        for stock in stocks:
            dataframe = news_dict[stock]
            dataframe = dataframe.set_index("ticker")
            mean = round(dataframe["compound"].mean() * 100, 0)
            values.append(mean)
        df = pd.DataFrame(stocks, columns=["ticker"])
        df["sentiment_score"] = values
        return df        


    def sentiment_5 (self, stocks):
        df = pd.DataFrame()
        symbols = []
        sentiments = []
        for stock in stocks:
            try:           
                newS = pd.read_pickle(self.single_news / f"df_single_news_full_{stock}.pkl")
                fd = self.sentiment_4(newS, [stock])
                symbols.append(fd["ticker"].loc[0])
                sentiments.append(fd["sentiment_score"].loc[0])
                fd.to_pickle(self.sentiment / f"{stock}_sentiment.pkl")
            except Exception:
                print(f"BAD TICKER {stock} 4")
                stocks.remove(stock)
        df["ticker"] = symbols
        df["sentiment_score"] = sentiments
        return df        


    def run_mod_3(self):     
        if exists(self.saveRec / "recommender_03_return_dataFrame.pkl"):
            x = pd.read_pickle(self.saveRec / "recommender_03_return_dataFrame.pkl")
            return x

        else:
            data_2_tickers = list(pd.read_pickle(self.saveRec / "recommender_02_return_dataFrame.pkl")['ticker'])
            news_tables, new_stock_list_1 = self.sentiment_1(data_2_tickers)
            parsed_news_df, new_stock_list_2 = self.sentiment_2(new_stock_list_1, news_tables)
            new_stock_list_3 = self.sentiment_3(new_stock_list_2)
            df_final = self.sentiment_5(new_stock_list_3)

            df_final = df_final[df_final['sentiment_score'] >= 0.0]
            df_final = pd.DataFrame(df_final.copy()).sort_values('sentiment_score', ascending=False).sort_values('sentiment_score', ascending=False)
            df_final.to_pickle(self.saveRec / "recommender_03_return_dataFrame.pkl")
            print(f"[3] Sentiment Analysis - Successful Securities = [{len(df_final['ticker'])}]]")
            print(df_final.shape)
            return df_final          