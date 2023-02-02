import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import streamlit as st

import src.tools.functions as f0

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = [15, 10]
plt.rc("font", size=14)
np.random.seed(0)



class Optimal_SMA(object):
    
    
    def __init__(self, ticker, end_date):
        self.name = ticker
        self.end_date = str(end_date)[:10]
        self.company_longName = f0.company_longName(self.name)


    def build_optimal_sma(self, data, graphit=True, cc=0.0, ccc=0.0):
        n_forward = 1
        train_size = 0.80        
        data = pd.DataFrame(data)
        data.columns = [x.lower() for x in data.columns]
        
        data["Forward Close"] = data["adjclose"].shift(-n_forward)
        data["Forward Return"] = (data["Forward Close"] - data["adjclose"]) / data["adjclose"]
        result = []
        
        for sma_length in range(2, 50):
            data["SMA"] = data["adjclose"].rolling(sma_length).mean()
            data["input"] = [int(x) for x in data["adjclose"] > data["SMA"]]
            df = pd.DataFrame(data.copy())

            training = df.head(int(train_size * df.shape[0]))
            test = df.tail(int((1 - train_size) * df.shape[0]))
            tr_returns = training[training["input"] == 1]["Forward Return"]
            test_returns = test[test["input"] == 1]["Forward Return"]

            mean_forward_return_training = tr_returns.mean()
            mean_forward_return_test = test_returns.mean()
            pvalue = ttest_ind(tr_returns, test_returns, equal_var=False)[1]
            result.append(
                {
                    "sma_length": sma_length,
                    "training_forward_return": mean_forward_return_training,
                    "test_forward_return": mean_forward_return_test,
                    "p-value": pvalue,
                }
            )

        result.sort(key=lambda x: -x["training_forward_return"])
        best_sma = SMA_window = result[0]["sma_length"]
        SMA_window_col = str(SMA_window)

        # Create a short simple moving average column
        data[SMA_window_col] = (data["adjclose"].rolling(window=SMA_window, min_periods=1).mean())
        data["Signal"] = 0.0
        data["Signal"] = np.where(data[SMA_window_col] <= data["adjclose"], 1.0, 0.0)

        # create a new column 'Position' which is a day-to-day difference of the 'Signal' column.
        data["Position"] = data["Signal"].diff()


        if graphit == True:
            fig, ax = plt.subplots()

            plt.plot(data["adjclose"], label=self.company_longName)
            plt.plot(data[SMA_window_col], label="SMA-{}".format(best_sma))
            # plot 'buy' signals
            plt.plot(
                data[data["Position"] == 1].index,
                data[SMA_window_col][data["Position"] == 1], 
                "^", markersize=15, color="g", alpha=0.7, label="buy"
                )
            # plot 'sell' signals
            plt.plot(
                data[data["Position"] == -1].index, 
                data[SMA_window_col][data["Position"] == -1], 
                "v", markersize=15, color="r", alpha=0.7, label="sell"
                )
            plt.ylabel("Price in $", fontsize=20, fontweight="bold")
            plt.xlabel("Date", fontsize=20, fontweight="bold")
            plt.title(f"{self.name} - {str(SMA_window)} Crossover", fontsize=30, fontweight="bold")
            plt.xlabel("Date", fontsize=20, fontweight="bold")
            plt.ylabel("Price", fontsize=20, fontweight="bold")
            plt.title(f"{self.company_longName} ({self.name}) - SMA", fontsize=30, fontweight="bold")
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(15)
            ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
            plt.xticks(rotation=45)
            plt.yticks(rotation=90)
            ax.legend(loc="best", prop={"size": 16})
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            

        df_pos = data[(data["Position"] == 1) | (data["Position"] == -1)]
        action_lst = []
        for x in df_pos["Position"]:
            if x == 1:
                action_lst.append("Buy")
            else:
                action_lst.append("Sell")
        df_pos["Action"] = action_lst


        if df_pos["Action"][-1] == "Buy":
            st.metric(f"[{cc}/{ccc}]", f"{self.name}", f"{df_pos['Position'][-1]}")
            return self.name
        
        elif df_pos["Action"][-1] == "Sell":
            st.metric(f"[{cc}/{ccc}]", f"{self.name}", f"{df_pos['Position'][-1]}")
