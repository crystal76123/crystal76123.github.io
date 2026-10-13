"""
Python Stock Analytics Project: Topic 1
Strategy: Moving Average Crossover vs Buy-and-Hold
"""

# AI platforms ChatGPT and Gemini were used to assist with debugging and code organization. 

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt



# Settings

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"

# The 200-day average needs 200 trading days of history. Downloading 2019 as well means
# the averages are ready on the first day of the test, so the strategy is not forced
# to sit in cash for most of 2020 while buy-and-hold is invested.
WARMUP_START = "2019-01-01"

INITIAL_CAPITAL = 10000.0
FAST_MA = 50
SLOW_MA = 200



# Download Data

def download_data(tickers, start, end):
    print(f"Downloading data for: {tickers}")
    return yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False
    )



# Metrics

def calculate_metrics(cum_growth, daily_returns):
    total_return = (cum_growth.iloc[-1] - 1) * 100

    days = (cum_growth.index[-1] - cum_growth.index[0]).days
    annualized_return = (cum_growth.iloc[-1] ** (365 / days) - 1) * 100

    # Share of invested days (non-zero return) that ended with a gain
    active_days = daily_returns[daily_returns != 0]
    positive_day_pct = (active_days > 0).sum() / len(active_days) * 100 if len(active_days) > 0 else 0

    peak = cum_growth.cummax()
    drawdown = (cum_growth - peak) / peak
    max_drawdown = drawdown.min() * 100

    return total_return, annualized_return, positive_day_pct, max_drawdown



# Backtest

def backtest(data, tickers):
    results = {}

    for ticker in tickers:
        df = data[ticker].copy().dropna()

        df["MA50"] = df["Adj Close"].rolling(FAST_MA).mean()
        df["MA200"] = df["Adj Close"].rolling(SLOW_MA).mean()

        df["Signal"] = 0
        df.loc[df["MA50"] > df["MA200"], "Signal"] = 1
        df["Position"] = df["Signal"].shift(1).fillna(0)

        df["Market_Return"] = df["Adj Close"].pct_change().fillna(0)

        # Keep only the test period; both portfolios start with $10,000 on its first day
        df = df[df.index >= START_DATE].copy()
        df.iloc[0, df.columns.get_loc("Market_Return")] = 0

        df["Strategy_Return"] = df["Market_Return"] * df["Position"]

        df["Market_Growth"] = (1 + df["Market_Return"]).cumprod()
        df["Strategy_Growth"] = (1 + df["Strategy_Return"]).cumprod()

        df["Market_Value"] = INITIAL_CAPITAL * df["Market_Growth"]
        df["Strategy_Value"] = INITIAL_CAPITAL * df["Strategy_Growth"]

        df["Market_Peak"] = df["Market_Value"].cummax()
        df["Market_Drawdown"] = (df["Market_Value"] - df["Market_Peak"]) / df["Market_Peak"]

        df["Strategy_Peak"] = df["Strategy_Value"].cummax()
        df["Strategy_Drawdown"] = (df["Strategy_Value"] - df["Strategy_Peak"]) / df["Strategy_Peak"]

        m_tot, m_ann, m_win, m_dd = calculate_metrics(df["Market_Growth"], df["Market_Return"])
        s_tot, s_ann, s_win, s_dd = calculate_metrics(df["Strategy_Growth"], df["Strategy_Return"])

        results[ticker] = {
            "B&H Total %": round(m_tot, 2),
            "B&H Ann %": round(m_ann, 2),
            "B&H Positive Day %": round(m_win, 2),
            "B&H Max DD %": round(m_dd, 2),
            "Strat Total %": round(s_tot, 2),
            "Strat Ann %": round(s_ann, 2),
            "Strat Positive Day %": round(s_win, 2),
            "Strat Max DD %": round(s_dd, 2)
        }

        df.to_csv(f"{ticker}_processed.csv")

    return pd.DataFrame(results).T



# Plots

def create_plots(tickers):
    for ticker in tickers:
        df = pd.read_csv(f"{ticker}_processed.csv", index_col=0, parse_dates=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot(df.index, df["Market_Value"], label="Buy & Hold", linewidth=2)
        ax1.plot(df.index, df["Strategy_Value"], label="Strategy", linewidth=2)

        ax1.set_title(f"{ticker}: Portfolio Value", fontsize=14)
        ax1.set_ylabel("Value ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(df.index, df["Market_Drawdown"], 0, alpha=0.25, label="B&H Drawdown")
        ax2.fill_between(df.index, df["Strategy_Drawdown"], 0, alpha=0.35, label="Strategy Drawdown")

        ax2.set_title("Drawdown Comparison", fontsize=12)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{ticker}_results.png", dpi=300)
        plt.close()



# Main

def main():
    data = download_data(TICKERS, WARMUP_START, END_DATE)

    results = backtest(data, TICKERS)

    print("\n" + "=" * 80)
    print("BACKTESTING SUMMARY STATISTICS")
    print("=" * 80)
    print(results)

    results.to_csv("summary_metrics.csv")

    print("\nCreating plots...")
    create_plots(TICKERS)

    print("Done. Project complete.")



# Run Program

if __name__ == "__main__":
    main()