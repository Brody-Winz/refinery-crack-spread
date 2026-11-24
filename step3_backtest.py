import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_backtest_v2(db_path="refinery_market.db"):
    # 1. Load data from SQLite
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT trade_date, crack_321 FROM market_prices ORDER BY trade_date ASC;",
        conn,
        parse_dates=["trade_date"]
    ).set_index("trade_date")
    conn.close()

    # 2. Indicators: 60-day Z-score and 20-day rolling volatility ($/bbl/day)
    df["ma_60"] = df["crack_321"].rolling(60).mean()
    df["std_60"] = df["crack_321"].rolling(60).std()
    df["z_score"] = (df["crack_321"] - df["ma_60"]) / df["std_60"]
    df["daily_change"] = df["crack_321"].diff().fillna(0)
    df["vol_20"] = df["daily_change"].rolling(20).std()
    df["month"] = df.index.month

    # 3. Hysteresis + Stop-Loss + Spring Turnaround Filter
    signals = np.zeros(len(df))
    pos = 0
    z_vals = df["z_score"].values
    months = df["month"].values

    for i in range(len(df)):
        z = z_vals[i]
        m = months[i]
        if np.isnan(z):
            signals[i] = 0
            continue

        if pos == 0:
            # Enter Long if deeply oversold; Enter Short if overbought (avoiding Feb-Apr turnaround rally)
            if z < -1.5 and z > -3.0:
                pos = 1
            elif z > 1.5 and z < 3.0 and m not in (2, 3, 4):
                pos = -1
        elif pos == 1:
            # Exit Long on mean reversion (z >= 0) OR stop-loss (z <= -3.0)
            if z >= 0.0 or z <= -3.0:
                pos = 0
        elif pos == -1:
            # Exit Short on mean reversion (z <= 0) OR stop-loss (z >= 3.0)
            if z <= 0.0 or z >= 3.0:
                pos = 0

        signals[i] = pos

    df["raw_signal"] = signals

    # 4. Volatility-Targeted Position Sizing (target $1.00/bbl daily vol, capped at 1.5x)
    target_daily_vol = 1.00
    df["vol_scale"] = (target_daily_vol / df["vol_20"].replace(0, np.nan)).clip(upper=1.5).fillna(0)
    df["sized_signal"] = df["raw_signal"] * df["vol_scale"]

    # 5. Shift by 1 day to prevent look-ahead bias
    df["position"] = df["sized_signal"].shift(1).fillna(0)

    # 6. P&L minus $0.05/bbl transaction cost on traded volume
    df["gross_pnl"] = df["position"] * df["daily_change"]
    df["turnover"] = df["position"].diff().abs().fillna(0)
    df["net_pnl"] = df["gross_pnl"] - (df["turnover"] * 0.05)

    # Count actual trade entries/exits (when direction changes)
    trade_events = int((df["raw_signal"].diff().fillna(0) != 0).sum())

    # 7. Cumulative P&L & Drawdown
    capital_per_bbl = 50.0
    df["cum_pnl"] = df["net_pnl"].cumsum()
    df["equity"] = capital_per_bbl + df["cum_pnl"]
    df["running_max"] = df["equity"].cummax()
    df["drawdown_pct"] = ((df["equity"] - df["running_max"]) / df["running_max"]) * 100.0

    # 8. Performance Metrics
    daily_returns = df["net_pnl"] / capital_per_bbl
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    max_dd = df["drawdown_pct"].min()

    print(f"Total Trade Events:       {trade_events}")
    print(f"Cumulative Net P&L:       ${df['cum_pnl'].iloc[-1]:.2f} / bbl")
    print(f"Annualized Sharpe Ratio:  {sharpe:.2f}")
    print(f"Max Drawdown:             {max_dd:.2f}%")

    # 9. Plot Equity Curve & Drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df.index, df["cum_pnl"], color="#1f77b4", linewidth=1.8, label="Cumulative Net P&L ($/bbl)")
    ax1.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax1.set_title(f"3-2-1 Crack Spread Strategy v2 (Sharpe: {sharpe:.2f} | Max DD: {max_dd:.1f}%)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cumulative P&L ($/bbl)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(df.index, df["drawdown_pct"], 0, color="#d62728", alpha=0.4, label="Drawdown (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_equity_curve.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    run_backtest_v2()