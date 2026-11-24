import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def load_prices_from_sql(db_path="refinery_market.db"):
    conn = sqlite3.connect(db_path)
    # SQL query pulling clean pricing data + extracting month for turnaround logic
    query = """
        SELECT 
            trade_date,
            CAST(strftime('%m', trade_date) AS INTEGER) AS month,
            wti,
            wcs,
            rbob_bbl,
            ulsd_bbl,
            crack_321
        FROM market_prices
        ORDER BY trade_date ASC;
    """
    df = pd.read_sql_query(query, conn, parse_dates=["trade_date"])
    conn.close()
    return df.set_index("trade_date")

def solve_daily_refinery_lp(row):
    """
    Decision variables (bbl/day):
      x[0]: WTI in Max-Gasoline mode
      x[1]: WTI in Max-Distillate mode
      x[2]: WCS in Max-Gasoline mode
      x[3]: WCS in Max-Distillate mode
    """
    wti, wcs = row["wti"], row["wcs"]
    rbob, ulsd = row["rbob_bbl"], row["ulsd_bbl"]
    resid = 0.75 * wti  # Low-value fuel oil / resid priced at 75% of WTI

    # Volumetric yield matrix: [RBOB, ULSD, Resid]
    yields = np.array([
        [0.52, 0.28, 0.20],  # WTI Max-Gas
        [0.38, 0.42, 0.20],  # WTI Max-Dist
        [0.40, 0.25, 0.35],  # WCS Max-Gas
        [0.28, 0.37, 0.35],  # WCS Max-Dist
    ])

    product_prices = np.array([rbob, ulsd, resid])
    crude_costs = np.array([wti, wti, wcs, wcs])
    var_opex = np.array([3.50, 3.50, 5.50, 5.50])  # Heavy crude costs more to upgrade

    # Gross margin ($/bbl) for each decision variable
    revenues = yields @ product_prices
    margins = revenues - crude_costs - var_opex

    # Seasonal turnaround: 80k bbl/day capacity in March (3) & October (10)
    cdu_cap = 80_000.0 if row["month"] in (3, 10) else 100_000.0
    min_run = 0.60 * cdu_cap

    # linprog MINIMIZES c^T x, so negate margins to MAXIMIZE profit
    c = -margins

    # Inequality constraints: A_ub @ x <= b_ub
    A_ub = [
        [ 1,  1,  1,  1],   # Total crude <= CDU capacity
        [-1, -1, -1, -1],   # Total crude >= min turndown
        [ 0,  0,  1,  1],   # Heavy sour (WCS) <= 40,000 (Coker limit)
        [ 1,  0,  1,  0],   # Max-Gasoline mode <= 65,000 (FCC limit)
        [ 0,  1,  0,  1],   # Max-Distillate mode <= 65,000 (Hydrocracker limit)
    ]
    b_ub = [cdu_cap, -min_run, 40_000.0, 65_000.0, 65_000.0]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")

    total_bbl = res.x.sum()
    opt_margin_per_bbl = (-res.fun) / total_bbl if total_bbl > 0 else 0.0
    gas_mode_share = (res.x[0] + res.x[2]) / total_bbl

    return pd.Series({
        "opt_margin_bbl": opt_margin_per_bbl,
        "daily_profit_usd": -res.fun,
        "total_bbl": total_bbl,
        "gas_mode_pct": gas_mode_share * 100.0,
        "heavy_crude_pct": ((res.x[2] + res.x[3]) / total_bbl) * 100.0
    })

def main():
    df = load_prices_from_sql()
    lp_results = df.apply(solve_daily_refinery_lp, axis=1)
    combined = pd.concat([df, lp_results], axis=1)

    # Save LP results back to SQLite for backtesting
    conn = sqlite3.connect("refinery_market.db")
    combined.to_sql("refinery_lp_output", conn, if_exists="replace", index=True)
    conn.close()

    print(combined[["crack_321", "opt_margin_bbl", "gas_mode_pct"]].tail())

    # Plot: LP Optimized Margin vs. Benchmark 3-2-1 Crack Spread
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(combined.index, combined["crack_321"], label="Benchmark 3-2-1 Crack ($/bbl)", alpha=0.7)
    ax1.plot(combined.index, combined["opt_margin_bbl"], label="LP Optimal Margin ($/bbl)", color="#d62728", alpha=0.85)
    ax1.set_title("Refinery LP Optimal Margin vs. Market 3-2-1 Crack Spread", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Margin ($/bbl)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom panel: How the refinery flips between Max-Gasoline and Max-Distillate
    ax2.fill_between(combined.index, combined["gas_mode_pct"], color="#2ca02c", alpha=0.5, label="Max-Gasoline Mode (%)")
    ax2.set_ylabel("Gasoline Mode (%)")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("refinery_lp_results.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()