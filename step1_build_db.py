import sqlite3
import yfinance as yf
import pandas as pd

def build_database(db_path="refinery_market.db"):
    # 1. Fetch 10 years of daily front-month NYMEX futures
    tickers = ["CL=F", "RB=F", "HO=F"]
    raw = yf.download(tickers, start="2016-01-01", auto_adjust=False)

    df = raw["Close"].copy()
    df = df.rename(columns={"CL=F": "wti", "RB=F": "rbob_gal", "HO=F": "ulsd_gal"})
    df = df.dropna()

    # 2. Remove extreme single-day negative WTI anomaly (April 20, 2020)
    df = df[df["wti"] > 5.0].copy()

    # 3. Unit conversions ($/gal -> $/bbl) & benchmark 3-2-1 Crack Spread
    df["rbob_bbl"] = df["rbob_gal"] * 42.0
    df["ulsd_bbl"] = df["ulsd_gal"] * 42.0
    df["crack_321"] = (2 * df["rbob_bbl"] + 1 * df["ulsd_bbl"] - 3 * df["wti"]) / 3.0

    # Model heavy sour crude (WCS) using a static $14/bbl historical average discount to WTI
    # (Daily WCS physical differentials require proprietary Argus/Platts feeds)
    df["wcs"] = df["wti"] - 14.0

    # Format Date as string for clean SQLite indexing
    df.index = df.index.strftime("%Y-%m-%d")
    df.index.name = "trade_date"

    # 4. Write to SQLite
    conn = sqlite3.connect(db_path)
    df.to_sql("market_prices", conn, if_exists="replace", index=True)
    conn.close()
    print(f"Saved {len(df)} trading days to {db_path} (table: market_prices)")

if __name__ == "__main__":
    build_database()