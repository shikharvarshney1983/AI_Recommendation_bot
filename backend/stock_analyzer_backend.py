# Install necessary libraries before running:
# pip install Flask Flask-Cors yfinance pandas pandas-ta transformers torch

from flask import Flask, jsonify, render_template, session
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from news_analyzer import get_news_and_announcements # Import the new function

# -----------------------------------------------------------------------------
# Backend Setup (Flask)
# -----------------------------------------------------------------------------
app = Flask(__name__)

app.secret_key = 'BAD_SECRET_KEY'  # Set a secret key for session management
# Note: In production, use a more secure key and consider using environment variables.
# Enable CORS to allow the React frontend to communicate with this backend
CORS(app)

# -----------------------------------------------------------------------------
# Technical Indicator Calculation Functions
# -----------------------------------------------------------------------------

def calculate_vstop(df, atr_period=14, multiplier=2.0, lookback_period=1):
    """
    Calculates the Volatility Stop (VStop).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("prices should be a pandas DataFrame")
    if atr_period <= 0 or lookback_period <= 0 or multiplier <= 0:
        raise ValueError("Parameters should be positive values")
    if 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns")
    atr = df.ta.atr(length=atr_period, append=True)
    fHigh, fLow = 0, 0
    vstop = pd.Series(index=df.index)
    for i in range(len(df)):
        if i < atr_period:
            vstop.iloc[i] = df['Close'].iloc[i] + (multiplier * atr.iloc[i])
        else:
            high_period = df['Close'].iloc[i - lookback_period:i + 1].max()
            low_period = df['Close'].iloc[i - lookback_period:i + 1].min()
            if df['High'].iloc[i - 1] > vstop.iloc[i - 1]:
                if fHigh < 1: vstop.iloc[i] = high_period - multiplier * atr.iloc[i]
                else: vstop.iloc[i] = max(vstop.iloc[i - 1], high_period - multiplier * atr.iloc[i])
                fHigh += 1; fLow = 0
            else:
                if fLow < 1: vstop.iloc[i] = low_period + multiplier * atr.iloc[i]
                else: vstop.iloc[i] = min(vstop.iloc[i - 1], low_period + multiplier * atr.iloc[i])
                fLow += 1; fHigh = 0
    df['VStop'] = vstop
    return df['VStop'].iloc[-1]

def calculate_relative_strength(stock_data, index_data):
    """
    Calculates the relative strength of a stock against an index (NIFTY 50).
    """
    stock_normalized = stock_data['Close'] / stock_data['Close'].iloc[0]
    index_normalized = index_data['Close'] / index_data['Close'].iloc[0]
    rs = stock_normalized / index_normalized
    return rs.iloc[-1]

def get_chart_pattern(df):
    """
    Identifies basic chart patterns.
    """
    last_price = df['Close'].iloc[-1]
    high_52_week = df['High'][-252:].max()
    if last_price >= high_52_week:
        return "New 52-Week High Breakout"
    return "No clear pattern"

# -----------------------------------------------------------------------------
# Main Analysis Function
# -----------------------------------------------------------------------------
def get_stock_analysis(ticker, timeframe):
    """
    Fetches stock data and performs all technical and fundamental analysis.
    """
    try:
        interval_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
        period_map = {"daily": "2y", "weekly": "5y", "monthly": "10y"}
        interval = interval_map.get(timeframe.lower(), "1d")
        period = period_map.get(timeframe.lower(), "2y")

        stock_ticker = yf.Ticker(ticker)
        stock_data = stock_ticker.history(period=period, interval=interval, auto_adjust=True)
        nifty_data = yf.Ticker("^NSEI").history(period=period, interval=interval, auto_adjust=True)

        if stock_data.empty:
            return {"error": f"No data found for ticker {ticker} on a {timeframe} timeframe."}

        daily_data = stock_ticker.history(period="1y", interval="1d", auto_adjust=True)
        high_52w = daily_data['High'].max() if not daily_data.empty else 0
        low_52w = daily_data['Low'].min() if not daily_data.empty else 0

        stock_data.ta.ema(length=21, append=True)
        stock_data.ta.ema(length=50, append=True)
        stock_data.ta.ema(length=100, append=True)
        stock_data.ta.ema(length=200, append=True)
        stock_data.ta.sma(length=21, append=True)
        stock_data.ta.sma(length=50, append=True)
        stock_data.ta.sma(length=100, append=True)
        stock_data.ta.sma(length=200, append=True)
        stock_data.ta.rsi(length=14, append=True)
        stock_data.ta.adx(length=14, append=True)
        stock_data.ta.psar(append=True)
        stock_data.ta.donchian(lower_length=20, upper_length=20, append=True)

        latest_data = stock_data.iloc[-1]
        current_price = latest_data['Close']
        ema21, ema50, ema100, ema200 = latest_data.get('EMA_21', 0), latest_data.get('EMA_50', 0), latest_data.get('EMA_100', 0), latest_data.get('EMA_200', 0)
        sma21, sma50, sma100, sma200 = latest_data.get('SMA_21', 0), latest_data.get('SMA_50', 0), latest_data.get('SMA_100', 0), latest_data.get('SMA_200', 0)
        rsi, adx = latest_data.get('RSI_14', 0), latest_data.get('ADX_14', 0)
        psar_long, psar_short = latest_data.get('PSARl_0.02_0.2'), latest_data.get('PSARs_0.02_0.2')
        psar = psar_long if not pd.isna(psar_long) else psar_short
        donchian_upper = latest_data.get('DCU_20_20', 0)
        
        vstop = calculate_vstop(stock_data)
        relative_strength = calculate_relative_strength(stock_data, nifty_data)
        chart_pattern = get_chart_pattern(daily_data)

        info = stock_ticker.info
        pe, forwardPE, eps, forwardEPS, priceToBook = info.get('trailingPE'), info.get('forwardPE'), info.get('trailingEps'), info.get('forwardEps'), info.get('priceToBook')

        cfoPat, interestCoverage = None, None
        try:
            financials, cashflow = stock_ticker.financials, stock_ticker.cashflow
            if not financials.empty and not cashflow.empty:
                pat = financials.loc['Net Income Continuous Operations'].iloc[0] if 'Net Income Continuous Operations' in financials.index else 0
                cfo = cashflow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow.index else 0
                if pat and cfo: cfoPat = cfo / pat
                ebit = financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else 0
                interest_expense = financials.loc['Interest Expense'].iloc[0] if 'Interest Expense' in financials.index else 0
                if ebit and interest_expense: interestCoverage = ebit / abs(interest_expense)
        except Exception as e: print(f"Could not calculate some fundamental ratios: {e}")

        # Fetch and analyze news
        news_data = get_news_and_announcements(ticker.split(".")[0], info.get('longName'))

        is_strong_uptrend = current_price > ema50 and current_price > sma50 and current_price > ema200 and current_price > sma200
        is_buy_signal = is_strong_uptrend and rsi > 55 and adx > 25 and current_price > vstop and current_price > psar
        is_sell_signal = current_price < ema50 and current_price < vstop and current_price < psar and rsi < 45
        
        recommendation, reason = "Hold", f"On a {timeframe} basis, conditions are mixed. Advisable to wait for a clearer trend."
        if is_buy_signal:
            recommendation, reason = "Buy", f"Strong bullish signals on the {timeframe} chart. Price is above key moving averages (50 & 200), with strong RSI (>55) and ADX (>25) momentum."
        elif is_sell_signal:
            recommendation, reason = "Sell", f"Bearish signals on the {timeframe} chart. Price has dropped below key support levels (50-period MA, VStop, PSAR) with weakening RSI."

        return {
            "currentPrice": current_price, "high52w": high_52w, "low52w": low_52w,
            "ema21": ema21, "ema50": ema50, "ema100": ema100, "ema200": ema200,
            "sma21": sma21, "sma50": sma50, "sma100": sma100, "sma200": sma200,
            "rsi": rsi, "adx": adx, "psar": psar, "donchianUpper": donchian_upper,
            "vstop": vstop, "relativeStrength": relative_strength, "chartPattern": chart_pattern,
            "recommendation": recommendation, "reason": reason,
            "pe": pe, "forwardPE": forwardPE, "eps": eps, "forwardEPS": forwardEPS,
            "priceToBook": priceToBook, "cfoPat": cfoPat, "interestCoverage": interestCoverage,
            "news": news_data # Add news to the response
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": "An error occurred while processing the request. The ticker might be invalid, delisted, or has missing data."}

# -----------------------------------------------------------------------------
# API Endpoint
# -----------------------------------------------------------------------------
@app.route('/analyze/<ticker>/<timeframe>', methods=['GET'])
def analyze_stock_endpoint(ticker, timeframe):
    analysis_result = get_stock_analysis(ticker, timeframe)
    print(analysis_result)
    return jsonify(analysis_result)

@app.route("/")
def index():
    return render_template('index.html')
    
# -----------------------------------------------------------------------------
# Run the Flask App
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
