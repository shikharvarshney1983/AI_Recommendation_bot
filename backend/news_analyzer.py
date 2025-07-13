# news_analyzer.py
import yfinance as yf
from transformers import pipeline
import torch

# Initialize the sentiment analysis pipeline with a financial model
# This might download the model on the first run.
# I've replaced the original model with 'ProsusAI/finbert', a widely-used and reliable model
# for financial sentiment analysis, which should resolve the loading error.
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="ProsusAI/finbert"
    )
except Exception as e:
    print(f"Failed to load sentiment model: {e}")
    sentiment_analyzer = None

def get_news_analysis(ticker_symbol):
    """
    Fetches the latest news for a stock ticker and analyzes its sentiment.
    """
    if not sentiment_analyzer:
        return [{"error": "Sentiment analysis model is not available."}]

    try:
        # Appending '.NS' for National Stock Exchange (NSE) tickers in India
        if not ticker_symbol.upper().endswith('.NS'):
            ticker_symbol += '.NS'
            
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        
        if not news:
            print(f"No news found for {ticker_symbol}.")
            return [{"error": f"No news found for {ticker_symbol}."}]

        analyzed_news = []
        for article in news[:8]:  # Analyze the top 8 articles
            title = article["content"].get('title')
            link = article["content"]["clickThroughUrl"].get('url')
            publisher = article["content"].get('publisher')

            if not title:
                continue

            # Analyze sentiment of the headline
            try:
                # The model returns a list of dictionaries
                result = sentiment_analyzer(title)
                sentiment = result[0] if result else {'label': 'neutral', 'score': 0.0}
                
                # Add a brief interpretation based on sentiment
                interpretation = "This could be a neutral signal for the stock."
                # Using a slightly lower threshold as models can be conservative
                if sentiment['label'].lower() == 'positive' and sentiment['score'] > 0.6:
                    interpretation = "This news is likely to have a positive impact on the stock price."
                elif sentiment['label'].lower() == 'negative' and sentiment['score'] > 0.6:
                    interpretation = "This news could potentially have a negative impact on the stock price."

                analyzed_news.append({
                    "title": title,
                    "link": link,
                    "publisher": publisher,
                    "sentiment": sentiment['label'],
                    "sentiment_score": round(sentiment['score'], 2),
                    "interpretation": interpretation
                })
            except Exception as e:
                print(f"Could not analyze sentiment for '{title}': {e}")
                continue

        return analyzed_news

    except Exception as e:
        print(f"An error occurred while fetching news for {ticker_symbol}: {e}")
        return [{"error": f"Could not fetch or analyze news for {ticker_symbol}."}]


if __name__ == '__main__':
    # Example usage with an Indian stock ticker (NSE)
    stock_ticker = "GLENMARK.NS" 
    analysis_results = get_news_analysis(stock_ticker)
    
    # Print the results in a readable format
    print(f"\nSentiment Analysis for {stock_ticker}.NS News:\n")
    
    # This logic is updated to handle three cases:
    # 1. The list is empty (no processable news found).
    # 2. The list contains an error message from the function.
    # 3. The list contains valid, analyzed news articles.
    if not analysis_results:
        print("No processable news was found for this ticker.")
    elif "error" in analysis_results[0]:
        print(f"An error occurred: {analysis_results[0]['error']}")
    else:
        for result in analysis_results:
            print(f"Title: {result['title']}")
            print(f"Publisher: {result['publisher']}")
            print(f"Sentiment: {result['sentiment']} (Score: {result['sentiment_score']})")
            print(f"Interpretation: {result['interpretation']}")
            print("-" * 30)
