# news_analyzer.py

# Make sure to install required libraries:
# pip install bse-india transformers torch beautifulsoup4 requests yfinance
import yfinance as yf
from transformers import pipeline
import torch
from bse import BSE
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

# --- Constants ---

# A list of reliable news sources to prioritize in Google searches
RELIABLE_NEWS_SOURCES = [
    "moneycontrol.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "reuters.com",
    "ndtv.com/business",
    "businesstoday.in",
    "thehindubusinessline.com",
    "marketsmojo.com"
]

# Keywords to identify and filter out non-essential corporate announcements
IRRELEVANT_ANNOUNCEMENT_KEYWORDS = [
    'investor presentation', 'analyst meet', 'conference call', 'postal ballot',
    'voting results', 'agm', 'e-voting', 'intimation of schedule', 
    'loss of share certificate', 'transcript', 'compliance certificate'
]


# --- Initializations ---

# Initialize the sentiment analysis pipeline with a financial-tuned model
# This might download the model files on the first run
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )
except Exception as e:
    print(f"Failed to load sentiment model: {e}")
    sentiment_analyzer = None

# Initialize the BSE API client
try:
    # The 'bse-india' library may download some data on first use
    bse = BSE(download_folder='./bse_downloads')
except Exception as e:
    print(f"Failed to initialize BSE client: {e}")
    bse = None

# --- Helper Functions ---

def analyze_text_sentiment(text):
    """
    Analyzes a single piece of text for sentiment and provides an interpretation.
    Returns a dictionary with sentiment label, score, and interpretation.
    """
    if not sentiment_analyzer or not text:
        return {'label': 'neutral', 'score': 0.0, 'interpretation': 'Analysis not available.'}
    
    try:
        result = sentiment_analyzer(text)
        sentiment = result[0] if result else {'label': 'neutral', 'score': 0.0}
        
        interpretation = "This could be a neutral signal for the stock."
        # Provide stronger interpretation for high-confidence scores
        if sentiment['label'].lower() == 'positive' and sentiment['score'] > 0.75:
            interpretation = "This is likely to have a positive impact on the stock price."
        elif sentiment['label'].lower() == 'negative' and sentiment['score'] > 0.75:
            interpretation = "This could potentially have a negative impact on the stock price."
        
        sentiment['interpretation'] = interpretation
        return sentiment
        
    except Exception as e:
        print(f"Could not analyze sentiment for '{text[:50]}...': {e}")
        return {'label': 'neutral', 'score': 0.0, 'interpretation': 'Error during analysis.'}

def get_google_news(company_name):
    """
    Fetches news from Google by scraping the news tab.
    Filters results to include only reliable sources.
    """
    analyzed_news = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    # Construct a search query for the Google News tab
    query = f'"{company_name}" stock news'
    search_url = f"https://www.google.com/search?q={query}&tbm=nws"
    
    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all search result containers
        news_results = soup.find_all('div', class_='SoaBEf', limit=15)
        
        for item in news_results:
            link_tag = item.find('a')
            title_tag = item.find('div', role='heading')
            publisher_tag = item.find('span')

            if link_tag and title_tag and publisher_tag:
                title = title_tag.get_text()
                link = link_tag['href']
                publisher = publisher_tag.get_text()
                
                # Filter by our list of reliable sources
                if any(source in link for source in RELIABLE_NEWS_SOURCES):
                    sentiment_data = analyze_text_sentiment(title)
                    analyzed_news.append({
                        "title": title,
                        "link": link,
                        "publisher": publisher,
                        "sentiment": sentiment_data.get('label'),
                        "sentiment_score": sentiment_data.get('score'),
                        "interpretation": sentiment_data.get('interpretation'),
                        "type": "News"
                    })
        
        return analyzed_news[:8] # Return the top 8 most relevant articles

    except Exception as e:
        print(f"Error fetching Google News for {company_name}: {e}")
        return []


def get_bse_announcements(company_name):
    """
    Fetches and analyzes corporate announcements from the BSE website.
    Filters out irrelevant announcements based on keywords.
    """
    if not bse:
        return []
        
    analyzed_announcements = []
    try:
        # Get the BSE scrip code from the company name.
        # This is a best-effort attempt and might fail for some complex names.
        # search_term = company_name.split(' ')[0].replace('.', '')

        scrip_code = bse.getScripCode(company_name)

        # scrip_code = scrip_code_data[0]['scrip_code'] if scrip_code_data else None

        if not scrip_code:
            print(f"Could not find BSE scrip code for {company_name}")
            return []

        to_date = datetime.now()
        from_date = to_date - timedelta(days=30)
        
        announcements = bse.announcements(
            scripcode=scrip_code,
            from_date=datetime.strptime(from_date.strftime("%Y-%m-%d"), "%Y-%m-%d"),
            to_date=datetime.strptime(to_date.strftime("%Y-%m-%d"), "%Y-%m-%d")
        )

        if announcements and 'Table' in announcements:
            for ann in announcements['Table'][:10]: # Analyze top 10 recent announcements
                subject = ann.get('HEADLINE', '').lower()
                
                # Filter out announcements containing irrelevant keywords
                if any(keyword in subject for keyword in IRRELEVANT_ANNOUNCEMENT_KEYWORDS):
                    continue
                
                sentiment_data = analyze_text_sentiment(ann.get('HEADLINE'))
                
                analyzed_announcements.append({
                    "title": ann.get('HEADLINE'),
                    "link": f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{ann.get('ATTACHMENTNAME')}",
                    "publisher": "BSE Announcement",
                    "sentiment": sentiment_data.get('label'),
                    "sentiment_score": sentiment_data.get('score'),
                    "interpretation": sentiment_data.get('interpretation'),
                    "type": "Announcement"
                })
        return analyzed_announcements

    except Exception as e:
        print(f"Error fetching BSE announcements for {company_name}: {e}")
        return []


# --- Main Function ---

def get_news_and_announcements(ticker_symbol, company_name):
    """
    Fetches news from Google and announcements from BSE, analyzes their sentiment,
    and returns a combined, sorted list.
    """
    if not sentiment_analyzer:
        return [{"error": "Sentiment analysis model is not available."}]

    if not company_name:
        return [{"error": "Company name is required for news analysis."}]

    # Fetch from both primary sources in parallel (in a real-world app, use threading)
    google_news = get_google_news(company_name)
    bse_announcements = get_bse_announcements(ticker_symbol)

    # Combine the results
    combined_results = google_news + bse_announcements
    
    # If both new methods fail, fall back to yfinance as a last resort
    if not combined_results:
        print("Primary sources failed, falling back to yfinance for news.")
        try:
            ticker = yf.Ticker(ticker_symbol+".NS")  # Append '.NS' for NSE tickers
            news = ticker.news
            for article in news[:5]:
                title = article["content"].get('title')
                if not title: continue
                sentiment_data = analyze_text_sentiment(title)
                combined_results.append({
                    "title": title,
                    "link": article["content"]["clickThroughUrl"].get('url'),
                    "publisher": article["content"].get('publisher'),
                    "sentiment": sentiment_data.get('label'),
                    "sentiment_score": sentiment_data.get('score'),
                    "interpretation": sentiment_data.get('interpretation'),
                    "type": "News (YFinance)"
                })
        except Exception as e:
            print(f"yfinance fallback also failed: {e}")

    # Sort results to show the most impactful (highest sentiment score) items first
    sorted_results = sorted(combined_results, key=lambda x: x.get('sentiment_score', 0), reverse=True)
    
    return sorted_results if sorted_results else [{"error": f"No news or announcements found for {company_name}."}]

if __name__ == '__main__':
    # Example usage with an Indian stock ticker (NSE)
    # print(get_bse_announcements("Glenmark"))
    stock_ticker = "GLENMARK" 
    company_name = "Glenmark Pharmaceuticals Ltd"
    analysis_results = get_news_and_announcements(stock_ticker,company_name)
    
    # Print the results in a readable format
    print(f"\nSentiment Analysis for {stock_ticker} News:\n")
    
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
            print(f"type: {result['interpretation']}")
            print("-" * 30)
