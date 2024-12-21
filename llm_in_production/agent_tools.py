import os

import pandas as pd
import requests
import yfinance as yf


def get_stock_prices(company_stock_ticker_symbol: str, days: int = 7) -> dict:
    # YOUR CODE HERE START
    """Get recent information about a given stock.

    :param company_stock_ticker_symbol: The stock ticker symbol for a given company, e.g. Microsoft is "MSFT".
    :param days: How far to look back
    :return: The recent price information about a given stock.
    """

    # time.sleep(4) #To avoid rate limit error
    stock = yf.Ticker(company_stock_ticker_symbol)
    df = stock.history(period=f"{days}d")
    df = df[["Close", "Volume"]]
    df.index = [str(x).split()[0] for x in list(df.index)]
    df.index.rename("Date", inplace=True)
    # YOUR CODE HERE END
    return df.to_json()


def get_news_stories(topic: str) -> dict:
    # YOUR CODE HERE START
    """Get the (stock) topic headlines from a given topic

    :param topic: The (stock) topic you want to retrieve news stories about, e.g. "Microsoft".
    :return: The top headlines for that (stock) topic
    """
    today = pd.Timestamp.today()
    start = today.date
    end = today - pd.Timedelta(days=30)
    end = end.date
    for term in ["stock", "news"]:
        if term not in topic:
            topic += f" {term}"
    request = requests.get(
        f"https://newsapi.org/v2/everything?q={topic}&from={start}&to={end}&sortBy=popularity&apiKey={os.environ['NEWS_API_KEY']}"
    ).json()
    articles = pd.DataFrame(request["articles"])
    titles = articles["title"]
    # YOUR CODE HERE END
    return titles.to_json()


def get_current_weather(location: str) -> dict:
    """Get the current weather conditions in a given location

    :param location: The city (and state), e.g. "San Francisco, CA"
    :return: The weather conditions
    """
    return requests.get(
        f"https://api.weatherapi.com/v1/current.json?key={os.environ['WEATHER_API_KEY']}&q={location}"
    ).json()
