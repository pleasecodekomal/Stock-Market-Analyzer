

import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import os
api_key = os.getenv("API_KEY")
 
query = 'stock market'

url = f'https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={api_key}'
response = requests.get(url)
data = response.json()

analyzer = SentimentIntensityAnalyzer()
results = []

for article in data.get('articles', []):
    title = article.get('title', '')
    if title:
        sentiment = analyzer.polarity_scores(title)
        results.append({
            'title': title,
            'score': sentiment['compound']
        })

df = pd.DataFrame(results)
print(df)
