import pandas as pd
import json
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NewsAPI:
    def __init__(self, dataset_path):
        # Load the dataset
        self.data = pd.read_csv(dataset_path)
        # Initialize VADER Sentiment Analyzer
        self.analyzer = SentimentIntensityAnalyzer()

    @staticmethod
    def to_unix(timestamp_str):
        """Convert a datetime string to Unix time, handling multiple formats."""
        formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]  # List of possible formats
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                return int(dt.timestamp())
            except ValueError:
                continue
        raise ValueError(f"Time data '{timestamp_str}' does not match any known formats.")

    @staticmethod
    def contains_keyword(text, keyword):
        """Check if the text contains the keyword (case-insensitive)."""
        if pd.isna(text):
            return 0
        return int(keyword.lower() in text.lower())

    def analyze_sentiment(self, text):
        """Analyze sentiment of a given text using VADER."""
        if pd.isna(text):
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return self.analyzer.polarity_scores(text)

    def clean_article(self, row, stock_name, stock_ticker):
        """Build JSON object for an article."""
        title_sent = self.analyze_sentiment(row['title'])
        body_sent = self.analyze_sentiment(row.get('article', ''))  # Default to empty string if missing
        return {
            "url": row['url'],
            "stock_name": stock_name,
            "stock_ticker": stock_ticker,
            "datetime": self.to_unix(row['date']),
            "source": row['publication'],
            "title_has_name": self.contains_keyword(row['title'], stock_name),
            "title_has_ticker": self.contains_keyword(row['title'], stock_ticker),
            "title_neg": title_sent['neg'],
            "title_neu": title_sent['neu'],
            "title_pos": title_sent['pos'],
            "title_compound": title_sent['compound'],
            "body_has_name": self.contains_keyword(row.get('article', ''), stock_name),
            "body_has_ticker": self.contains_keyword(row.get('article', ''), stock_ticker),
            "body_neg": body_sent['neg'],
            "body_neu": body_sent['neu'],
            "body_pos": body_sent['pos'],
            "body_compound": body_sent['compound'],
            "other": {
                "title": row['title'],
                "body": row.get('article', ''),
                "author": row['author']
            }
        }

    def get_all_articles(self, stock_name, stock_ticker, output_prefix="articles"):
        """Find articles containing the stock name and save JSON output in batches."""
        results = []
        batch_count = 0
        for idx, row in self.data.iterrows():
            try:
                # Skip rows where title or article are missing
                if pd.isna(row['title']):
                    continue
                article_body = row.get('article', '')
                if self.contains_keyword(row['title'], stock_name) or self.contains_keyword(article_body, stock_name):
                    results.append(self.clean_article(row, stock_name, stock_ticker))
                
                # Save every 100 results to avoid data loss
                if len(results) == 100:
                    batch_file = f"{output_prefix}_batch_{batch_count}.json"
                    with open(batch_file, "w") as f:
                        json.dump(results, f, indent=4)
                    print(f"Saved {len(results)} articles to {batch_file}")
                    results = []  # Reset results for the next batch
                    batch_count += 1

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        # Save any remaining results
        if results:
            batch_file = f"{output_prefix}_batch_{batch_count}.json"
            with open(batch_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved {len(results)} articles to {batch_file}")


f_name = '' #Add your CSV file name here
stock_name = '' #Add your stock name here
stock_ticker = '' #Add your stock ticker here

news_api = NewsAPI(f_name)
news_api.get_all_articles(stock_name, stock_ticker)