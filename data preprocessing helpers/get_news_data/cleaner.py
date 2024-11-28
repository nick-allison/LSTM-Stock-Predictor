import json
from datetime import datetime

from . import sentiment_analysis

#timestamp string to unix time
def to_unix(str):
    dt = datetime.strptime(str, "%Y-%m-%dT%H:%M:%SZ")
    return int(dt.timestamp())

#saves time later; self explanatory functionality
def x_has_y(x, y):
    if y in x:
        return 1
    else:
        return 0

def clean(json_obj, stock_name, stock_ticker):
    #sentiment analysis of title and body
    title_sent = sentiment_analysis.analyze(json_obj['title'])
    body_sent = sentiment_analysis.analyze(json_obj['content'])

    #gives back relevant article info
    return {
        'url': json_obj['url'],
        'stock_name': stock_name,
        'stock_ticker': stock_ticker,
        'datetime':to_unix(json_obj['publishedAt']),
        'source': json_obj['source']['id'],
        'title_has_name': x_has_y(json_obj['title'].lower(), stock_name.lower()),
        'title_has_ticker': x_has_y(json_obj['title'].lower(), stock_ticker.lower()),
        'title_neg':title_sent['neg'],
        'title_neu':title_sent['neu'],
        'title_pos':title_sent['pos'],
        'title_compound':title_sent['compound'],
        'body_has_name': x_has_y(json_obj['content'].lower(), stock_name.lower()),
        'body_has_ticker':x_has_y(json_obj['content'].lower(), stock_ticker.lower()),
        'body_neg':body_sent['neg'],
        'body_neu':body_sent['neu'],
        'body_pos':body_sent['pos'],
        'body_compound':body_sent['compound'],
        'other':
                {
                    'title': json_obj['title'],
                    'body': json_obj['content'],
                    'author':json_obj['author']
                }
    }