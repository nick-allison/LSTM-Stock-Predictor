from newsapi import NewsApiClient
import json

#API Key
api_key_ = 'b6b911e095964c6597c8689407f7cebb'

stock_name = 'tesla'
outfile = open(stock_name + ".json", 'w')

#NewsAPIClient object (used for queries)
newsapi = NewsApiClient(api_key=api_key_)


sources = ['abc-news', 'al-jazeera-english', 'ars-technica', 'associated-press', 'axios', 'bleacher-report', 
           'bloomberg', 'breitbart-news', 'business-insider', 'buzzfeed', 'cbs-news', 'cnn', 'cnn-es', 
           'crypto-coins-news', 'engadget', 'entertainment-weekly', 'espn', 'espn-cric-info', 'fortune', 
           'fox-news', 'fox-sports', 'google-news', 'hacker-news', 'ign', 'mashable', 'medical-news-today', 
           'msnbc', 'mtv-news', 'national-geographic', 'national-review', 'nbc-news', 'new-scientist', 'newsweek', 
           'new-york-magazine', 'next-big-future', 'nfl-news', 'nhl-news', 'politico', 'polygon', 'recode', 
           'reddit-r-all', 'reuters', 'techcrunch', 'techradar', 'the-american-conservative', 'the-hill', 
           'the-huffington-post', 'the-next-web', 'the-verge', 'the-wall-street-journal', 'the-washington-post', 
           'the-washington-times', 'time', 'usa-today', 'vice-news', 'wired']

#make the list of sources into 1 string
source_str = ''
for i in sources:
    source_str += i
    if i != 'wired':
        source_str += ','

#Sample query
top_headlines = newsapi.get_top_headlines(q=stock_name,
                                          sources= source_str,
                                          #category='business',
                                          language='en',
                                          #country='us'
                                          )


#Samply query
all_articles = newsapi.get_everything(q=stock_name,
                                      sources=source_str,
                                      from_param='2024-09-22',
                                      to='2024-09-28',
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)

for i in top_headlines['articles']:
    outfile.write(json.dumps(i, indent=4))

outfile.close()