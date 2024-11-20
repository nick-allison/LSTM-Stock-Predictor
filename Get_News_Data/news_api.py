# Get_News_data/news_api.py

from newsapi import NewsApiClient
import json
import time

from . import cleaner

#API Key
api_key_ = 'b6b911e095964c6597c8689407f7cebb'

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
#top_headlines = newsapi.get_top_headlines(q=stock_name,
                                          #sources= source_str,
                                          #category='business',
                                          #language='en',
                                          #country='us'
                                         # )


#Samply query
#all_articles = newsapi.get_everything(q=stock_name,
                                      #sources=source_str,
                                      #from_param='2024-09-22',
                                      #to='2024-10-20',
                                      #language='en',
                                      #sort_by='relevancy',
                                      #page=2)


#of is the open file: gets 1 page of results
def get_articles(stock, stock_ticker, from_date, to_date, page, of = None):
    all_articles = newsapi.get_everything(q=stock,
                                      sources=source_str,
                                      from_param= from_date,
                                      to=to_date,
                                      language='en',
                                      sort_by='relevancy',
                                      page=page)
    if of == None:
        of = open(stock + ".json", 'w')
    for i in all_articles['articles']:
        if i['url'] != 'https://removed.com':
            of.write(json.dumps(cleaner.clean(i, stock, stock_ticker), indent=4))
    of.close()

#of is the name of the file to write to.  Tries to get every page, 
# but may only get several(free plan restrictions)

def get_all_articles(stock, stock_ticker, from_date, to_date, of=None):
    # Collect articles in a list
    all_articles_list = []
    page = 1
    max_retries = 5  # Limit retries per page
    retries = 0

    while True:
        try:
            # API call to get articles
            all_articles = newsapi.get_everything(
                q=stock,
                sources=source_str,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='relevancy',
                page=page
            )

            # Check if articles are available
            if all_articles is None or len(all_articles['articles']) == 0:
                print(f"No more articles found at page {page}. Stopping...")
                break

            # Add articles to the list after cleaning them
            for article in all_articles['articles']:
                if article['url'] != 'https://removed.com':  # Avoids empty entry
                    cleaned_article = cleaner.clean(article, stock, stock_ticker)
                    all_articles_list.append(cleaned_article)

            # Move to the next page
            page += 1
            retries = 0  # Reset retries after a successful fetch

        except Exception as e:
            print(f"Error trying to pull page {page}: {str(e)}")
            retries += 1

            if retries >= max_retries:
                print(f"Max retries reached for page {page}. Moving on...")
                break

            # Pause before retrying
            time.sleep(2)

    # Write all collected articles to the output file as a single JSON array
    if of is None:
        of = f"{stock}.json"

    try:
        with open(of, 'w') as outfile:
            json.dump(all_articles_list, outfile, indent=4)
            print(f"Successfully written all articles to {of}")
    except Exception as e:
        print(f"Error writing articles to file: {str(e)}")