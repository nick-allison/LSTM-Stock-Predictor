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
def get_all_articles(stock, stock_ticker, from_date, to_date, of = None):
    if of == None:
        outfile = open(stock + ".json", 'w')
    else:
        outfile = open(of, 'w')
    page = 1
    #api call
    all_articles = newsapi.get_everything(q=stock,
                                      sources=source_str,
                                      from_param= from_date,
                                      to=to_date,
                                      language='en',
                                      sort_by='relevancy',
                                      page=page)
    while all_articles != None:
        for i in all_articles['articles']: #for each article the api returns
            if i['url'] != 'https://removed.com': #avoids empty entry
                    outfile.write(json.dumps(cleaner.clean(i, stock, stock_ticker), indent=4)) #runs clean function to get important info before writing to file
        page = page + 1
        try:
            #api call
            all_articles = newsapi.get_everything(q=stock,
                                      sources=source_str,
                                      from_param= from_date,
                                      to=to_date,
                                      language='en',
                                      sort_by='relevancy',
                                      page=page)
        except:
            print("Error trying to pull page " + str(page))
            page = page - 1
            # when the api throws an error for too many requests in a short time,
            # this attempts to bypass this by waiting 2 seconds and 
            # then trying to get the same page value that failed
            time.sleep(2)
    outfile.close()



