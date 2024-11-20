from get_news_data import news_api
from get_news_data import cleaner

#example usage(as many pages of API output as possible to output file)
##################news_api.get_all_articles('tesla', 'TSLA', '2024-09-23', '2024-10-20')
#stock: stock name
#stock_ticker: stock ticker
#from_date: articles on or after, in form YYYY-MM-DD
#to_date: articles on or before, in form YYYY-MM-DD
#of, optional, the name of the file to output to.  Default is <stock>.json

#example usage(page 1 of API output to output file)
#news_api.get_articles('tesla', 'TSLA', '2024-10-04', '2024-10-31', 1)
#stock: stock name
#stock_ticker: stock ticker
#from_date: articles on or after, in form YYYY-MM-DD
#to_date: articles on or before, in form YYYY-MM-DD
#page: which page of the API output to look at
#of, optional, the name of the variable for an existing open file.
    # if none is given, then <stock>.json is created and used



#THE CURRENT tesla.json file is around 5 pages of API output
#Json info:

#title_has_name: is the name of the stock in the title
    #other similar fields work the same
#all of the ones that end in _neg, _neu, _pos, _compound are different
#sentiment analysis result fields.

news_api.get_all_articles('Dell', 'DELL', '2024-11-01', '2024-11-19')