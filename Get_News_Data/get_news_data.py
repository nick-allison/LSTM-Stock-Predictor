from bs4 import BeautifulSoup
import requests
import json

stock_name = "tesla"
outfile = open(stock_name + ".json", 'w')

#AP news search page for stock in question
search_page = requests.get("https://apnews.com/search?q=" + stock_name + "#nt=navsearch")


#Use Beautiful Soup to parse 

soup = BeautifulSoup(search_page.content, 'html.parser')

#list of html div tags that have a link to an article about stock on the search page
divs = soup.findAll('div', {'class': 'PagePromo-title'})

#get urls out and add them to 'links' list
links = []
for i in divs:
    links.append(i.find('a').get('href'))

#go through each link and get info
for i in links:
    article_page = requests.get(i)
    soup2 = BeautifulSoup(article_page.content, 'html.parser')

    #get article body text
    div = soup2.find('div', {'class': 'RichTextStoryBody RichTextBody'})
    if div:
        #get timestampt from webpage
        timestamp = soup2.find('bsp-timestamp').get('data-timestamp')
        all_ps = div.findAll('p')
        str = ""
        for p in all_ps:
            str = str + p.text + "\n\n"
             #add text and timestampt to a json file
            dict = {"link": i,
            "timestamp": timestamp,
            "text": str}
            json_object = json.dumps(dict, indent=4)
            #write to json
            outfile.write(json_object)

outfile.close()

