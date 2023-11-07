import requests
from bs4 import BeautifulSoup
from datetime import datetime
now = datetime.now()

r = requests.get('https://www.pls-zh.ch/plsFeed/rss') 
soup = BeautifulSoup(r.text, 'html.parser')
items_list = soup.find_all('item')
parkings = []
for item in items_list:
    open_places = item.find('description').text
    open_places_just_number = ''.join(x for x in open_places if x.isdigit())
    parkings.append([item.find('title').text,open_places_just_number,now.strftime("%d.%m.%y %H:%M:%S")])
with open("PARKING.csv", "a") as f:
    for entry in parkings:
        # write each item on a new line
        f.write("%s\n" % ",".join(entry))
    print('Done')


