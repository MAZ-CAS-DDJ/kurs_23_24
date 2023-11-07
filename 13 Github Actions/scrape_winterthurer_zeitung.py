from pathlib import Path
import datetime
import pytz
from itertools import chain
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    html_content = requests.get("https://www.winterthurer-zeitung.ch/").text
    soup = BeautifulSoup(html_content)
    main_headline = soup.select(".article h2 a")[0].text.strip()
    main_headline = "\n- " + datetime.datetime.now().strftime("%H:%M %d.%m.%y") + " " + main_headline
    with open("SCRAPE.md", "a",) as f:
        f.write(main_headline)
    print("Hat alles geklappt.")