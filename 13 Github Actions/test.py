from pathlib import Path
import datetime
import pytz
from itertools import chain
import requests
from bs4 import BeautifulSoup
html_content = requests.get("https://www.winterthurer-zeitung.ch/").text
soup = BeautifulSoup(html_content)
main_headline = soup.select(".article h2 a")[0].text.strip()
main_headline
