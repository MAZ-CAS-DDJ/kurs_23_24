from pathlib import Path
import datetime
import pytz
from itertools import chain
import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    html_content = requests.get("https://www.winterthurer-zeitung.ch/").text
    soup = BeautifulSoup(html_content,"lxml")
    main_headline = soup.select(".article.articletype-0 h2 a")[0].text.strip()
    main_headline = "\n- " + datetime.datetime.now().strftime("%H:%M %d.%m.%y") + " " + main_headline
    readme = Path("SCRAPE.md").read_text(encoding="utf8")
    readme_title = "# Latest 84xo Headlines"
    new_readme = readme[:readme.find(readme_title)] + f"{main_headline}\n\n"
    with open("SCRAPE.md", "w+") as f:
        f.write(new_readme)


