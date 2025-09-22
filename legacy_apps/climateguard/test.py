import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime
import re

@dataclass
class Article:
    title: str
    content: str
    url: str
    date: str
    topic: str
    source: str

class VentasbalsScraper:
    def _scrape_presidentlv(self, url):
        print(url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extracting title
        title = soup.find("h1").text.strip() if soup.find("h1") else ""

        # Extracting content
        content = " ".join([p.text for p in soup.find_all("p")])

        # Extracting topic (if available)
        topic = (
            soup.find("meta", {"property": "og:title"})["content"]
            if soup.find("meta", {"property": "og:title"})
            else ""
        )

        # Extracting date (if available)
        date = (
            soup.find("time", {"class": "entry-date published"}).text.strip()
            if soup.find("time", {"class": "entry-date published"})
            else ""
        )

        return Article(title=title, content=content, url=url, date=date, topic=topic, source=url)

# Usage example
if __name__ == "__main__":
    scraper = VentasbalsScraper()
    url = "https://www.president.lv/lv/jaunums/valsts-prezidents-18-novembri-rigas-pili-pasniedz-augstakos-latvijas-valsts-apbalvojumus-90-izcilam-personibam"
    article = scraper._scrape_presidentlv(url)
    if article:
        print(f"Title: {article.title}")
        print(f"Date: {article.date}")
        print(f"Topic: {article.topic}")
        print(f"Content preview: {article.content[:200]}...")
    else:
        print("Failed to scrape the article.")
