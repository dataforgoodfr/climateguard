import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import json
from models import Article
from newspaper import Article as NewspaperArticle
from urllib.parse import urlparse

class NewsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scrape_article(self, url):
        # Try NewspaperArticle first
        newspaper_article = NewspaperArticle(url)
        newspaper_article.download()
        newspaper_article.parse()

        if newspaper_article.text:
            return Article(
                title=newspaper_article.title,
                content=newspaper_article.text,
                url=url,
                date=str(newspaper_article.publish_date) if newspaper_article.publish_date else '',
                topic='',  # NewspaperArticle doesn't provide a topic
                source=url
            )
        
        # If NewspaperArticle fails to extract text, use custom scrapers
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        if 'lsm.lv' in url:
            return self._scrape_lsm(soup, url)
        elif 'delfi.lv' in url:
            return self._scrape_delfi(soup, url)
        elif 'nra.lv' in url:
            return self._scrape_nra(soup, url)
        else:
            raise ValueError("Unsupported website")

    def _scrape_lsm(self, soup, url):
        content = ' '.join([p.text for p in soup.find_all('p')])
        title = soup.find('h1').text.strip() if soup.find('h1') else ''
        topic = soup.find('meta', {'property': 'article:section'})['content'] if soup.find('meta', {'property': 'article:section'}) else ''
        date = soup.find('meta', {'property': 'article:published_time'})['content'] if soup.find('meta', {'property': 'article:published_time'}) else ''
        
        return Article(
            title=title,
            content=content,
            url=url,
            date=date,
            topic=topic,
            source=url
        )

    def _scrape_delfi(self, soup, url):
        content = ' '.join([p.text for p in soup.find_all('p', class_='C-article-body__paragraph')])
        title = soup.find('h1', class_='C-article-headline').text.strip() if soup.find('h1', class_='C-article-headline') else ''
        topic = soup.find('a', class_='C-article-info__category').text.strip() if soup.find('a', class_='C-article-info__category') else ''
        date = soup.find('time', class_='C-article-info__time')['datetime'] if soup.find('time', class_='C-article-info__time') else ''
        
        return Article(
            title=title,
            content=content,
            url=url,
            date=date,
            topic=topic,
            source=url
        )

    def _scrape_nra(self, soup, url):
        content = ' '.join([p.text for p in soup.find_all('p', class_='article-text')])
        title = soup.find('h1', class_='article-title').text.strip() if soup.find('h1', class_='article-title') else ''
        topic = soup.find('span', class_='article-category').text.strip() if soup.find('span', class_='article-category') else ''
        date = soup.find('time', class_='article-date')['datetime'] if soup.find('time', class_='article-date') else ''
        
        return Article(
            title=title,
            content=content,
            url=url,
            date=date,
            topic=topic,
            source=url
        )

# Usage example:
if __name__ == "__main__":
    scraper = NewsScraper()
    urls = [
        "https://www.lsm.lv/raksts/dzive--stils/vide-un-dzivnieki/03.10.2024-zinojums-lidz-gadsimta-beigam-latvija-prognozeta-krasta-linijas-atkapsanas-par-47-72-metriem.a571093/",
        "https://www.delfi.lv/bizness/56234200/eiropas-zinas/120042670/zinam-problemu-un-neizmantojam-risinajumus-ko-latvijas-iedzivotaji-doma-par-klimata-parmainam",
        "https://www.delfi.lv/bizness/56234200/eiropas-zinas/120042670/kutri-izmantojam-dzerama-udens-kranus-kapec-iedzivotajiem-trukst-pamudinajuma-dzivot-zalak",
        "https://nra.lv/pasaule/465572-sliktas-zinas-baltvina-cienitajiem.htm",
        "https://www.lsm.lv/raksts/dzive--stils/vide-un-dzivnieki/20.09.2024-par-zalaku-rigu-spriedis-piecas-sestdienas-ko-sagaida-no-pirmas-iedzivotaju-klimata-asamblejas.a569637/"
    ]

    articles = []

    for url in urls:
        article = scraper.scrape_article(url)
        articles.append(article)
        print(f"Scraped: {article.title}")
        print(f"Content length: {len(article.content)}")
        print(f"Date: {article.date}")
        print("---")

    # Save to JSON
    output_file = 'scraped_articles.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([article.dict() for article in articles], f, ensure_ascii=False, indent=4)

    print(f"\nArticles saved to {output_file}")