from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import json
from models import Article
from newspaper import Article as NewspaperArticle
from newspaper.article import ArticleException
import multiprocessing
from functools import partial


class NewsScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def scrape_article(self, url):
        try:
            # Try NewspaperArticle first
            newspaper_article = NewspaperArticle(url)
            try:
                newspaper_article.download()
                newspaper_article.parse()

                if newspaper_article.text:
                    return Article(
                        title=newspaper_article.title,
                        content=newspaper_article.text,
                        url=url,
                        date=(
                            str(newspaper_article.publish_date)
                            if newspaper_article.publish_date
                            else ""
                        ),
                        topic="",  # NewspaperArticle doesn't provide a topic
                        source=url,
                    )
            except ArticleException:
                print(f"NewspaperArticle failed for {url}. Falling back to custom scraper.")

            # If NewspaperArticle fails, use custom scrapers
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, "html.parser")
            domain = urlparse(url).netloc

     
            if 'lsm.lv' in domain:
                return self._scrape_lsm(soup, url)
            elif 'delfi.lv' in domain:
                return self._scrape_delfi(soup, url)
            elif 'nra.lv' in domain:
                return self._scrape_nra(soup, url)
            elif 'la.lv' in domain:
                return self._scrape_la(soup, url)
            elif 'diena.lv' in domain:
                return self._scrape_diena(soup, url)
            elif 'ventasbalss.lv' in domain:
                return self._scrape_ventasbalss(soup, url)
            elif 'reitingi.lv' in domain:
                return self._scrape_reitingi(soup, url)
            elif 'bnn.lv' or 'ir.lv' or 'latgaleslaiks.lv'in domain:
                return self._scrape_bnn_or_ir_or_latgaleslaiks(soup, url)
            elif 'president.lv' or 'ogrenet.lv' in domain:
                print("Unsupported website")
            else:
                raise ValueError("Unsupported website")
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def _scrape_lsm(self, soup, url):
        content = " ".join([p.text for p in soup.find_all("p")])
        title = soup.find("h1").text.strip() if soup.find("h1") else ""
        topic = (
            soup.find("meta", {"property": "article:section"})["content"]
            if soup.find("meta", {"property": "article:section"})
            else ""
        )
        date = (
            soup.find("meta", {"property": "article:published_time"})["content"]
            if soup.find("meta", {"property": "article:published_time"})
            else ""
        )

        return Article(title=title, content=content, url=url, date=date, topic=topic, source=url)

    def _scrape_bnn_or_ir_or_latgaleslaiks(self, soup, url):

        # Extracting title
        title = soup.find("h1").text.strip() if soup.find("h1") else ""

        # Extracting content
        content = " ".join([p.text for p in soup.find_all("p")])

        # Extracting topic (usually under a breadcrumb or meta tag)
        topic = (
            soup.find("meta", {"property": "article:section"})["content"]
            if soup.find("meta", {"property": "article:section"})
            else ""
        )

        # Extracting date (if available)
        date = (
            soup.find("time", {"class": "entry-date published updated"}).text.strip()
            if soup.find("time", {"class": "entry-date published updated"})
            else ""
        )

        return Article(title=title, content=content, url=url, date=date, topic=topic, source=url)

    def _scrape_reitingi(self, soup, url):
        title = soup.find("h1").text.strip() if soup.find("h1") else ""
        content = " ".join([p.text for p in soup.find_all("p")])
        topic = soup.find("meta", {"name": "news_keywords"})["content"] if soup.find("meta", {"name": "news_keywords"}) else ""
        date = soup.find("div", class_="date").text.strip() if soup.find("div", class_="date") else ""

        return Article(title=title, content=content, url=url, date=date, topic=topic, source=url)


    def _scrape_delfi(self, soup, url):
        content = " ".join([p.text for p in soup.find_all("p", class_="C-article-body__paragraph")])
        title = (
            soup.find("h1", class_="C-article-headline").text.strip()
            if soup.find("h1", class_="C-article-headline")
            else ""
        )
        topic = (
            soup.find("a", class_="C-article-info__category").text.strip()
            if soup.find("a", class_="C-article-info__category")
            else ""
        )
        date = (
            soup.find("time", class_="C-article-info__time")["datetime"]
            if soup.find("time", class_="C-article-info__time")
            else ""
        )
        if not content:
            content_div = soup.find_all("article", class_="mt-4")
            if content_div:
                content = " ".join(p.text for p in content_div[0].findAll("section"))
                title = " ".join(p.text for p in content_div[0].findAll("h1"))

        return Article(title=title, content=content, url=url, date=date, topic=topic, source=url)

    def _scrape_nra(self, soup, url):
        content = " ".join([p.text for p in soup.find_all("p", class_="article-text")])
        title = (
            soup.find("h1", class_="article-title").text.strip()
            if soup.find("h1", class_="article-title")
            else ""
        )
        topic = (
            soup.find("span", class_="article-category").text.strip()
            if soup.find("span", class_="article-category")
            else ""
        )
        date = (
            soup.find("time", class_="article-date")["datetime"]
            if soup.find("time", class_="article-date")
            else ""
        )
        if not content:
            content_div = soup.find_all("div", class_="text-bl-wrap")
            if content_div:
                content = " ".join(p.text for p in content_div[0].findAll("p"))

            title_div = soup.find_all("div", class_="section-title")
            if title_div:
                title = " ".join(p.text for p in title_div[0].findAll("h1"))

        return Article(title=title, content=content, url=url, date=date, topic=topic, source=url)

    
    def _scrape_ventasbalss(self, soup, url):
        try:
            
            title = soup.find('h1', class_='article-title').text.strip() if soup.find('h1', class_='article-title') else ''
            content_div = soup.find('div', class_='article-content')
            paragraphs = content_div.find_all(['p', 'h2', 'h3']) if content_div else []
            content = ' '.join([p.text.strip() for p in paragraphs])
            
            return Article(
                title=title,
                content=content,
                url=url,
                date="",
                topic="",
                source=url
            )
        except Exception as e:
            print(f"Error scraping ventasbalss.lv: {str(e)}")
            return None


    def _scrape_la(self, soup, url):
        try:
            title = soup.find('article').find('h1', class_='article-title').text.strip() if soup.find('article').find('h1', class_='article-title') else ''
            
            date_str = soup.find('article').find('div', class_='article-date').text.strip() if soup.find('article').find('div', class_='article-date') else ''
            
            content_div = soup.find('article', class_='article-content-block')
            
            
            # Extract content from p tags
            paragraphs = content_div.find_all('p') if content_div else []
            content = ' '.join([p.text.strip() for p in paragraphs])
            
            topic = soup.find('div', class_='article-breadcrumbs').find_all('a')[-1].text.strip() if soup.find('div', class_='article-breadcrumbs').find_all('a') else ''
            
            return Article(
                title=title,
                content=content,
                url=url,
                date=date_str,
                topic=topic,
                source=url
            )
        except AttributeError as e:
            print(f"Error scraping la.lv: {str(e)}")
            return None

    def _parse_la_date(self, date_str):
        # Convert Latvian month names to numbers
        lv_months = {
            'janvāris': '01', 'februāris': '02', 'marts': '03', 'aprīlis': '04',
            'maijs': '05', 'jūnijs': '06', 'jūlijs': '07', 'augusts': '08',
            'septembris': '09', 'oktobris': '10', 'novembris': '11', 'decembris': '12'
        }
        
        # Split the date string
        day, month, year = date_str.split()
        
        # Convert month to number
        month_num = lv_months[month.lower()]
        
        # Format the date as YYYY-MM-DD
        return f"{year}-{month_num}-{day.zfill(2)}"

    def _scrape_diena(self, soup, url):
        try:
            title = soup.find('h1', class_='article-headline').text.strip() if soup.find('h1', class_='article-headline') else ''
            content = ' '.join([p.text for p in soup.find('div', class_='article-body').find_all('p')]) if soup.find('div', class_='article-body') else ''
            date = soup.find('time', class_='article-date')['datetime'] if soup.find('time', class_='article-date') else ''
            topic = soup.find('a', class_='article-category').text.strip() if soup.find('a', class_='article-category') else ''
            
            return Article(
                title=title,
                content=content,
                url=url,
                date=date,
                topic=topic,
                source=url
            )
        except AttributeError as e:
            print(f"Error scraping diena.lv: {str(e)}")
            return None

def scrape_single_article(scraper, url):
    article = scraper.scrape_article(url)
    if article:
        if len(article.content) > 100:
            print(f"Scraped: {article.title}")
            print(f"Content length: {len(article.content)}")
            print(f"Date: {article.date}")
            print("---")
            return article, None
        else:
            print(f"Skipped: {article.title} - Content length too short")
            return None, url
    else:
        print(f"Failed to scrape: {url}")
        print("---")
        return None, url

if __name__ == "__main__":
    scraper = NewsScraper()

    # Read URLs from the all_urls.json file
    with open('data/all_urls.json', 'r', encoding='utf-8') as f:
        urls = json.load(f)

    # Create a partial function with the scraper object
    scrape_func = partial(scrape_single_article, scraper)

    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()

    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=4) as pool:
        # Map the scraping function to the URLs in parallel
        results = pool.map(scrape_func, urls)

    # Separate the results into scraped articles and failed URLs
    scraped_articles = [article for article, _ in results if article]
    failed_urls = [url for _, url in results if url]

    # Save successfully scraped articles to JSON
    output_file = 'data/scraped_articles_2.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([article.dict() for article in scraped_articles], f, ensure_ascii=False, indent=4)

    print(f"\nSuccessfully scraped articles saved to {output_file}")

    # Save failed URLs to a separate file
    failed_file = 'data/failed_urls_2.json'
    with open(failed_file, 'w', encoding='utf-8') as f:
        json.dump(failed_urls, f, ensure_ascii=False, indent=4)

    print(f"Failed URLs saved to {failed_file}")