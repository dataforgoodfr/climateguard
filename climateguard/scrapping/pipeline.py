from climateguard.scrapping.urls_from_gdelt import GDELTScrapper
from climateguard.scrapping.articles import NewsScraper
from pathlib import Path
import json
import multiprocessing
from functools import partial

class ScrapFromGDelt:
    def __init__(self):
        self.gdelt_scraper = GDELTScrapper()
        self.news_scraper = NewsScraper()

    def run(self, keyword: str, years: list[int], output_dir: Path):
        # Step 1: Find themes related to the keyword
        themes = self.gdelt_scraper.find_themes_related_to_keyword(keyword)
        print(f"Themes related to {keyword}: {themes}")

        # Step 2: Find articles for these themes and years
        articles_df = self.gdelt_scraper.find_articles(themes=themes, years=years)

        # Step 3: Extract URLs from the DataFrame
        urls = articles_df["url"].tolist()

        # Save the list of URLs to a separate file
        self._save_urls(urls, output_dir)

        # Step 4: Scrape each URL using multiprocessing
        scraped_articles, failed_urls = self._scrape_urls_parallel(urls)

        # Step 5: Save results
        self._save_results(scraped_articles, failed_urls, output_dir)

    def _save_urls(self, urls: list, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        urls_file = output_dir / 'all_urls.json'
        with open(urls_file, 'w', encoding='utf-8') as f:
            json.dump(urls, f, ensure_ascii=False, indent=4)
        print(f"All URLs saved to {urls_file}")

    def _scrape_urls_parallel(self, urls):
        # Create a partial function with self.news_scraper
        scrape_func = partial(self._scrape_single_url, news_scraper=self.news_scraper)

        # Use all available cores
        num_cores = multiprocessing.cpu_count()
        
        # Create a multiprocessing pool
        with multiprocessing.Pool(num_cores) as pool:
            results = pool.map(scrape_func, urls)

        # Process results
        scraped_articles = []
        failed_urls = []
        for result in results:
            if result['success']:
                article = result['article']
                scraped_articles.append(article)
                print(f"Scraped: {article.title}")
                print(f"Content length: {len(article.content)}")
                print(f"Date: {article.date}")
                print("---")
            else:
                failed_urls.append(result['url'])
                print(f"Failed to scrape: {result['url']}")
                print("---")

        return scraped_articles, failed_urls

    @staticmethod
    def _scrape_single_url(url, news_scraper):
        article = news_scraper.scrape_article(url)
        if article:
            return {'success': True, 'article': article}
        else:
            return {'success': False, 'url': url}

    def _save_results(self, scraped_articles, failed_urls, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save successfully scraped articles to JSON
        output_file = output_dir / 'scraped_articles.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([article.dict() for article in scraped_articles], f, ensure_ascii=False, indent=4)

        print(f"\nSuccessfully scraped articles saved to {output_file}")

        # Save failed URLs to a separate file
        failed_file = output_dir / 'failed_urls.json'
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_urls, f, ensure_ascii=False, indent=4)

        print(f"Failed URLs saved to {failed_file}")

if __name__ == "__main__":
    pipeline = ScrapFromGDelt()
    output_dir = Path(__file__).parent.parent / "data"
    pipeline.run(keyword="CLIMATE", years=[2022, 2023, 2024], output_dir=output_dir)
