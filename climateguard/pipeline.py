import asyncio
from pathlib import Path
import json
from typing import Literal, List, Dict

from climateguard.scrapping.pipeline import ScrapFromGDelt
from climateguard.detect_claims import detect_claims
from climateguard.models import Article, Claims

class Pipeline:
    def __init__(self):
        self.scrap_from_gdelt = ScrapFromGDelt()
        self.data_dir = Path(__file__).parent / "data"

    def run(self, keyword: str, years: list[int], language: Literal["French", "English", "Latvian"]) -> List[Dict]:
        scraped_articles_file = self.data_dir / 'scraped_articles.json'

        if not scraped_articles_file.exists():
            print("Scraped articles not found. Starting scraping process...")
            articles_data = self.scrap_from_gdelt.run(keyword, years, self.data_dir)
        else:
            print("Scraped articles found. Loading from file...")
            with open(scraped_articles_file, 'r', encoding='utf-8') as f:
                articles_data = json.load(f)

        # Process articles and detect claims
        results = []
        for article_data in articles_data:
            article = Article(**article_data)
            claims, n_tokens = detect_claims(article, language)
            results.append({
                "article": article.dict(),
                "claims": claims.dict(),
                "n_tokens": n_tokens
            })

        print(f"Processed {len(results)} articles with claims")
        return results

def main():
    pipeline = Pipeline()
    processed_articles = pipeline.run(keyword="CLIMATE", years=[2022, 2023, 2024], language="English")
    

if __name__ == "__main__":
    main()
