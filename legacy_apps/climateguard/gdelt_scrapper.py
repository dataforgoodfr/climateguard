from urllib.request import urlopen
import pandas as pd
import gdeltdoc as gdelt
import functools
import itertools
from pathlib import Path

class GDELTScrapper: 
    THEMES_URL = "http://data.gdeltproject.org/api/v2/guides/LOOKUP-GKGTHEMES.TXT"

    @functools.cached_property
    def themes_df(self) -> pd.DataFrame:    
        # Fetch the content using urllib
        with urlopen(self.THEMES_URL) as response:
            data = response.read().decode()
        
        # Split the data into lines
        lines = data.strip().split("\n")
        
        # Split each line into key-value pairs
        rows = [line.split("\t") for line in lines]
        
        # Create a DataFrame from the rows
        df = pd.DataFrame(rows, columns=['theme', 'count'])
        df['count'] = df['count'].astype(int)
        
        return df

    def find_themes_related_to_keyword(self, keyword: str) -> list[str]: 
        return self.themes_df[self.themes_df["theme"].str.contains(keyword, case=False)]["theme"].to_list()

    def find_articles(self, themes: list[str], years: list[int]) -> pd.DataFrame: 
        partial_articles_dfs = []

        gd = gdelt.GdeltDoc()
        for theme, year in itertools.product(themes, years): 
            f = gdelt.Filters(
                #keyword = "climate change",
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31", 
                theme=theme, 
                country="LG", # Latvia
            )
        
            partial_articles_df = gd.article_search(f)
            print(f"{len(partial_articles_df)} articles found for theme {theme}, in {year}")
            partial_articles_dfs.append(partial_articles_df)

        articles_df = pd.concat(partial_articles_dfs)
            
        articles_df = articles_df[articles_df["language"] == "Latvian"]
        articles_df["seendate"] = pd.to_datetime(articles_df["seendate"])

        print(f"Deleting {articles_df["url"].duplicated().sum()} duplicates")
        articles_df = articles_df.drop_duplicates("url")
        print(f"{len(articles_df)} unique articles found")
        return articles_df


# Usage example:
if __name__ == "__main__":
    scraper = GDELTScrapper()

    # Find themes related to climate
    themes = scraper.find_themes_related_to_keyword("CLIMATE")
    print(f"Themes related to climate: {themes}")

    # Find articles for these themes and year range
    articles_df = scraper.find_articles(themes=themes, years=[2022, 2023, 2024])

    # This can be used as input for NewsScraper
    article_urls = articles_df["url"].to_list()

    # Save dataframe to a csv file
    file_path = Path(__file__).parent.parent / "data/latvian_article_links.csv"
    articles_df.to_csv(file_path)