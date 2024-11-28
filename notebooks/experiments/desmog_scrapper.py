import csv
import json
import re
import string
from typing import List, Literal
from bs4 import BeautifulSoup
import pandas as pd
import requests
from pydantic import BaseModel
from tqdm import tqdm


class DesmogItem(BaseModel):
    name: str
    countries: List[str]
    url: str
    item_type: Literal["individual", "organization"]

    key_quotes: List[str] = []
    stance_quotes: List[str] = []
    blockquotes: List[str] = []


def get_desmog_items() -> List[DesmogItem]:
    """
    Scrap the database main page to gather links to article pages.
    Returns:
        List[Dict]: name, country, url)
    """
    BASE_URL = "https://www.desmog.com/climate-disinformation-database/"
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.content, "html.parser")
    element_list = soup.find("div", class_="display-view-entries grid-view-entries active")
    results = []
    for x in element_list:
        classes = x.get("class", [])
        if "type-individual" in classes:
            item_type = "individual"
        elif "type-organization" in classes:
            item_type = "organization"
        else:
            print("Item foud neither individual or organization")
            continue

        url = x.find("a")["href"]
        name = x.find("div", class_="grid-view-entry-name entry-name").text.strip()
        countries = x.find("div", class_="grid-view-entry-country").text.strip()
        countries = [country.strip() for country in countries.split(",")]
        results.append(
            DesmogItem(
                name=name,
                countries=countries,
                url=url,
                item_type=item_type,
            )
        )
    return results


def extract_quotes_from_section(sections: List[str]) -> List[str]:
    extracted_quotes = []
    for s in sections:
        extracted_quotes.extend(re.findall(r"“(.*?)”", s))
    return extracted_quotes


def get_article_content(link: str):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")

    content = soup.find("div", class_="elementor-widget-wrap elementor-element-populated")

    divs = content.find_all("div", class_="elementor-widget-container")

    for div in divs:
        if div.find("p"):
            content = div
            break

    sections = {}
    current_section = None
    paragraphs = []
    for element in content.find_all(["h1", "h2", "p"]):
        # Remove all <sup> tags from the current element
        for sup in element.find_all("span"):
            sup.decompose()

        if element.name in ["h1", "h2", "h3"]:
            if current_section and paragraphs:
                sections[current_section] = paragraphs
                paragraphs = []

            current_section = "".join(
                part for part in element.descendants if isinstance(part, str)
            ).strip()
            current_section = current_section.replace("\xa0", " ")
        elif element.name == "p":
            paragraph_text = "".join(
                part for part in element.descendants if isinstance(part, str)
            ).strip()
            paragraphs.append(paragraph_text)
    # Add the last section to the dictionary (if it has any paragraphs)
    if current_section and paragraphs:
        sections[current_section] = paragraphs

    # Extract quotes
    blockquotes_elts = content.find_all("blockquote")
    blockquotes = []
    for quote in blockquotes_elts:
        text = " ".join([p.text[1:-1] for p in quote.find_all("p")])
        text = text.strip()
        if text.endswith("”"):
            text = text[:-1]
        blockquotes.append(text)
    sections["blockquotes"] = blockquotes

    if "Key Quotes" in sections.keys():
        sections["key_quotes"] = extract_quotes_from_section(sections["Key Quotes"])
    if "Stance on Climate Change" in sections.keys():
        sections["stance_quotes"] = extract_quotes_from_section(
            sections["Stance on Climate Change"]
        )

    return sections


def save_desmog_to_csv(items: List[DesmogItem], filename: str):
    """Flatten and save to CSV"""
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "countries",
                "url",
                "item_type",
                "key_quotes",
                "stance_quotes",
                "blockquotes",
            ],
        )
        # Write the header
        writer.writeheader()

        # Write each item
        for item in items:
            writer.writerow(
                {
                    "name": item.name,
                    "countries": json.dumps(item.countries),
                    "url": item.url,
                    "item_type": item.item_type,
                    "key_quotes": json.dumps(item.key_quotes),
                    "stance_quotes": json.dumps(item.stance_quotes),
                    "blockquotes": json.dumps(item.blockquotes),
                }
            )


def load_desmog_from_csv(filename: str) -> List[DesmogItem]:
    items = []
    with open(filename, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Deserialize the JSON strings back into Python lists
            items.append(
                DesmogItem(
                    name=row["name"],
                    countries=json.loads(row["countries"]),
                    url=row["url"],
                    item_type=row["item_type"],
                    key_quotes=json.loads(row["key_quotes"]),
                    stance_quotes=json.loads(row["stance_quotes"]),
                    blockquotes=json.loads(row["blockquotes"]),
                )
            )
    return items


def create_quote_dataframe(items: List[DesmogItem]) -> pd.DataFrame:
    """Prepares a the format of dataset of quotes from DesmogItems."""
    rows = []

    # Iterate over the items
    for item in items:
        # Process key_quotes
        for quote in item.key_quotes:
            if len(quote) < 10:
                continue
            rows.append(
                {
                    "name": item.name,
                    "countries": ", ".join([country.strip() for country in item.countries]),
                    "url": item.url,
                    "item_type": item.item_type,
                    "quote": quote,
                    "section_source": "key_quote",
                }
            )
        # Process stance_quotes
        for quote in item.stance_quotes:
            if len(quote) < 10:
                continue
            rows.append(
                {
                    "name": item.name,
                    "countries": ", ".join([country.strip() for country in item.countries]),
                    "url": item.url,
                    "item_type": item.item_type,
                    "quote": quote,
                    "section_source": "stance_quote",
                }
            )
        # Process blockquotes
        for quote in item.blockquotes:
            if len(quote) < 10:
                continue
            rows.append(
                {
                    "name": item.name,
                    "countries": ", ".join([country.strip() for country in item.countries]),
                    "url": item.url,
                    "item_type": item.item_type,
                    "quote": quote,
                    "section_source": "block_quote",
                }
            )

    # Create a DataFrame from the rows
    return pd.DataFrame(rows)


def remove_blockquotes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove blockquotes if key_quotes or stance_quotes exist for the same name."""
    # Step 1: Remove blockquotes if key_quotes or stance_quotes exist
    # Identify names with key_quotes or stance_quotes
    valid_sources = df[df["section_source"].isin(["key_quote", "stance_quote"])]
    names_with_valid_sources = valid_sources["name"].unique()

    # Remove blockquotes for those names
    df = df[
        ~((df["name"].isin(names_with_valid_sources)) & (df["section_source"] == "block_quote"))
    ]
    return df


def add_data_quality_columns(df: pd.DataFrame) -> None:
    """Add data quality columns to sort on if necessary."""

    def words_count(quote: str) -> int:
        return len(quote.split())

    df["word_count"] = df["quote"].apply(words_count)

    # a full quote is a quote starting by an uppercase and finishing by punctuation
    # meaning the quote is most likely not out of context
    def is_full_quote(quote):
        if not isinstance(quote, str):
            return False
        return quote[0].isupper() and quote[-1] in string.punctuation

    df["is_full"] = df["quote"].apply(is_full_quote)


if __name__ == "__main__":
    desmog_items = get_desmog_items()

    for item in tqdm(desmog_items):
        try:
            content = get_article_content(item.url)
            item.blockquotes = content.get("blockquotes", [])
            item.key_quotes = content.get("key_quotes", [])
            item.stance_quotes = content.get("stance_quotes", [])
        except Exception:
            print(f"ERROR: parsing: {item.url} - skipped")

    file_path = "desmog_dataset.csv"
    save_desmog_to_csv(desmog_items, file_path)
