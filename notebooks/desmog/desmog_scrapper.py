import re
import string
from typing import List, Literal, Optional
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
        # Remove footnotes
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


def desmog_from_csv(filename: str) -> List[DesmogItem]:
    df = pd.read_csv(filename)

    df.fillna("[]", inplace=True)
    df.replace("", "[]")
    df["key_quotes"] = df["key_quotes"].apply(pd.eval)
    df["stance_quotes"] = df["stance_quotes"].apply(pd.eval)
    df["blockquotes"] = df["blockquotes"].apply(pd.eval)
    df["countries"] = df["countries"].apply(pd.eval)

    items = df.to_dict("records")
    data = [DesmogItem.model_validate(item) for item in items]
    return data


def desmog_to_csv(items: List[DesmogItem], filename: str):
    data = [item.model_dump() for item in items]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def create_quote_dataframe(items: List[DesmogItem]) -> pd.DataFrame:
    """Prepares a the format of dataset of quotes from DesmogItems."""
    data = [item.model_dump() for item in items]
    df = pd.DataFrame(data)

    df_long = df.melt(
        id_vars=["name", "countries", "item_type", "url"],
        value_vars=["stance_quotes", "key_quotes", "blockquotes"],
        var_name="section_source",
        value_name="quotes",
    )
    df_exploded = df_long.explode("quotes")
    df_exploded.rename(columns={"quotes": "quote"}, inplace=True)
    df_exploded.dropna(subset=["quote"], inplace=True)
    df_exploded = df_exploded[~(df_exploded["quote"] == "")]

    # Drop duplicates removing blockquotes in priority
    df_exploded.sort_values(by="section_source", inplace=True, ascending=False)
    df_exploded.drop_duplicates(subset="quote", keep="first", inplace=True)

    # Handles surrogates
    df_exploded = df_exploded.map(
        lambda x: str(x).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    )

    return df_exploded


def remove_blockquotes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove blockquotes if a given source already has key_quotes or stance_quotes."""
    valid_sources = df[df["section_source"].isin(["key_quote", "stance_quote"])]
    names_with_valid_sources = valid_sources["name"].unique()

    df = df[
        ~((df["name"].isin(names_with_valid_sources)) & (df["section_source"] == "block_quote"))
    ]
    return df


def add_data_quality_columns(df: pd.DataFrame) -> None:
    """
    Add data quality columns to filter on.
    - word count
    - boolean to indicate if the sentence is a full sentence
    """

    def words_count(quote: str) -> int:
        return len(quote.split())

    df["word_count"] = df["quote"].apply(words_count)

    # a full quote is a quote starting by an uppercase and finishing
    # by punctuation meaning the quote is most likely not out of context
    def is_full_quote(quote):
        if not isinstance(quote, str) or len(quote) < 5:
            return False
        return quote[0].isupper() and quote[-1] in string.punctuation

    df["is_full"] = df["quote"].apply(is_full_quote)


if __name__ == "__main__":
    desmog_items = get_desmog_items()

    n = 0
    for item in tqdm(desmog_items):
        try:
            content = get_article_content(item.url)
            item.blockquotes = content.get("blockquotes", [])
            item.key_quotes = content.get("key_quotes", [])
            item.stance_quotes = content.get("stance_quotes", [])
            n += 1
            if n == 5:
                break
        except Exception:
            print(f"ERROR: parsing: {item.url} - skipped")

    file_path = ".data/desmog_dataset.csv"
    desmog_to_csv(desmog_items, file_path)
