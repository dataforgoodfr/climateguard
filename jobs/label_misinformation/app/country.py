import logging
import os
from typing import Optional, Literal, Union, List
from dataclasses import dataclass, field

from secret_utils import get_secret_docker


@dataclass
class Country:
    code: str = field()  # ISO alpha-3 code for the country
    name: str = field()  # Name of the country in lowercase
    language: str = field()  # Language of the country
    bucket: str = field()  # Bucket for country data
    model: str = field()  # Model name for analyzing the country data
    prompt_version: str = field()  # Prompty version to be used in prod
    label_studio_id: int = field()  # ID for Cloud Storage ID (@see README)
    label_studio_project: int = (
        field()
    )  # ID for Label Studio project (visible on the web UI url)
    channels: List[str] = field()

    def verify_code(self, code: str):
        return code.lower() == self.code

    def verify_name(self, name: str):
        return name.lower() == self.name

    def verify_language(self, language: str):
        return language.lower() == self.language


FRANCE_COUNTRY = Country(
    code="fra",
    name="france",
    language="french",
    bucket=os.getenv("BUCKET_OUTPUT", "climateguard"),
    model=get_secret_docker("MODEL_NAME", "gpt-4o-mini"),
    prompt_version=get_secret_docker("PROMPT_VERSION", "0.0.1"),
    label_studio_id=os.getenv("LABEL_STUDIO_PROJECT_ID", 4),
    label_studio_project=os.getenv("LABEL_STUDIO_PROJECT", 4),
    channels=[
        "tf1",
        "france2",
        "fr3-idf",
        "m6",
        "arte",
        "bfmtv",
        "lci",
        "franceinfotv",
        "itele",
        "europe1",
        "france-culture",
        "france-inter",
        "sud-radio",
        "rmc",
        "rtl",
        "france24",
        "france-info",
        "rfi",
    ],
)
BELGIUM_COUNTRY = Country(
    code="bel",
    name="belgium",
    language="french",
    bucket=os.getenv("BUCKET_OUTPUT_BELGIUM", "climateguard-belgium"),
    model=get_secret_docker("MODEL_NAME", "gpt-4o-mini"),
    prompt_version=get_secret_docker("PROMPT_VERSION", "0.0.1"),
    label_studio_id=os.getenv("LABEL_STUDIO_PROJECT_ID_BELGIUM", 12),
    label_studio_project=os.getenv("LABEL_STUDIO_PROJECT_BELGIUM", 17),
    channels=["CANALZ", "LAUNE", "LN24", "RTL", "LATROIS"],
)
BRAZIL_COUNTRY = Country(
    code="bra",
    name="brazil",
    language="portuguese",
    bucket=os.getenv("BUCKET_OUTPUT_BRAZIL", "climateguard-brazil"),
    model=get_secret_docker("MODEL_NAME_BRAZIL", "gpt-4o-mini"),  # add as get env
    prompt_version=get_secret_docker("PROMPT_VERSION", "0.0.1"),
    label_studio_id=os.getenv("LABEL_STUDIO_PROJECT_ID_BRAZIL", 5),  # pass as getenv
    label_studio_project=os.getenv("LABEL_STUDIO_PROJECT_BRAZIL", 5),
    channels=[
        "tvglobo",
        "tvrecord",
        "sbt",
        "redebandeirantes",
        "jovempan",
        "cnnbrasil",
    ],
)

GERMANY_COUNTRY = Country(
    code="deu",
    name="germany",
    language="german",
    bucket=os.getenv("BUCKET_OUTPUT_GERMANY", "climateguard-germany"),
    model=get_secret_docker("MODEL_NAME_GERMANY", "gpt-4o-mini"),  # add as get env
    prompt_version=get_secret_docker("PROMPT_VERSION", "0.0.1"),
    label_studio_id=os.getenv("LABEL_STUDIO_PROJECT_ID_GERMANY", 14),  # pass as getenv
    label_studio_project=os.getenv("LABEL_STUDIO_PROJECT_GERMANY", 19),
    channels=[
        "daserste",
        "zdf",
        "rtl-television",
        "sat1",
        "prosieben",
        "kabel-eins",
    ],
)

SPAIN_COUNTRY = Country(
    code="esp",
    name="spain",
    language="spanish",
    bucket=os.getenv("BUCKET_OUTPUT", "safeguards-climate-spain-dev"),
    model=get_secret_docker("MODEL_NAME", "gpt-4o-mini"),  # add as get env
    prompt_version=get_secret_docker("PROMPT_VERSION", "0.0.1"),
    label_studio_id=os.getenv("LABEL_STUDIO_PROJECT_ID", 1),  # pass as getenv
    label_studio_project=os.getenv("LABEL_STUDIO_PROJECT", 1),
    channels=[
        "antenna-3",
        "rtve-24h",
        "rtve-la-1",
        "lasexta-news",
        "telecinco-news",
        "cuatro-news",
    ],
)

POLAND_COUNTRY = Country(
    code="pol",
    name="poland",
    language="polish",
    bucket=os.getenv("BUCKET_OUTPUT", "safeguards-climate-poland-dev"),
    model=get_secret_docker("MODEL_NAME", "gpt-4o-mini"), 
    prompt_version=get_secret_docker("PROMPT_VERSION", "0.0.1"),
    label_studio_id=os.getenv("LABEL_STUDIO_PROJECT_ID", 1),  
    label_studio_project=os.getenv("LABEL_STUDIO_PROJECT", 1),
    channels=[
        # Deploying only first three chains, rest will be deployed after the summer
        "polskie-radio",
        "radio-zet",
        "eska",
        "polsat",
        "tvn",
        "tvp",
        "tokfm",
    ],
)


def get_all_countries():
    return sorted(
        [
            FRANCE_COUNTRY,
            BELGIUM_COUNTRY,
            BRAZIL_COUNTRY,
            GERMANY_COUNTRY,
            SPAIN_COUNTRY,
            POLAND_COUNTRY,
        ],
        key=lambda x: x.name,
    )


@dataclass
class CountryCollection:
    code: str = field(default="None")
    name: str = field(default="all")
    language: str = field(default="all")
    countries: List[Country] = field(default_factory=get_all_countries)

    def __iter__(self):
        for country in self.countries:
            yield country

    def verify_code(self, code: str):
        return any(
            [code.lower() == country.code for country in self.countries]
            + [code.lower() == self.code]
        )

    def verify_name(self, name: str):
        return any(
            [name.lower() == country.name for country in self.countries]
            + [name.lower() == self.name]
        )

    def verify_language(self, language: str):
        return any(
            [language.lower() == country.language for country in self.countries]
            + [language.lower() == self.language]
        )


ALL_COUNTRIES = CountryCollection(name="all", countries=get_all_countries())
LEGACY_COUNTRIES = CountryCollection(
    name="legacy",
    code="None",
    language="french",
    countries=[BELGIUM_COUNTRY, FRANCE_COUNTRY],
)
PROD_COUNTRIES = CountryCollection(
    name="prod",
    code="None",
    language="all",
    countries=[BRAZIL_COUNTRY, FRANCE_COUNTRY, GERMANY_COUNTRY],
)


def get_country_or_collection_from_name(name: str):
    for entity in [
        *ALL_COUNTRIES.countries,
        LEGACY_COUNTRIES,
        PROD_COUNTRIES,
        ALL_COUNTRIES,
    ]:
        if entity.verify_name(name):
            return entity
    raise NotImplementedError(
        f"Country {name} not included in setup. Visit app/country.py for more"
    )


def get_countries(name: str):
    entity: Union[Country, CountryCollection] = get_country_or_collection_from_name(
        name
    )
    if isinstance(entity, Country):
        return CountryCollection(
            name=entity.name,
            code=entity.code,
            language=entity.language,
            countries=[entity],
        )
    return entity
