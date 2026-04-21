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
    channels_no_whisper: List[str] = field(default_factory=lambda: [])

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
    bucket=os.getenv("BUCKET_OUTPUT", "safeguards-climate-belgium-dev"),
    model=get_secret_docker("MODEL_NAME", "gpt-4o-mini"),
    prompt_version=get_secret_docker("PROMPT_VERSION", "0.0.1"),
    label_studio_id=os.getenv("LABEL_STUDIO_PROJECT_ID", 1),
    label_studio_project=os.getenv("LABEL_STUDIO_PROJECT", 1),
    channels=[
        "CANALZ",
        "RTL",
        "LAUNE",
        "LN24",
        "LATROIS",
        "ACTV",
        "CANALC",
        "BX1",
        "CANALZOOM",
        "MATELE",
        "NOTELE",
        "RTC",
        "TELEMB",
        "TELESAMBRE",
        "TVCOM",
        "TVLUX",
        "VEDIA",
        "la-premiere",
        "bel-rtl",
        "vivacite",
        "ln-radio",
    ],
    channels_no_whisper=[
        "CANALZ",
        "RTL",
        "LAUNE",
        "LN24",
        "LATROIS",
        "ACTV",
        "CANALC",
        "BX1",
        "CANALZOOM",
        "MATELE",
        "NOTELE",
        "RTC",
        "TELEMB",
        "TELESAMBRE",
        "TVCOM",
        "TVLUX",
        "VEDIA",
    ],
)
BELGIUM_FLANDERS_COUNTRY = Country(
    code="bel-fla",
    name="belgium-flanders",
    language="dutch",
    bucket=os.getenv("BUCKET_OUTPUT", "safeguards-climate-belgium-flanders-dev"),
    model=get_secret_docker("MODEL_NAME", "gpt-4o-mini"),
    prompt_version=get_secret_docker("PROMPT_VERSION", "0.0.1"),
    label_studio_id=os.getenv("LABEL_STUDIO_PROJECT_ID", 1),
    label_studio_project=os.getenv("LABEL_STUDIO_PROJECT", 1),
    channels=[
        "canvas",
        "radio-2",
        "vtm",
        "radio-1",
        "stu-bru",
        "qmusic",
        "play",
        "vrt1",
    ],
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
        "tvbrasil",
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
    bucket=os.getenv("BUCKET_OUTPUT", "safeguards-climate-germany-dev"),
    model=get_secret_docker("MODEL_NAME", "gpt-4o-mini"),  # add as get env
    prompt_version=get_secret_docker("PROMPT_VERSION", "0.0.1"),
    label_studio_id=os.getenv("LABEL_STUDIO_PROJECT_ID", 1),  # pass as getenv
    label_studio_project=os.getenv("LABEL_STUDIO_PROJECT", 1),
    channels=[
        "daserste",
        "zdf",
        "rtl-television",
        "sat1",
        "prosieben",
        "kabel-eins",
    ],
    channels_no_whisper=["daserste", "zdf"],
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
        "tvp",
        "polsat",
        "tvn",
        "polskie-radio",
        "tokfm",
        "radio-zet",
        "eska",
        "tv-republika",
        "tv-trwam",
        "radio-maryja",
        "tvs",
        "wpolsce24",
        "tv-puls",
        "fokus-tv",
    ],
)


def get_all_countries():
    return sorted(
        [
            FRANCE_COUNTRY,
            BELGIUM_COUNTRY,
            BELGIUM_FLANDERS_COUNTRY,
            BRAZIL_COUNTRY,
            GERMANY_COUNTRY,
            SPAIN_COUNTRY,
            POLAND_COUNTRY,
        ],
        key=lambda x: x.name,
    )


@dataclass
class CountryCollection:
    code: str = field(default="all")
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


ALL_COUNTRIES = CountryCollection(name="all", code="all", countries=get_all_countries())
LEGACY_COUNTRIES = CountryCollection(
    name="legacy",
    code="legacy",
    language="french",
    countries=[BELGIUM_COUNTRY, FRANCE_COUNTRY],
)
PROD_COUNTRIES = CountryCollection(
    name="prod",
    code="prod",
    language="all",
    countries=[BRAZIL_COUNTRY, FRANCE_COUNTRY],
)


def convert_to_base_country_name(name: str):
    return name.split("-")[0]


def get_country_or_collection_from_code(code: str):
    for entity in [
        *ALL_COUNTRIES.countries,
        LEGACY_COUNTRIES,
        PROD_COUNTRIES,
        ALL_COUNTRIES,
    ]:
        if entity.verify_code(code):
            return entity
    raise NotImplementedError(
        f"Country with code '{code}' not included in setup. Visit app/country.py for more"
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
