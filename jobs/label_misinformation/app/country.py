import logging
import os
from typing import Optional, Literal, Union, List
from dataclasses import dataclass, field

@dataclass
class Country:
    code: str = field() # ISO alpha-3 code for the country
    name: str = field() # Name of the country in lowercase
    language: str = field() # Language of the country

    def verify_code(self, code: str):
        return code.lower() == self.code

    def verify_name(self, name: str):
        return name.lower() == self.name

    def verify_language(self, language: str):
        return language.lower() == self.language

FRANCE_COUNTRY = Country(code="fra", name="france", language="french")
BELGIUM_COUNTRY = Country(code="bel", name="belgium", language="french")
BRAZIL_COUNTRY = Country(code="bra", name="brazil", language="portuguese")


def get_all_countries():
    return sorted([FRANCE_COUNTRY, BELGIUM_COUNTRY, BRAZIL_COUNTRY], key=lambda x: x.name)


@dataclass
class CountryCollection:
    code: str = field(default="None")
    name: str = field(default="all")
    language: str = field(default="all")
    countries: List[Country] = field(default_factory = get_all_countries)

    def __iter__(self):
            for country in self.countries:
                yield country

    def verify_code(self, code: str):
        return any([code.lower() == country.code for country in self.countries] + [code.lower() == self.code])

    def verify_name(self, name: str):
        return any([name.lower() == country.name for country in self.countries] + [name.lower() == self.name])

    def verify_language(self, language: str):
        return any([language.lower() == country.language for country in self.countries] + [language.lower() == self.language])


ALL_COUNTRIES = CountryCollection(name="all")
LEGACY_COUNTRIES = CountryCollection(name="legacy", code="None", language="french", countries=[BELGIUM_COUNTRY, FRANCE_COUNTRY])

def get_country_or_collection_from_name(name:str):
    for entity in [*ALL_COUNTRIES.countries, LEGACY_COUNTRIES, ALL_COUNTRIES]:
        if entity.verify_name(name):
            return entity
    raise NotImplementedError(f"Country {name} not included in setup. Visit app/country.py for more")
