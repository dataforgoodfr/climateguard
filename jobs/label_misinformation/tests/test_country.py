import os
import sys

import pytest

sys.path.append(os.path.abspath("/app"))

from app.country import (
    ALL_COUNTRIES,
    BELGIUM_COUNTRY,
    BRAZIL_COUNTRY,
    FRANCE_COUNTRY,
    GERMANY_COUNTRY,
    POLAND_COUNTRY,
    SPAIN_COUNTRY,
    LEGACY_COUNTRIES,
    PROD_COUNTRIES,
    Country,
    CountryCollection,
    get_country_or_collection_from_name,
)


def test_empty_country():
    with pytest.raises(TypeError):
        Country()


def test_empty_country_collection():
    assert CountryCollection() == ALL_COUNTRIES


def test_country_collection_presets():
    assert ALL_COUNTRIES == CountryCollection(
        name="all", countries=[BELGIUM_COUNTRY, BRAZIL_COUNTRY, FRANCE_COUNTRY, GERMANY_COUNTRY, POLAND_COUNTRY, SPAIN_COUNTRY]
    )
    assert LEGACY_COUNTRIES == CountryCollection(
        name="legacy",
        code="None",
        language="french",
        countries=[BELGIUM_COUNTRY, FRANCE_COUNTRY],
    )
    assert PROD_COUNTRIES == CountryCollection(
        name="prod",
        code="None",
        language="all",
        countries=[BRAZIL_COUNTRY, FRANCE_COUNTRY, GERMANY_COUNTRY],
    )


def test_get_country_or_collection_from_name():
    for entity in (
        FRANCE_COUNTRY,
        BELGIUM_COUNTRY,
        BRAZIL_COUNTRY,
        GERMANY_COUNTRY,
        POLAND_COUNTRY,
        SPAIN_COUNTRY,
        PROD_COUNTRIES,
        LEGACY_COUNTRIES,
        ALL_COUNTRIES,
    ):
        assert get_country_or_collection_from_name(entity.name) == entity
