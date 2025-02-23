from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List


class TaxonomyEnum(Enum):
    CARDS = "cards"
    DISINFORMATION_GRADE = "disinformation_grade"


class LabelEnum(Enum):
    pass


class CardsEnum(LabelEnum):
    ONE = "one"
    TWO = "two"


class DisinformationGradeEnum():
    ONE = "one"
    TWO = "two"


class MediaNameEnum(Enum):
    TF1 = "tf1"
    FRANCE2 = "france2"
    FRANCE3 = "fr3-idf"
    M6 = "m6"
    ARTE = "arte"
    D8 = "d8"
    BFMTV = "bfmtv"
    LCI = "lci"
    FRANCE_INFO_TV = "franceinfotv"
    ITELE = "itele"
    EUROPE1 = "europe1"
    FRANCE_CULTURE = "france-culture"
    FRANCE_INTER = "france-inter"
    SUD_RADIO = "sud-radio"
    RMC = "rmc"
    RTL = "rtl"
    FRANCE24 = "france24"
    FRANCE_INFO = "france-info"
    RFI = "rfi"


@dataclass
class Document:
    document_id: str
    transcript: str
    source: MediaNameEnum
    diffusion_datetime: datetime


@dataclass
class ClassifiedDocument(Document):
    disinformation_document: bool
    disinformation_label: LabelEnum


@dataclass
class AnnotationCampaign:
    date: datetime
    documents: List[Document]
    taxonomy_type: TaxonomyEnum


class Documents(ABC):
    @abstractmethod
    def get_documents_from_date(self, date: datetime) -> List[Document]:
        raise NotImplementedError

    @abstractmethod
    def save_documents(self, documents: List[Document]) -> None:
        raise NotImplementedError
