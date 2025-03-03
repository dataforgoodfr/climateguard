import pytest
from app.mediatree_utils import *
from datetime import datetime
import logging

def test_get_url_mediatree():
    date_string = "2024-12-12 10:10:10"
    date = datetime.fromisoformat(date_string)
    output = get_url_mediatree(channel="itele", date=date)
    assert output == "https://keywords.mediatree.fr/player/?fifo=itele&start_cts=1733998210&end_cts=1733999410&position_cts=1733998210"