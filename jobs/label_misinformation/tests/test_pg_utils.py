import pytest
import sys, os
sys.path.append(os.path.abspath('/app'))
from app.pg_utils import *
from datetime import datetime
import logging

def test_get_keywords_for_a_day():
    # save to pg some keywords
    # TODO

    assert True == False