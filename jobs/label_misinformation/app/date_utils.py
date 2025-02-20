from datetime import datetime, timedelta
import logging

def set_date(date: str = "") -> datetime:
    if date == "":
        logging.info(f"Using yesterday's date")
        return (datetime.now() - timedelta(days=1) )
    else:
        logging.info(f"Using date {date}")
        clean_date = date.split(" ")[0]  # Keep only "YYYY-MM-DD"
        return datetime.strptime(clean_date, "%Y-%m-%d")