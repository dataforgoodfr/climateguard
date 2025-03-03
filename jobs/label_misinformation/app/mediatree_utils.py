import logging


def get_url_mediatree(date, channel) -> str:
    # https://keywords.mediatree.fr/player/?fifo=france-inter&start_cts=1729447079&end_cts=1729447201&position_cts=1729447080
    # todo must create url with https://keywords.mediatree.fr/player/?fifo=france-inter&start_cts=1729447079&end_cts=1729447201&position_cts=1729447080
    timestamp = str(int(date.timestamp()))
    timestamp_end = str(int(date.timestamp() + 1200))
    return f"https://keywords.mediatree.fr/player/?fifo={channel}&start_cts={timestamp}&end_cts={timestamp_end}&position_cts={timestamp}"
