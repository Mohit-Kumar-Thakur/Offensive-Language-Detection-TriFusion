import re

URL_PATTERN = re.compile(r"http\S+")
USER_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")

def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = URL_PATTERN.sub(" URL ", text)
    text = USER_PATTERN.sub(" USER ", text)
    text = HASHTAG_PATTERN.sub(" HASHTAG ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
