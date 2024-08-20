import re
import unicodedata


def normalize_text(text):
    """
    Normalize text by removing accents, special characters, numbers, and extra spaces.
    :param text:
    :return:
    """
    text = text.lower()
    # text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
