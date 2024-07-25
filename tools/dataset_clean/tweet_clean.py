# !/usr/bin/env python
# -*- coding: utf-8 -*-


import re

import nltk
import numpy as np
import pandas as pd

nltk.download("punkt")
nltk.download("stopwords")
import string


class TweetDataset:
    def __init__(self):
        pass

    def _preprocess(self, text):
        text = self._remove_amp(text)
        text = self._remove_links(text)
        text = self._remove_hashes(text)
        text = self._remove_retweets(text)
        text = self._remove_mentions(text)
        text = self._remove_multiple_spaces(text)

        # text = self._lowercase(text)
        text = self._remove_punctuation(text)
        # text = self._remove_numbers(text)

        text_tokens = self._tokenize(text)
        text_tokens = self._stopword_filtering(text_tokens)
        # text_tokens = self._stemming(text_tokens)
        text = self._stitch_text_tokens_together(text_tokens)

        return text.strip()

    def _remove_amp(self, text):
        return text.replace("&amp;", " ")

    def _remove_mentions(self, text):
        return re.sub(r"(@.*?)[\s]", " ", text)

    def _remove_multiple_spaces(self, text):
        return re.sub(r"\s+", " ", text)

    def _remove_retweets(self, text):
        return re.sub(r"^RT[\s]+", " ", text)

    def _remove_links(self, text):
        return re.sub(r"https?:\/\/[^\s\n\r]+", " ", text)

    def _remove_hashes(self, text):
        return re.sub(r"#", " ", text)

    def _stitch_text_tokens_together(self, text_tokens):
        return " ".join(text_tokens)

    def _tokenize(self, text):
        return nltk.word_tokenize(text, language="english")

    def _stopword_filtering(self, text_tokens):
        stop_words = nltk.corpus.stopwords.words("english")

        return [token for token in text_tokens if token not in stop_words]

    def _stemming(self, text_tokens):
        porter = nltk.stem.porter.PorterStemmer()
        return [porter.stem(token) for token in text_tokens]

    def _remove_numbers(self, text):
        return re.sub(r"\d+", " ", text)

    def _lowercase(self, text):
        return text.lower()

    def _remove_punctuation(self, text):
        return "".join(
            character for character in text if character not in string.punctuation
        )
