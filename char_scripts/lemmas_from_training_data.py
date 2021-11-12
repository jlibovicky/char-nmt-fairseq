#!/usr/bin/env python3

"""Collects what lemmas and forms are in the training data."""

import argparse
from collections import defaultdict
import json
import logging
import sys

import spacy_udpipe

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class LemmaStat(object):
    def __init__(self, lemma: str) -> None:
        self.lemma: str = lemma
        self.count: int = 0
        self.forms = defaultdict(int)

    def add_form(self, form: str) -> None:
        self.forms[form] += 1
        self.count += 1

    def to_dict(self):
        return {
            "lemma": self.lemma,
            "count": self.count,
            "forms": self.forms}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("lng", type=str, help="Language code.")
    parser.add_argument(
        "input", type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    args = parser.parse_args()

    logging.info("Initialize UDPipe model.")
    spacy_udpipe.download(args.lng)
    nlp = spacy_udpipe.load(args.lng)

    lemma_stats = {}
    total_tokens = 0

    logging.info("Start iterating over the data.")
    for line_n, line in enumerate(args.input):
        doc = nlp(line)
        for token in doc:
            total_tokens += 1
            form = token.text
            lemma = token.lemma_

            if lemma not in lemma_stats:
                lemma_stats[lemma] = LemmaStat(lemma)
            lemma_stats[lemma].add_form(form.lower())

        if line_n % 1000 == 999:
            logging.info(
                "Processed %d sentences, %d tokens, %d unique lemmas.",
                line_n + 1, total_tokens, len(lemma_stats))

    logging.info("Finished.")

    print(json.dumps([s.to_dict() for s in lemma_stats.values()]))


if __name__ == "__main__":
    main()
