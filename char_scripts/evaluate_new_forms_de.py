#!/usr/bin/env python3

"""Evaluate precision of new forms."""

import argparse
import pickle
import logging
import sys

import spacy_udpipe

from lemmas_from_training_data import LemmaStat

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("lng", type=str)
    parser.add_argument(
        "train_data_stats", type=argparse.FileType("rb"),
        help="JSON with lemmas and forms from train data.")
    parser.add_argument(
        "reference", type=argparse.FileType("r"))
    parser.add_argument(
        "hypothesis", type=argparse.FileType("r"), nargs="?",
        default=sys.stdin)
    args = parser.parse_args()

    logging.info("Load UDPipe model.")
    nlp = spacy_udpipe.load(args.lng)

    logging.info("Loading lemma stats from training data.")
    lemma_stats = pickle.load(args.train_data_stats)
    known_lemmas = set(lemma_stats.keys())
    known_forms = set(next(iter(lemma_stats.values())).forms.keys())
    logging.info("Iterate over hypotheses and references.")

    unseen_lemma_count = 0
    unseen_lemma_hits = 0

    seen_lemma_count = 0
    seen_lemma_hits = 0

    unseen_form_count = 0
    unseen_form_hits = 0

    seen_form_count = 0
    seen_form_hits = 0

    for ref, hyp in zip(args.reference, args.hypothesis):
        hyp_doc = nlp(hyp.strip())
        hyp_lemmas = {tok.lemma_ for tok in hyp_doc}
        hyp_lemmatized = {(tok.lemma_, tok.text.lower()) for tok in hyp_doc}

        for token in nlp(ref.strip()):
            form = token.text.lower()
            lemma = token.lemma_
            if lemma not in known_lemmas:
                unseen_lemma_count += 1
                if lemma in hyp_lemmas:
                    unseen_lemma_hits += 1
            else:
                seen_lemma_count += 1
                if lemma in hyp_lemmas:
                    seen_lemma_hits += 1

                if form not in known_forms:
                    unseen_form_count += 1
                    if (lemma, form.lower()) in hyp_lemmatized:
                        unseen_form_hits += 1
                else:
                    seen_form_count += 1
                    if (lemma, form.lower()) in hyp_lemmatized:
                        seen_form_hits += 1

    logging.info("Test data contain %d unseen lemmas.", unseen_lemma_count)
    unseen_lemma_precision = unseen_lemma_hits / unseen_lemma_count
    logging.info("Unseen lemma recall %.3f.", 100 * unseen_lemma_precision)
    logging.info("Seen lemma recall   %.3f.", 100 * seen_lemma_hits / seen_lemma_count)

    logging.info(
        "Test data contain %d unseen forms of seen lemmas.",
        unseen_form_count)
    unseen_form_precision = unseen_form_hits / unseen_form_count
    logging.info("Unseen form recall %.3f.", 100 * unseen_form_precision)
    logging.info("Seen form recall %.3f.", 100 * seen_form_hits / seen_form_count)
    logging.info("Done.")


if __name__ == "__main__":
    main()
