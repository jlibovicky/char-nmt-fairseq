#!/usr/bin/env python3

"""Evaluate precision of new forms."""

import argparse
import json
import pickle
import logging
import sys

from lemmas_from_training_data import LemmaStat

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "train_data_stats", type=argparse.FileType("rb"),
        help="Pickle file with lemmas and forms from train data.")
    args = parser.parse_args()

    logging.info("Loading lemma stats from training data.")

    lemma_stats = pickle.load(args.train_data_stats)
    logging.info("Loaded.")
    print("[")
    logging.info("Iterate over hypotheses and references.")

    for stat in lemma_stats.values():
        if stat.count < 5:
            continue
        print(json.dumps(stat.to_dict()) + ",")

    print("]")

if __name__ == "__main__":
    main()
