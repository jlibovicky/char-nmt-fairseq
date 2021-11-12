#!/usr/bin/env python3

import argparse
import random

import numpy as np
import scipy.stats
import sacrebleu
from tqdm import trange
from comet.models import download_model


def load_file(fh):
    sentences = []
    for line in fh:
        sentences.append(line.strip())
    fh.close()
    return sentences


def confidence_interval(data, confidence=0.95):
    m, se = np.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return m, h

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("src", type=argparse.FileType("r"))
    parser.add_argument("ref", type=argparse.FileType("r"))
    parser.add_argument("hyp", type=argparse.FileType("r"))
    parser.add_argument("--use-comet", default=False, action="store_true")
    parser.add_argument("--use-bertscore", default=False, action="store_true")
    parser.add_argument("--n-samples", default=1000, type=int)
    parser.add_argument("--confidence", default=0.95, type=int)
    args = parser.parse_args()

    srcs = load_file(args.src)
    refs = load_file(args.ref)
    hyps = load_file(args.hyp)

    assert len(srcs) == len(refs) == len(hyps)

    bleu_score = sacrebleu.BLEU().corpus_score(hyps, [refs], n_bootstrap=args.n_samples)
    print(f"BLEU   {bleu_score.score:.4f}  {bleu_score._ci:.4f}")
    chrf_score = sacrebleu.CHRF().corpus_score(hyps, [refs], n_bootstrap=args.n_samples)
    print(f"chrF   {chrf_score.score / 100:.6f}  {chrf_score._ci / 100:.6f}")

    comet = download_model("wmt-large-da-estimator-1719")
    comet_data = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, ref, hyp in zip(srcs, refs, hyps)]
    comet_res = comet.predict(comet_data, cuda=True, show_progress=True)[1]

    comet_mean, comet_int = confidence_interval(comet_res)
    print(f"COMET  {comet_mean:.6f}  {comet_int:.6f}")


if __name__ == "__main__":
    main()
