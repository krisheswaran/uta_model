"""
Download play texts from their public domain sources.

Usage:
    conda run -n uta_model python scripts/download_plays.py cherry_orchard
    conda run -n uta_model python scripts/download_plays.py hamlet
    conda run -n uta_model python scripts/download_plays.py  # downloads all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from config import PLAYS, RAW_DIR

GUTENBERG_BASE = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
GUTENBERG_FALLBACK = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# DraCor TEI API — versioned endpoint with unversioned fallback
DRACOR_TEI_URLS = [
    "https://dracor.org/api/v1/corpora/{corpus}/plays/{play}/tei",
    "https://dracor.org/api/corpora/{corpus}/plays/{play}/tei",
]


def download_gutenberg(play_id: str, gutenberg_id: int) -> Path:
    out_path = RAW_DIR / f"{play_id}.txt"
    if out_path.exists():
        print(f"  [{play_id}] already downloaded, skipping")
        return out_path

    for url_template in (GUTENBERG_BASE, GUTENBERG_FALLBACK):
        url = url_template.format(id=gutenberg_id)
        print(f"  [{play_id}] fetching {url}")
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            out_path.write_bytes(response.content)
            print(f"  [{play_id}] saved to {out_path}")
            return out_path
        print(f"  [{play_id}] got {response.status_code}, trying fallback...")

    raise RuntimeError(f"Could not download Gutenberg text for {play_id} (id={gutenberg_id})")


def download_dracor_tei(play_id: str, corpus: str, play: str) -> Path:
    out_path = RAW_DIR / f"{play_id}.xml"
    if out_path.exists():
        print(f"  [{play_id}] already downloaded, skipping")
        return out_path

    for url_template in DRACOR_TEI_URLS:
        url = url_template.format(corpus=corpus, play=play)
        print(f"  [{play_id}] fetching {url}")
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            out_path.write_bytes(response.content)
            print(f"  [{play_id}] saved to {out_path}")
            return out_path
        print(f"  [{play_id}] got {response.status_code}, trying next URL...")

    raise RuntimeError(
        f"Could not download DraCor TEI for {play_id} "
        f"(corpus={corpus!r}, play={play!r}). "
        "Check https://dracor.org for the correct corpus/play slug."
    )


def download_play(play_id: str) -> Path:
    config = PLAYS.get(play_id)
    if config is None:
        raise ValueError(f"Unknown play_id: {play_id!r}. Available: {list(PLAYS)}")

    if config["source"] == "gutenberg":
        return download_gutenberg(play_id, config["gutenberg_id"])
    elif config["source"] == "dracor_tei":
        return download_dracor_tei(play_id, config["dracor_corpus"], config["dracor_play"])
    else:
        raise ValueError(f"Unknown source: {config['source']!r}")


def main():
    parser = argparse.ArgumentParser(description="Download play texts")
    parser.add_argument("play_ids", nargs="*", help="Play IDs to download (default: all)")
    args = parser.parse_args()

    targets = args.play_ids or list(PLAYS.keys())
    print(f"Downloading: {targets}")
    for play_id in targets:
        try:
            path = download_play(play_id)
            print(f"  OK: {path}")
        except Exception as exc:
            print(f"  ERROR [{play_id}]: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
