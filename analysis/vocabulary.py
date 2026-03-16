"""
Canonical Tactic Vocabulary — Phase B

Clusters free-text tactic strings into canonical categories using
sentence-transformer embeddings and agglomerative clustering.

Usage:
    # Build vocabulary from existing parsed plays
    conda run -n uta_model python analysis/vocabulary.py build

    # Show clusters for review
    conda run -n uta_model python analysis/vocabulary.py show

    # Assign canonical tactics to all BeatStates in parsed plays
    conda run -n uta_model python analysis/vocabulary.py normalize

    # Add new raw tactics from a freshly analyzed play
    conda run -n uta_model python analysis/vocabulary.py ingest cherry_orchard
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PARSED_DIR, DATA_DIR
from schemas import CanonicalTactic, Play, TacticVocabulary

VOCAB_DIR = DATA_DIR / "vocab"
VOCAB_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_PATH = VOCAB_DIR / "tactic_vocabulary.json"

# Cosine distance threshold for agglomerative clustering.
# Lower = more clusters (finer-grained); higher = fewer clusters (coarser).
# 0.35 typically yields ~30-40 clusters from 297 tactics.
DEFAULT_DISTANCE_THRESHOLD = 0.35


# ────────────────────────────────────────────────────────────────────────────
# Step 1: Collect raw tactics from parsed plays
# ────────────────────────────────────────────────────────────────────────────

def collect_raw_tactics(play_ids: list[str] | None = None) -> Counter:
    """Collect all raw tactic strings and their counts from parsed plays.

    Uses raw JSON parsing to avoid Pydantic validation errors from
    any malformed BeatState entries in the data.
    """
    if play_ids is None:
        play_ids = [p.stem for p in PARSED_DIR.glob("*.json")]

    tactics: Counter = Counter()
    for play_id in play_ids:
        path = PARSED_DIR / f"{play_id}.json"
        if not path.exists():
            print(f"  [!] Skipping {play_id}: {path} not found")
            continue
        data = json.load(open(path))
        for act in data.get("acts", []):
            for scene in act.get("scenes", []):
                for beat in scene.get("beats", []):
                    for bs in beat.get("beat_states", []):
                        t = bs.get("tactic_state", "").strip().lower()
                        if t:
                            tactics[t] += 1
    return tactics


# ────────────────────────────────────────────────────────────────────────────
# Step 2: Expand bare words to tactic sentences for better embeddings
# ────────────────────────────────────────────────────────────────────────────

_TACTIC_DEFINITIONS: dict[str, str] = {
    # Top 30 tactics get hand-crafted definitions for embedding quality.
    # Others fall through to the generic template.
    "deflect": "deflect — redirect attention away from a threatening topic to avoid confrontation",
    "dismiss": "dismiss — reject the other person's position as unworthy of engagement",
    "reassure": "reassure — calm the other person's fears by minimizing danger or offering comfort",
    "mock": "mock — use ridicule or imitation to diminish the other person's standing",
    "command": "command — issue a direct order expecting compliance",
    "appease": "appease — make peace by yielding ground or offering concessions",
    "plead": "plead — appeal to the other person's compassion from a position of vulnerability",
    "probe": "probe — ask pointed questions to uncover hidden information or test boundaries",
    "expose": "expose — reveal something the other person is trying to hide",
    "provoke": "provoke — deliberately agitate the other person to force a reaction",
    "shame": "shame — make the other person feel guilt or disgrace about their behavior",
    "challenge": "challenge — directly contest the other person's authority or claims",
    "test": "test — present a trial to see how the other person responds under pressure",
    "flatter": "flatter — offer praise, sincere or strategic, to win favor or disarm",
    "confess": "confess — reveal a personal truth or secret to change the dynamic",
    "interrogate": "interrogate — press the other person with demanding questions",
    "warn": "warn — signal impending danger or consequences to create urgency",
    "summon": "summon — call forth someone's attention, courage, or presence",
    "disarm": "disarm — neutralize the other person's resistance through charm or surprise",
    "dominate": "dominate — assert power and control over the other person",
    "instruct": "instruct — teach or explain something from a position of knowledge",
    "dazzle": "dazzle — overwhelm with brilliance, charm, or spectacle",
    "redirect": "redirect — steer the conversation or focus to a different subject",
    "comply": "comply — yield to the other person's wishes or demands",
    "embrace": "embrace — accept or welcome the other person warmly",
    "inform": "inform — deliver factual news or knowledge to change understanding",
    "seduce": "seduce — draw the other person in through allure and desire",
    "implore": "implore — beg urgently with emotional intensity",
    "surrender": "surrender — give up resistance and submit completely",
    "withdraw": "withdraw — pull away emotionally or physically from engagement",
    "threaten": "threaten — promise harm or negative consequences to coerce",
    "soothe": "soothe — gently calm distress through tenderness",
    "confront": "confront — force the other person to face an uncomfortable truth",
    "confide": "confide — share something private to build intimacy or trust",
    "restrain": "restrain — hold back the other person or oneself from action",
    "retreat": "retreat — fall back from a position under pressure",
    "concede": "concede — admit the other person is right or yield a point",
    "coax": "coax — gently persuade through patience and warmth",
    "scold": "scold — reprimand the other person for wrongdoing",
    "stall": "stall — delay or buy time to avoid a decision or confrontation",
    "remind": "remind — bring back to awareness something forgotten or suppressed",
    "reproach": "reproach — express disapproval tinged with sorrow or disappointment",
    "lament": "lament — express grief or sorrow to evoke sympathy",
    "nurture": "nurture — care for and support the other person protectively",
    "persist": "persist — continue pressing despite resistance",
    "silence": "silence — force the other person to stop speaking",
    "rally": "rally — gather courage or support for collective action",
    "deny": "deny — refuse to acknowledge a truth or accusation",
    "correct": "correct — point out an error to establish factual authority",
    "affirm": "affirm — validate the other person's feelings or position",
    "remind": "remind — bring something forgotten back to awareness",
    "dispatch": "dispatch — send the other person away on an errand or mission",
    "recruit": "recruit — enlist the other person into your cause or plan",
    "absorb": "absorb — take in what the other person is saying without pushing back",
    "report": "report — deliver factual observations or news",
    "clarify": "clarify — make your position or the situation clearer",
    "shepherd": "shepherd — gently guide and protect the other person",
    "rally": "rally — build morale and collective resolve",
    "submit": "submit — yield completely to the other person's authority",
    "grandstand": "grandstand — perform for an audience to boost your own image",
    "persist": "persist — keep pressing forward despite resistance or rejection",
    "reminisce": "reminisce — recall shared past to reconnect or manipulate",
    "consecrate": "consecrate — elevate the moment to sacred or ceremonial status",
    "pledge": "pledge — make a solemn commitment or promise",
    "overwhelm": "overwhelm — overpower the other person with force of emotion or argument",
    "eulogize": "eulogize — praise someone at length, often to honor or memorialize",
    "rationalize": "rationalize — construct logical justification for emotional behavior",
    "endure": "endure — bear suffering without yielding or complaining",
    "demand": "demand — insist forcefully that the other person comply",
    "alert": "alert — urgently draw attention to an immediate threat or crisis",
    "threaten": "threaten — promise harm or punishment to coerce compliance",
    "accuse": "accuse — charge the other person with wrongdoing",
    "deny": "deny — refuse to admit or acknowledge what is being said",
    "stall": "stall — delay action or decision to buy time",
    "retreat": "retreat — pull back from engagement to protect oneself",
    "interrogate": "interrogate — aggressively question to extract information",
    "disarm": "disarm — neutralize hostility through unexpected warmth or humor",
}


def expand_tactic_to_sentence(verb: str) -> str:
    """Expand a bare tactic verb into a sentence anchoring it as a theatrical action.

    Uses hand-crafted definitions for top tactics to ensure high-quality
    embeddings. Falls through to a generic template for the long tail.
    """
    key = verb.lower().strip()
    if key in _TACTIC_DEFINITIONS:
        return _TACTIC_DEFINITIONS[key]
    # For the long tail, use a light template: enough context to signal
    # "theatrical action verb" without so much boilerplate that the template
    # dominates the embedding. The verb itself carries most of the signal.
    return f"to {verb} someone"


# ────────────────────────────────────────────────────────────────────────────
# Step 3: Embed and cluster
# ────────────────────────────────────────────────────────────────────────────

def _load_embedding_model():
    """Load the sentence-transformers model (cached after first call)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_tactics(tactics: list[str], model=None) -> np.ndarray:
    """Embed a list of tactic sentences. Returns (N, D) array."""
    if model is None:
        model = _load_embedding_model()
    sentences = [expand_tactic_to_sentence(t) for t in tactics]
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings)


def cluster_tactics(
    tactics: list[str],
    counts: list[int],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    min_count_for_seed: int = 3,
) -> TacticVocabulary:
    """Cluster raw tactic strings into canonical categories.

    Uses a two-pass approach:
    1. Cluster only the "seed" tactics (those with hand-crafted definitions or
       frequency >= min_count_for_seed) to form stable canonical clusters.
    2. Assign all remaining long-tail tactics to the nearest seed cluster,
       or flag as unmapped if too distant.

    This prevents the long tail from forming a single mega-cluster.

    Args:
        tactics: list of unique raw tactic strings (lowercased)
        counts: parallel list of occurrence counts
        distance_threshold: cosine distance threshold for clustering seeds
        min_count_for_seed: minimum occurrence count to be a seed tactic

    Returns:
        TacticVocabulary with clustered CanonicalTactics
    """
    from sklearn.cluster import AgglomerativeClustering

    model = _load_embedding_model()

    # Split into seed and long-tail
    seed_tactics, seed_counts = [], []
    tail_tactics, tail_counts = [], []
    for t, c in zip(tactics, counts):
        if t in _TACTIC_DEFINITIONS or c >= min_count_for_seed:
            seed_tactics.append(t)
            seed_counts.append(c)
        else:
            tail_tactics.append(t)
            tail_counts.append(c)

    # Pass 1: cluster the seeds
    seed_embeddings = embed_tactics(seed_tactics, model)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(seed_embeddings)

    # Group seed tactics by cluster label
    clusters: dict[int, list[tuple[str, int]]] = {}
    for tactic, count, label in zip(seed_tactics, seed_counts, labels):
        clusters.setdefault(label, []).append((tactic, count))

    # Build canonical tactics from seeds
    canonical_tactics = []
    cluster_embeddings = []  # centroid for each canonical tactic
    for label in sorted(clusters.keys()):
        members = clusters[label]
        members.sort(key=lambda x: -x[1])  # sort by frequency descending
        canonical_verb = members[0][0]
        canonical_id = canonical_verb.upper()

        ct = CanonicalTactic(
            canonical_id=canonical_id,
            canonical_verb=canonical_verb,
            description=expand_tactic_to_sentence(canonical_verb),
            members=[m[0] for m in members],
            category="",
        )
        canonical_tactics.append(ct)

        # Compute cluster centroid from seed member embeddings
        member_indices = [seed_tactics.index(m[0]) for m in members]
        centroid = seed_embeddings[member_indices].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # re-normalize
        cluster_embeddings.append(centroid)

    cluster_embeddings = np.array(cluster_embeddings)

    # Pass 2: assign long-tail tactics to nearest seed cluster
    # Use a TIGHTER threshold than seed clustering: the seed clusters
    # were built from well-defined tactics; long-tail items should only
    # join if they're genuinely close. This prevents generic "to X someone"
    # embeddings from flooding a single cluster.
    assign_threshold = distance_threshold * 0.7  # 30% tighter
    unmapped = []
    if tail_tactics:
        tail_embeddings = embed_tactics(tail_tactics, model)
        similarities = tail_embeddings @ cluster_embeddings.T

        for i, tactic in enumerate(tail_tactics):
            best_idx = int(np.argmax(similarities[i]))
            best_dist = 1.0 - float(similarities[i, best_idx])

            if best_dist <= assign_threshold:
                canonical_tactics[best_idx].members.append(tactic)
            else:
                unmapped.append(tactic)

    # Sort by total count descending
    count_map = dict(zip(tactics, counts))
    canonical_tactics.sort(
        key=lambda ct: -sum(count_map.get(m, 0) for m in ct.members)
    )

    return TacticVocabulary(
        version=1,
        tactics=canonical_tactics,
        unmapped=unmapped,
    )


# ────────────────────────────────────────────────────────────────────────────
# Step 4: Incremental assignment for new tactics
# ────────────────────────────────────────────────────────────────────────────

def assign_new_tactics(
    vocab: TacticVocabulary,
    new_tactics: list[str],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> TacticVocabulary:
    """Assign new tactic strings to existing clusters or flag as unmapped.

    Args:
        vocab: existing TacticVocabulary
        new_tactics: list of new raw tactic strings to assign
        distance_threshold: max cosine distance to assign to existing cluster

    Returns:
        Updated TacticVocabulary (modified in place and returned)
    """
    if not new_tactics:
        return vocab

    # Filter out tactics already in the vocabulary
    known = set()
    for ct in vocab.tactics:
        known.update(m.lower() for m in ct.members)
    truly_new = [t.lower().strip() for t in new_tactics if t.lower().strip() not in known]
    truly_new = list(set(truly_new))  # deduplicate

    if not truly_new:
        return vocab

    model = _load_embedding_model()

    # Embed existing canonical verbs and new tactics
    canonical_verbs = [ct.canonical_verb for ct in vocab.tactics]
    canonical_embeddings = embed_tactics(canonical_verbs, model)
    new_embeddings = embed_tactics(truly_new, model)

    # Compute cosine similarities (embeddings are already normalized)
    similarities = new_embeddings @ canonical_embeddings.T  # (N_new, N_canonical)

    for i, tactic in enumerate(truly_new):
        best_idx = int(np.argmax(similarities[i]))
        best_sim = float(similarities[i, best_idx])
        best_dist = 1.0 - best_sim

        if best_dist <= distance_threshold:
            # Assign to existing cluster
            vocab.tactics[best_idx].members.append(tactic)
        else:
            # Flag as unmapped
            if tactic not in vocab.unmapped:
                vocab.unmapped.append(tactic)

    return vocab


# ────────────────────────────────────────────────────────────────────────────
# Step 5: Normalize — assign canonical_tactic to BeatStates in parsed plays
# ────────────────────────────────────────────────────────────────────────────

def normalize_play(play: Play, vocab: TacticVocabulary) -> int:
    """Assign canonical_tactic to all BeatStates in a Play. Returns count of assignments."""
    assigned = 0
    for act in play.acts:
        for scene in act.scenes:
            for beat in scene.beats:
                for bs in beat.beat_states:
                    raw = bs.tactic_state.strip().lower()
                    if raw:
                        canonical = vocab.lookup(raw)
                        bs.canonical_tactic = canonical
                        if canonical:
                            assigned += 1
    return assigned


# ────────────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────────────

def save_vocabulary(vocab: TacticVocabulary, path: Path = VOCAB_PATH) -> None:
    path.write_text(vocab.model_dump_json(indent=2))
    print(f"Saved vocabulary (v{vocab.version}, {len(vocab.tactics)} canonical, "
          f"{len(vocab.unmapped)} unmapped) to {path}")


def load_vocabulary(path: Path = VOCAB_PATH) -> TacticVocabulary:
    if not path.exists():
        raise FileNotFoundError(f"No vocabulary at {path}. Run 'vocabulary.py build' first.")
    return TacticVocabulary.model_validate_json(path.read_text())


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def _cmd_build(args):
    """Build the canonical tactic vocabulary from all parsed plays."""
    print("Collecting raw tactics from parsed plays...")
    raw_counts = collect_raw_tactics()
    tactics = list(raw_counts.keys())
    counts = [raw_counts[t] for t in tactics]
    print(f"  Found {len(tactics)} unique tactics ({sum(counts)} instances)")

    threshold = args.threshold if hasattr(args, "threshold") else DEFAULT_DISTANCE_THRESHOLD
    print(f"\nClustering with distance threshold={threshold}...")
    vocab = cluster_tactics(tactics, counts, distance_threshold=threshold)
    print(f"  → {len(vocab.tactics)} canonical tactics")

    save_vocabulary(vocab)


def _cmd_show(args):
    """Display the vocabulary clusters for human review."""
    vocab = load_vocabulary()
    print(f"Tactic Vocabulary v{vocab.version}: {len(vocab.tactics)} canonical tactics\n")

    # Collect raw counts for display
    raw_counts = collect_raw_tactics()

    for i, ct in enumerate(vocab.tactics, 1):
        total = sum(raw_counts.get(m, 0) for m in ct.members)
        members_str = ", ".join(
            f"{m}({raw_counts.get(m, 0)})" for m in sorted(ct.members, key=lambda x: -raw_counts.get(x, 0))
        )
        print(f"{i:3d}. {ct.canonical_id:<20s} [{total:4d}] {members_str}")

    if vocab.unmapped:
        print(f"\nUnmapped ({len(vocab.unmapped)}): {', '.join(vocab.unmapped)}")


def normalize_play_json(data: dict, vocab: TacticVocabulary) -> int:
    """Assign canonical_tactic to all BeatStates in a raw JSON play dict.

    Uses raw JSON to avoid Pydantic validation errors from malformed entries.
    Returns count of assignments.
    """
    assigned = 0
    for act in data.get("acts", []):
        for scene in act.get("scenes", []):
            for beat in scene.get("beats", []):
                for bs in beat.get("beat_states", []):
                    raw = bs.get("tactic_state", "").strip().lower()
                    if raw:
                        canonical = vocab.lookup(raw)
                        bs["canonical_tactic"] = canonical
                        if canonical:
                            assigned += 1
    return assigned


def _cmd_normalize(args):
    """Assign canonical_tactic to all BeatStates in parsed plays."""
    vocab = load_vocabulary()
    play_ids = [p.stem for p in PARSED_DIR.glob("*.json")]

    for play_id in play_ids:
        path = PARSED_DIR / f"{play_id}.json"
        data = json.load(open(path))
        assigned = normalize_play_json(data, vocab)
        total = sum(
            1 for act in data.get("acts", [])
            for scene in act.get("scenes", [])
            for beat in scene.get("beats", [])
            for bs in beat.get("beat_states", [])
            if bs.get("tactic_state", "").strip()
        )
        print(f"  {play_id}: {assigned}/{total} beat states normalized")
        path.write_text(json.dumps(data, indent=2))
        print(f"    → Saved to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Canonical tactic vocabulary tools")
    sub = parser.add_subparsers(dest="command")

    build_p = sub.add_parser("build", help="Build vocabulary from parsed plays")
    build_p.add_argument("--threshold", type=float, default=DEFAULT_DISTANCE_THRESHOLD,
                         help=f"Cosine distance threshold (default: {DEFAULT_DISTANCE_THRESHOLD})")

    sub.add_parser("show", help="Display vocabulary clusters for review")
    sub.add_parser("normalize", help="Assign canonical tactics to all parsed BeatStates")

    ingest_p = sub.add_parser("ingest", help="Add new tactics from a play to the vocabulary")
    ingest_p.add_argument("play_id", help="Play ID to ingest new tactics from")

    args = parser.parse_args()

    if args.command == "build":
        _cmd_build(args)
    elif args.command == "show":
        _cmd_show(args)
    elif args.command == "normalize":
        _cmd_normalize(args)
    elif args.command == "ingest":
        vocab = load_vocabulary()
        raw = collect_raw_tactics([args.play_id])
        new_tactics = list(raw.keys())
        vocab = assign_new_tactics(vocab, new_tactics)
        save_vocabulary(vocab)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
