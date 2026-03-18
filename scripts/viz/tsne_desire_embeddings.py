#!/usr/bin/env python3
"""
t-SNE visualization of desire-state embeddings from the H6 experiment.

Generates a 3-panel figure:
  1. All desire strings colored by play (genre effect vs shared vocabulary)
  2. Same t-SNE colored by tactic persistence (persist vs change)
  3. Same t-SNE with ~15 annotated desire labels

Saves to docs/plots/desire_embedding_tsne.png
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "parsed"
VOCAB_PATH = PROJECT_ROOT / "data" / "vocab" / "tactic_vocabulary.json"
OUT_PATH = PROJECT_ROOT / "docs" / "plots" / "desire_embedding_tsne.png"

PLAY_IDS = ["cherry_orchard", "hamlet", "importance_of_being_earnest"]
PLAY_LABELS = {
    "cherry_orchard": "Cherry Orchard",
    "hamlet": "Hamlet",
    "importance_of_being_earnest": "Earnest",
}
PLAY_COLORS = {
    "cherry_orchard": "#2ca02c",      # green
    "hamlet": "#1f77b4",              # blue
    "importance_of_being_earnest": "#d62728",  # red
}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_tactic_vocab():
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    m2c = {}
    for entry in vocab["tactics"]:
        cid = entry["canonical_id"]
        for m in entry["members"]:
            m2c[m.lower().strip()] = cid
    return m2c


def resolve_tactic(bs, m2c):
    ct = bs.get("canonical_tactic")
    if ct:
        return ct
    ts = bs.get("tactic_state", "")
    if ts:
        return m2c.get(ts.upper().strip()) or m2c.get(ts.lower().strip())
    return None


def load_plays():
    plays = {}
    for pid in PLAY_IDS:
        path = DATA_DIR / f"{pid}.json"
        if path.exists():
            with open(path) as f:
                plays[pid] = json.load(f)
    return plays


def extract_all_desires_with_metadata(plays, m2c):
    """
    Returns two structures:
    1. desires_meta: list of dicts with {desire_str, play_id, character, act, scene, beat_idx}
       One entry per beat_state that has a non-empty desire.
    2. transitions: list of dicts with {desire_prev, desire_curr, tactic_persisted, play_id}
       For consecutive beats of the same character in the same scene.
    """
    desires_meta = []
    transitions = []

    for pid, play in plays.items():
        for act in play["acts"]:
            for scene in act["scenes"]:
                # Collect per-character beat sequences
                char_beats = defaultdict(list)
                for beat in sorted(scene["beats"], key=lambda b: b["index"]):
                    for bs in beat["beat_states"]:
                        d = bs.get("desire_state", "").strip()
                        if d:
                            desires_meta.append({
                                "desire": d,
                                "play_id": pid,
                                "character": bs["character"],
                                "act": act.get("act_number", act.get("act", 0)),
                                "scene": scene.get("scene_number", scene.get("scene", 0)),
                                "beat_idx": beat["index"],
                            })
                            char_beats[bs["character"]].append((beat["index"], bs))

                # Build transitions for tactic persistence
                for char, indexed_states in char_beats.items():
                    indexed_states.sort(key=lambda x: x[0])
                    for i in range(len(indexed_states) - 1):
                        _, bs_prev = indexed_states[i]
                        _, bs_curr = indexed_states[i + 1]
                        t_prev = resolve_tactic(bs_prev, m2c)
                        t_curr = resolve_tactic(bs_curr, m2c)
                        if t_prev is None or t_curr is None:
                            continue
                        d_prev = bs_prev.get("desire_state", "").strip()
                        d_curr = bs_curr.get("desire_state", "").strip()
                        if not d_prev or not d_curr:
                            continue
                        transitions.append({
                            "desire_prev": d_prev,
                            "desire_curr": d_curr,
                            "tactic_persisted": (t_prev == t_curr),
                            "play_id": pid,
                        })

    return desires_meta, transitions


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    plays = load_plays()
    m2c = load_tactic_vocab()
    desires_meta, transitions = extract_all_desires_with_metadata(plays, m2c)
    print(f"  {len(desires_meta)} desire instances, {len(transitions)} transitions")

    # Unique desires for embedding
    unique_desires = sorted(set(dm["desire"] for dm in desires_meta))
    print(f"  {len(unique_desires)} unique desire strings")

    # Embed
    print("Embedding desires...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(unique_desires, show_progress_bar=True, normalize_embeddings=True)
    desire_to_idx = {d: i for i, d in enumerate(unique_desires)}
    desire_to_emb = {d: embeddings[i] for i, d in enumerate(unique_desires)}

    # t-SNE on unique desires
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(unique_desires) - 1),
                learning_rate="auto", init="pca")
    coords = tsne.fit_transform(embeddings)  # shape: (n_unique, 2)

    desire_to_coord = {d: coords[i] for i, d in enumerate(unique_desires)}

    # ── Panel 1: Colored by play ─────────────────────────────────────────
    # Each unique desire may appear in multiple plays. Assign to the play
    # where it appears most frequently.
    desire_play_counts = defaultdict(lambda: defaultdict(int))
    for dm in desires_meta:
        desire_play_counts[dm["desire"]][dm["play_id"]] += 1

    desire_primary_play = {}
    for d, pcounts in desire_play_counts.items():
        desire_primary_play[d] = max(pcounts, key=pcounts.get)

    # ── Panel 2: Tactic persistence ──────────────────────────────────────
    # For each desire that appears as desire_curr in a transition, track whether
    # the tactic persisted. A desire can appear in multiple transitions;
    # we'll mark it as "persisted" if majority of its appearances had persistence.
    desire_persist_counts = defaultdict(lambda: {"persist": 0, "change": 0})
    for tr in transitions:
        d = tr["desire_curr"]
        if tr["tactic_persisted"]:
            desire_persist_counts[d]["persist"] += 1
        else:
            desire_persist_counts[d]["change"] += 1

    # ── Panel 3: Annotation selection ────────────────────────────────────
    # Strategy: pick desires that are interesting for various reasons
    # - outliers (far from centroid)
    # - tightly clustered (close neighbors)
    # - cross-play neighbors

    centroid = coords.mean(axis=0)
    dists_from_centroid = np.linalg.norm(coords - centroid, axis=1)

    # Find outliers (top 5 by distance from centroid)
    outlier_idxs = np.argsort(dists_from_centroid)[-5:]

    # Find tight cluster members: for each point, compute mean dist to 3 nearest neighbors
    from scipy.spatial.distance import cdist
    pairwise = cdist(coords, coords)
    np.fill_diagonal(pairwise, np.inf)
    nn3_mean = np.sort(pairwise, axis=1)[:, :3].mean(axis=1)
    tight_idxs = np.argsort(nn3_mean)[:5]

    # Find cross-play neighbors: pairs of desires from different plays that are close
    cross_play_pairs = []
    for i in range(len(unique_desires)):
        play_i = desire_primary_play[unique_desires[i]]
        neighbors = np.argsort(pairwise[i])[:5]
        for j in neighbors:
            play_j = desire_primary_play[unique_desires[j]]
            if play_i != play_j:
                cross_play_pairs.append((i, j, pairwise[i, j]))
    cross_play_pairs.sort(key=lambda x: x[2])
    cross_play_idxs = set()
    for i, j, _ in cross_play_pairs[:10]:
        cross_play_idxs.add(i)
        cross_play_idxs.add(j)
        if len(cross_play_idxs) >= 6:
            break

    # Combine and deduplicate annotation targets
    annotate_idxs = set(outlier_idxs) | set(tight_idxs) | cross_play_idxs
    # Limit to ~15
    annotate_idxs = list(annotate_idxs)[:15]

    # ── Plot ─────────────────────────────────────────────────────────────
    print("Plotting...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # --- Panel 1: By play ---
    ax1 = axes[0]
    for pid in PLAY_IDS:
        mask = [desire_primary_play[unique_desires[i]] == pid for i in range(len(unique_desires))]
        mask = np.array(mask)
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                    c=PLAY_COLORS[pid], label=PLAY_LABELS[pid],
                    alpha=0.6, s=25, edgecolors="none")
    ax1.set_title("Desire Embeddings by Play", fontsize=14, fontweight="bold")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.legend(loc="best", fontsize=10, framealpha=0.9)

    # --- Panel 2: By tactic persistence ---
    ax2 = axes[1]
    # Desires that appear in transitions
    persist_colors = []
    persist_xs = []
    persist_ys = []
    no_transition_xs = []
    no_transition_ys = []

    for i, d in enumerate(unique_desires):
        x, y = coords[i]
        if d in desire_persist_counts:
            pc = desire_persist_counts[d]
            total = pc["persist"] + pc["change"]
            # Majority rule
            if pc["persist"] > pc["change"]:
                persist_xs.append(x)
                persist_ys.append(y)
                persist_colors.append("#ff7f0e")  # orange = persisted
            else:
                persist_xs.append(x)
                persist_ys.append(y)
                persist_colors.append("#9467bd")  # purple = changed
        else:
            no_transition_xs.append(x)
            no_transition_ys.append(y)

    # Plot no-transition points as gray background
    ax2.scatter(no_transition_xs, no_transition_ys,
                c="#cccccc", alpha=0.3, s=15, edgecolors="none", label="No transition data")

    # Separate persist vs change for legend
    persist_mask = [c == "#ff7f0e" for c in persist_colors]
    change_mask = [c == "#9467bd" for c in persist_colors]
    persist_xs, persist_ys = np.array(persist_xs), np.array(persist_ys)
    persist_mask = np.array(persist_mask)
    change_mask = np.array(change_mask)

    if persist_mask.any():
        ax2.scatter(persist_xs[persist_mask], persist_ys[persist_mask],
                    c="#ff7f0e", alpha=0.6, s=25, edgecolors="none",
                    label="Tactic persisted")
    if change_mask.any():
        ax2.scatter(persist_xs[change_mask], persist_ys[change_mask],
                    c="#9467bd", alpha=0.6, s=25, edgecolors="none",
                    label="Tactic changed")

    ax2.set_title("Desire Embeddings by Tactic Persistence", fontsize=14, fontweight="bold")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend(loc="best", fontsize=10, framealpha=0.9)

    # --- Panel 3: Annotated ---
    ax3 = axes[2]
    # Plot all points in light gray
    ax3.scatter(coords[:, 0], coords[:, 1],
                c="#999999", alpha=0.3, s=15, edgecolors="none")
    # Highlight annotated points
    for idx in annotate_idxs:
        d = unique_desires[idx]
        x, y = coords[idx]
        play = desire_primary_play[d]
        ax3.scatter(x, y, c=PLAY_COLORS[play], s=60, edgecolors="black", linewidths=0.8, zorder=5)

        # Truncate label for readability
        label = d if len(d) <= 55 else d[:52] + "..."
        ax3.annotate(label, (x, y),
                     fontsize=6.5, fontweight="bold",
                     xytext=(8, 5), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="gray"),
                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                     zorder=6)

    ax3.set_title("Annotated Desire Strings", fontsize=14, fontweight="bold")
    ax3.set_xlabel("t-SNE 1")
    ax3.set_ylabel("t-SNE 2")

    # Color legend for panel 3
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=PLAY_COLORS[pid],
                              markersize=8, label=PLAY_LABELS[pid]) for pid in PLAY_IDS]
    ax3.legend(handles=legend_elements, loc="best", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved to {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
