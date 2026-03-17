"""
Relationship Builder — Phase B

Constructs relationship data from existing BeatState social_state values:

  Level 1: Pairwise RelationshipEdges — warmth/status time series per character pair
  Level 2: RelationalProfiles — per-character aggregate social tendencies
  Level 3: Cross-play pooling (computed at improv time, not here)

All numeric aggregation is pure computation over existing data (no LLM calls).
Only the optional RelationshipEdge.summary field requires an LLM call.

Usage:
    # Build relationships for a play (no API calls for numerics)
    conda run -n uta_model python analysis/relationship_builder.py cherry_orchard

    # Build with LLM-generated summaries (~$0.02/pair)
    conda run -n uta_model python analysis/relationship_builder.py cherry_orchard --summaries
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pvariance

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import BeatState, Play, RelationalProfile, RelationshipEdge


# ────────────────────────────────────────────────────────────────────────────
# Level 1: Pairwise RelationshipEdges
# ────────────────────────────────────────────────────────────────────────────

def _collect_pairwise_social_data(play: Play) -> dict[tuple[str, str], list[tuple[str, float, float]]]:
    """Collect per-beat social_state data for each directed character pair.

    Returns:
        Dict mapping (character_a, character_b) -> list of (beat_id, warmth_a_toward_b, status_a_vs_b)
        These are DIRECTED: (A, B) captures A's warmth/status when B is present.
    """
    # First, build a map of which characters are present per beat
    pairwise: dict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)

    for act in play.acts:
        for scene in act.scenes:
            for beat in scene.beats:
                # Get all beat_states in this beat
                states_by_char: dict[str, BeatState] = {}
                for bs in beat.beat_states:
                    states_by_char[bs.character] = bs

                chars = list(states_by_char.keys())
                # For each pair of characters present in this beat,
                # record each character's social_state (directed toward the other)
                for a in chars:
                    for b in chars:
                        if a != b:
                            bs_a = states_by_char[a]
                            pairwise[(a, b)].append((
                                beat.id,
                                bs_a.social_state.warmth,
                                bs_a.social_state.status,
                            ))

    return dict(pairwise)


def build_pairwise_edges(
    play: Play,
    min_beats: int = 3,
) -> list[RelationshipEdge]:
    """Build directed RelationshipEdges from existing BeatState social_state data.

    Args:
        play: Play with BeatStates already extracted.
        min_beats: Minimum co-occurring beats to create an edge.

    Returns:
        List of RelationshipEdge objects (directed: A->B and B->A are separate).
    """
    pairwise = _collect_pairwise_social_data(play)

    edges = []
    for (char_a, char_b), records in pairwise.items():
        if len(records) < min_beats:
            continue

        temp_by_beat = {}
        power_by_beat = {}
        for beat_id, warmth, status in records:
            temp_by_beat[beat_id] = warmth
            power_by_beat[beat_id] = status

        edges.append(RelationshipEdge(
            play_id=play.id,
            character_a=char_a,
            character_b=char_b,
            temperature_by_beat=temp_by_beat,
            power_by_beat=power_by_beat,
            summary="",  # populated separately if --summaries is used
        ))

    return edges


# ────────────────────────────────────────────────────────────────────────────
# Level 2: RelationalProfiles (per-character aggregate)
# ────────────────────────────────────────────────────────────────────────────

def build_relational_profiles(
    play: Play,
    edges: list[RelationshipEdge] | None = None,
) -> list[RelationalProfile]:
    """Build a RelationalProfile for each character from their pairwise edges.

    The profile captures how a character GENERALLY relates to others (directed),
    plus per-partner deviations from their default.

    Args:
        play: Play object (for play_id and character list).
        edges: Pre-built edges. If None, builds them from the play.
    """
    if edges is None:
        edges = build_pairwise_edges(play)

    # Group edges by character_a (the "from" character)
    edges_by_char: dict[str, list[RelationshipEdge]] = defaultdict(list)
    for edge in edges:
        edges_by_char[edge.character_a].append(edge)

    profiles = []
    for character, char_edges in edges_by_char.items():
        if not char_edges:
            continue

        # Compute per-partner mean warmth and status
        partner_warmths: dict[str, float] = {}
        partner_statuses: dict[str, float] = {}

        for edge in char_edges:
            warmth_vals = list(edge.temperature_by_beat.values())
            status_vals = list(edge.power_by_beat.values())
            if warmth_vals:
                partner_warmths[edge.character_b] = mean(warmth_vals)
            if status_vals:
                partner_statuses[edge.character_b] = mean(status_vals)

        if not partner_warmths:
            continue

        # Aggregate across all partners
        all_warmths = list(partner_warmths.values())
        all_statuses = list(partner_statuses.values())

        default_warmth = mean(all_warmths)
        default_status = mean(all_statuses)
        warmth_var = pvariance(all_warmths) if len(all_warmths) > 1 else 0.0
        status_var = pvariance(all_statuses) if len(all_statuses) > 1 else 0.0

        # Per-partner deviations from default
        partner_deviations = {}
        for partner in partner_warmths:
            partner_deviations[partner] = {
                "warmth_delta": round(partner_warmths[partner] - default_warmth, 4),
                "status_delta": round(partner_statuses.get(partner, default_status) - default_status, 4),
            }

        profiles.append(RelationalProfile(
            character=character,
            play_id=play.id,
            default_status_claim=round(default_status, 4),
            default_warmth=round(default_warmth, 4),
            status_variance=round(status_var, 4),
            warmth_variance=round(warmth_var, 4),
            partner_deviations=partner_deviations,
        ))

    # Sort by number of partners (most connected first)
    profiles.sort(key=lambda p: -len(p.partner_deviations))
    return profiles


# ────────────────────────────────────────────────────────────────────────────
# Optional: LLM-generated summaries for pairwise edges
# ────────────────────────────────────────────────────────────────────────────

def generate_edge_summaries(
    edges: list[RelationshipEdge],
    play: Play,
) -> list[RelationshipEdge]:
    """Add LLM-generated summary to each RelationshipEdge.

    Uses the critic model (Sonnet) to describe the relationship trajectory.
    Cost: ~$0.02 per edge.
    """
    import anthropic
    from config import ANTHROPIC_API_KEY, get_model

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    for edge in edges:
        if edge.summary:
            continue  # already has a summary

        warmth_vals = list(edge.temperature_by_beat.values())
        status_vals = list(edge.power_by_beat.values())
        n = len(warmth_vals)

        prompt = (
            f"PLAY: {play.title}\n"
            f"CHARACTER A: {edge.character_a}\n"
            f"CHARACTER B: {edge.character_b}\n"
            f"CO-OCCURRING BEATS: {n}\n"
            f"A's warmth toward B over time: {', '.join(f'{v:.1f}' for v in warmth_vals)}\n"
            f"A's status claim vs B over time: {', '.join(f'{v:.1f}' for v in status_vals)}\n\n"
            f"In 1-2 sentences, describe how {edge.character_a} relates to "
            f"{edge.character_b} across the play — the emotional temperature, "
            f"power dynamic, and any shifts."
        )

        response = client.messages.create(
            model=get_model("critic"),
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        edge.summary = response.content[0].text.strip()
        print(f"    {edge.character_a} → {edge.character_b}: {edge.summary[:80]}...")

    return edges


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────

def build_all_relationships(
    play: Play,
    min_beats: int = 3,
    generate_summaries: bool = False,
) -> Play:
    """Build and attach all relationship data to the play.

    Args:
        play: Play with BeatStates already extracted.
        min_beats: Minimum co-occurring beats for an edge.
        generate_summaries: If True, use LLM to generate edge summaries.

    Returns:
        Play with relationship_edges populated and relational profiles
        stored as JSON alongside bibles.
    """
    print(f"[relationship_builder] Building relationships for {play.title}...")

    # Level 1: Pairwise edges
    edges = build_pairwise_edges(play, min_beats=min_beats)
    print(f"  [+] {len(edges)} directed pairwise edges (min {min_beats} co-occurring beats)")

    if generate_summaries:
        print("  Generating edge summaries...")
        edges = generate_edge_summaries(edges, play)

    play.relationship_edges = edges

    # Level 2: Relational profiles
    profiles = build_relational_profiles(play, edges)
    print(f"  [+] {len(profiles)} relational profiles")

    for p in profiles:
        n_partners = len(p.partner_deviations)
        print(f"    {p.character}: default warmth={p.default_warmth:+.2f}, "
              f"status={p.default_status_claim:+.2f}, "
              f"warmth_var={p.warmth_variance:.3f}, "
              f"{n_partners} partners")

    return play, profiles


# ────────────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────────────

def save_profiles(profiles: list[RelationalProfile], play_id: str) -> Path:
    """Save relational profiles to data/vocab/{play_id}_relational_profiles.json."""
    import json
    from config import DATA_DIR

    out_dir = DATA_DIR / "vocab"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{play_id}_relational_profiles.json"
    data = [p.model_dump() for p in profiles]
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved profiles to {path}")
    return path


def load_profiles(play_id: str) -> list[RelationalProfile]:
    """Load relational profiles from disk."""
    import json
    from config import DATA_DIR

    path = DATA_DIR / "vocab" / f"{play_id}_relational_profiles.json"
    if not path.exists():
        raise FileNotFoundError(f"No profiles at {path}. Run relationship_builder first.")
    data = json.loads(path.read_text())
    return [RelationalProfile(**d) for d in data]


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    from config import PARSED_DIR

    parser = argparse.ArgumentParser(description="Build relationship edges and profiles")
    parser.add_argument("play_id", help="Play ID to process")
    parser.add_argument("--min-beats", type=int, default=3,
                        help="Minimum co-occurring beats for an edge (default: 3)")
    parser.add_argument("--summaries", action="store_true",
                        help="Generate LLM summaries for edges (~$0.02/pair)")
    args = parser.parse_args()

    path = PARSED_DIR / f"{args.play_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run analysis first.")

    play = Play.model_validate_json(path.read_text())
    play, profiles = build_all_relationships(
        play,
        min_beats=args.min_beats,
        generate_summaries=args.summaries,
    )

    # Save updated play (with edges)
    path.write_text(play.model_dump_json(indent=2))
    print(f"\n  Updated play saved to {path}")

    # Save profiles separately
    save_profiles(profiles, args.play_id)

    print(f"\n=== Relationships complete for {play.title} ===")


if __name__ == "__main__":
    main()
