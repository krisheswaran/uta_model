'use client';

import { useEffect, useState } from 'react';
import { ArrowLeft, Globe, Users, BookOpen, Link2, Activity, ChevronDown, ChevronRight, Search } from 'lucide-react';
import NavRail from './NavRail';
import TruncatedChip from './TruncatedChip';
import BeatSegmentationChart from './charts/BeatSegmentationChart';
import type { Play, CharacterBible, SceneBible, RelationshipEdge, SmoothedPlay } from '@/lib/types';
import { fetchSmoothedPlay } from '@/lib/data';

interface Props {
  playId: string;
}

export default function PlayView({ playId }: Props) {
  const [play, setPlay] = useState<Play | null>(null);
  const [smoothedPlay, setSmoothedPlay] = useState<SmoothedPlay | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`/data/bibles/${playId}_bibles.json`)
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }),
      fetchSmoothedPlay(playId),
    ])
      .then(([playData, smoothed]) => {
        setPlay(playData as Play);
        setSmoothedPlay(smoothed);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [playId]);

  return (
    <div style={{ minHeight: '100vh', background: 'var(--md-sys-color-background)' }}>
      <NavRail playId={playId} />

      <main
        style={{ flex: 1, padding: '24px 16px' }}
        
      >
        {/* Back + Breadcrumb */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 24 }}>
          <a
            href="/"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              color: 'var(--md-sys-color-primary)',
              textDecoration: 'none',
              fontSize: 14,
            }}
          >
            <ArrowLeft size={16} />
            All Plays
          </a>
        </div>

        {loading && <PlaySkeleton />}

        {!loading && error && (
          <div className="m3-card" style={{ color: 'var(--md-sys-color-error)', padding: 24 }}>
            Failed to load play: {error}
          </div>
        )}

        {!loading && !error && play && <PlayContent play={play} playId={playId} smoothedPlay={smoothedPlay} />}
      </main>
    </div>
  );
}

function PlaySkeleton() {
  return (
    <div>
      <div className="skeleton" style={{ height: 32, width: '50%', marginBottom: 8 }} />
      <div className="skeleton" style={{ height: 16, width: '25%', marginBottom: 32 }} />
      <div className="skeleton" style={{ height: 120, marginBottom: 16 }} />
      <div className="skeleton" style={{ height: 200, marginBottom: 16 }} />
    </div>
  );
}

function PlayContent({ play, playId, smoothedPlay }: { play: Play; playId: string; smoothedPlay: SmoothedPlay | null }) {
  const [worldBibleOpen, setWorldBibleOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTacticChip, setActiveTacticChip] = useState<string | null>(null);

  // Build smoothed tactic distributions per character from factor graph output
  const smoothedTacticDists = (() => {
    const dists: Record<string, Record<string, number>> = {};
    if (!smoothedPlay) return dists;
    for (const [char, charData] of Object.entries(smoothedPlay.characters)) {
      const counts: Record<string, number> = {};
      for (const beat of charData.beats) {
        const tactic = beat.smoothed_tactic;
        if (tactic) counts[tactic] = (counts[tactic] ?? 0) + 1;
      }
      dists[char] = counts;
    }
    return dists;
  })();

  // Get tactic distribution for a character — prefer smoothed (factor graph) over LLM
  function getTacticDist(cb: CharacterBible): Record<string, number> {
    return smoothedTacticDists[cb.character] ?? smoothedTacticDists[cb.character.toUpperCase()] ?? cb.tactic_distribution;
  }

  // Compute top 8 most common tactics across all characters (using factor graph labels when available)
  const topTactics = (() => {
    const tacticCounts: Record<string, number> = {};
    for (const cb of play.character_bibles) {
      const dist = getTacticDist(cb);
      for (const [tactic, weight] of Object.entries(dist)) {
        tacticCounts[tactic] = (tacticCounts[tactic] ?? 0) + weight;
      }
    }
    return Object.entries(tacticCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 8)
      .map(([t]) => t);
  })();

  // Filter characters
  const filteredCharacters = play.character_bibles.filter((cb) => {
    const q = searchQuery.toLowerCase();
    const dist = getTacticDist(cb);
    const matchesSearch = !q
      || cb.character.toLowerCase().includes(q)
      || cb.superobjective.toLowerCase().includes(q)
      || Object.keys(dist).some((t) => t.toLowerCase().includes(q))
      || cb.recurring_tactics.some((t) => t.toLowerCase().includes(q));

    const matchesChip = !activeTacticChip
      || dist[activeTacticChip] != null
      || cb.recurring_tactics.includes(activeTacticChip);

    return matchesSearch && matchesChip;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24, maxWidth: 1100 }}>
      {/* Header */}
      <header>
        <h1
          style={{
            fontFamily: 'var(--md-sys-typescale-display-font)',
            fontSize: 32,
            fontWeight: 400,
            margin: 0,
            color: 'var(--md-sys-color-on-surface)',
          }}
        >
          {play.title}
        </h1>
        <p style={{ color: 'var(--md-sys-color-on-surface-variant)', margin: '4px 0 0', fontSize: 15 }}>
          {play.author}
        </p>
      </header>

      {/* World Bible (collapsible) */}
      {play.world_bible && (
        <section id="world">
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginBottom: 12,
              color: 'var(--md-sys-color-primary)',
              cursor: 'pointer',
            }}
            onClick={() => setWorldBibleOpen((v) => !v)}
          >
            <Globe size={18} />
            <h2
              style={{
                fontFamily: 'var(--md-sys-typescale-display-font)',
                fontSize: 18,
                fontWeight: 500,
                margin: 0,
                color: 'var(--md-sys-color-on-surface)',
              }}
            >
              World Bible
            </h2>
            {worldBibleOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </div>
          {worldBibleOpen && <WorldBibleCard wb={play.world_bible} />}
        </section>
      )}

      {/* Characters */}
      {play.character_bibles.length > 0 && (
        <section>
          <SectionHeading icon={<Users size={18} />} title="Characters" />

          {/* Search input */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginBottom: 12,
              background: 'var(--md-sys-color-surface-container-high)',
              borderRadius: 28,
              padding: '8px 16px',
            }}
          >
            <Search size={16} style={{ color: 'var(--md-sys-color-on-surface-variant)', flexShrink: 0 }} />
            <input
              type="text"
              placeholder="Search characters by name, superobjective, or tactic..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                flex: 1,
                background: 'none',
                border: 'none',
                outline: 'none',
                color: 'var(--md-sys-color-on-surface)',
                fontSize: 14,
                fontFamily: 'var(--md-sys-typescale-body-font)',
              }}
            />
          </div>

          {/* Tactic filter chips */}
          {topTactics.length > 0 && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
              {topTactics.map((tactic) => (
                <button
                  key={tactic}
                  onClick={() => setActiveTacticChip((prev) => (prev === tactic ? null : tactic))}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    borderRadius: 8,
                    padding: '4px 12px',
                    fontSize: 12,
                    fontWeight: 500,
                    border: activeTacticChip === tactic
                      ? '1px solid var(--md-sys-color-primary)'
                      : '1px solid var(--md-sys-color-outline-variant)',
                    background: activeTacticChip === tactic
                      ? 'var(--md-sys-color-primary-container)'
                      : 'transparent',
                    color: activeTacticChip === tactic
                      ? 'var(--md-sys-color-on-primary-container)'
                      : 'var(--md-sys-color-on-surface-variant)',
                    cursor: 'pointer',
                    transition: 'background 0.2s, border-color 0.2s',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {tactic}
                </button>
              ))}
            </div>
          )}

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
              gap: 12,
            }}
          >
            {filteredCharacters.map((cb) => (
              <CharacterCard key={cb.character} cb={cb} playId={playId} tacticDist={getTacticDist(cb)} />
            ))}
          </div>
          {filteredCharacters.length === 0 && (
            <p style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14 }}>
              No characters match your search.
            </p>
          )}
        </section>
      )}

      {/* Beat Segmentation */}
      <section>
        <SectionHeading icon={<Activity size={18} />} title="Beat Segmentation" />
        <div className="m3-card" style={{ padding: '16px 20px' }}>
          <BeatSegmentationChart playId={playId} play={play} />
        </div>
      </section>

      {/* Scenes */}
      {play.scene_bibles.length > 0 && (
        <section>
          <SectionHeading icon={<BookOpen size={18} />} title="Scenes" />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {play.scene_bibles.map((sb) => (
              <SceneBibleRow key={`${sb.act}-${sb.scene}`} sb={sb} playId={playId} />
            ))}
          </div>
        </section>
      )}

      {/* Relationships */}
      {play.relationship_edges.length > 0 && (
        <section>
          <SectionHeading icon={<Link2 size={18} />} title="Relationships" />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {play.relationship_edges.map((edge, i) => (
              <RelationshipRow key={i} edge={edge} playId={playId} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function SectionHeading({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        marginBottom: 12,
        color: 'var(--md-sys-color-primary)',
      }}
    >
      {icon}
      <h2
        style={{
          fontFamily: 'var(--md-sys-typescale-display-font)',
          fontSize: 18,
          fontWeight: 500,
          margin: 0,
          color: 'var(--md-sys-color-on-surface)',
        }}
      >
        {title}
      </h2>
    </div>
  );
}

function WorldBibleCard({ wb }: { wb: NonNullable<Play['world_bible']> }) {
  return (
    <div className="m3-card-elevated fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <LabeledField label="Era" value={wb.era} />
        <LabeledField label="Genre" value={wb.genre} />
      </div>

      {wb.social_norms.length > 0 && (
        <div>
          <p className="m3-label-medium" style={{ marginBottom: 6 }}>SOCIAL NORMS</p>
          <ul style={{ margin: 0, paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {wb.social_norms.map((n, i) => (
              <li key={i} className="m3-body-medium">{n}</li>
            ))}
          </ul>
        </div>
      )}

      {wb.factual_timeline.length > 0 && (
        <div>
          <p className="m3-label-medium" style={{ marginBottom: 6 }}>FACTUAL TIMELINE</p>
          <ul style={{ margin: 0, paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {wb.factual_timeline.map((t, i) => (
              <li key={i} className="m3-body-medium">{t}</li>
            ))}
          </ul>
        </div>
      )}

      {wb.genre_constraints.length > 0 && (
        <div>
          <p className="m3-label-medium" style={{ marginBottom: 6 }}>GENRE CONSTRAINTS</p>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {wb.genre_constraints.map((gc, i) => (
              <TruncatedChip key={i} text={gc} truncate className="m3-chip m3-chip-tertiary" />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function LabeledField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="m3-label-medium" style={{ marginBottom: 2 }}>{label.toUpperCase()}</p>
      <p style={{ margin: 0, color: 'var(--md-sys-color-on-surface)', fontSize: 15 }}>{value}</p>
    </div>
  );
}

function CharacterCard({ cb, playId, tacticDist }: { cb: CharacterBible; playId: string; tacticDist: Record<string, number> }) {
  const tactics = Object.entries(tacticDist)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3);

  return (
    <a
      href={`/plays/${playId}/characters/${encodeURIComponent(cb.character)}`}
      style={{ textDecoration: 'none' }}
    >
      <div
        className="m3-card fade-in"
        style={{
          cursor: 'pointer',
          transition: 'background 0.2s, transform 0.15s',
          height: '100%',
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLDivElement).style.background =
            'var(--md-sys-color-surface-container-high)';
          (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-2px)';
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLDivElement).style.background =
            'var(--md-sys-color-surface-container)';
          (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)';
        }}
      >
        <p
          style={{
            fontFamily: 'var(--md-sys-typescale-display-font)',
            fontSize: 16,
            fontWeight: 500,
            color: 'var(--md-sys-color-primary)',
            margin: '0 0 6px',
          }}
        >
          {cb.character}
        </p>
        <p
          className="m3-body-medium"
          style={{
            margin: '0 0 10px',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
            fontSize: 12,
          }}
        >
          {cb.superobjective}
        </p>
        {tactics.length > 0 && (
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {tactics.map(([tactic]) => (
              <span key={tactic} className="m3-chip" style={{ fontSize: 10, padding: '2px 6px' }}>
                {tactic}
              </span>
            ))}
          </div>
        )}
      </div>
    </a>
  );
}

function SceneBibleRow({ sb, playId }: { sb: SceneBible; playId: string }) {
  return (
    <a
      href={`/plays/${playId}/scenes/${sb.act}/${sb.scene}`}
      style={{ textDecoration: 'none' }}
    >
      <div
        className="m3-card fade-in"
        style={{
          cursor: 'pointer',
          display: 'flex',
          gap: 16,
          alignItems: 'flex-start',
          transition: 'background 0.2s',
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLDivElement).style.background =
            'var(--md-sys-color-surface-container-high)';
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLDivElement).style.background =
            'var(--md-sys-color-surface-container)';
        }}
      >
        <div
          style={{
            minWidth: 56,
            textAlign: 'center',
            padding: '4px 8px',
            background: 'var(--md-sys-color-primary-container)',
            borderRadius: 8,
          }}
        >
          <p style={{ margin: 0, fontSize: 10, color: 'var(--md-sys-color-on-primary-container)', fontWeight: 600 }}>
            ACT {sb.act}
          </p>
          <p style={{ margin: 0, fontSize: 18, color: 'var(--md-sys-color-on-primary-container)', fontWeight: 700, lineHeight: 1.2 }}>
            {sb.scene}
          </p>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <p
            style={{
              margin: '0 0 4px',
              fontSize: 14,
              fontWeight: 500,
              color: 'var(--md-sys-color-on-surface)',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            {sb.dramatic_pressure}
          </p>
          {sb.what_changes && (
            <p
              className="m3-body-medium"
              style={{
                margin: 0,
                fontSize: 12,
                display: '-webkit-box',
                WebkitLineClamp: 1,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
              }}
            >
              Changes: {sb.what_changes}
            </p>
          )}
        </div>
      </div>
    </a>
  );
}

function RelationshipRow({ edge, playId }: { edge: RelationshipEdge; playId: string }) {
  return (
    <div className="m3-card fade-in" style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
      <a
        href={`/plays/${playId}/characters/${encodeURIComponent(edge.character_a)}`}
        style={{ color: 'var(--md-sys-color-primary)', textDecoration: 'none', fontWeight: 500, fontSize: 14 }}
      >
        {edge.character_a}
      </a>
      <span style={{ color: 'var(--md-sys-color-outline)', fontSize: 12 }}>↔</span>
      <a
        href={`/plays/${playId}/characters/${encodeURIComponent(edge.character_b)}`}
        style={{ color: 'var(--md-sys-color-primary)', textDecoration: 'none', fontWeight: 500, fontSize: 14 }}
      >
        {edge.character_b}
      </a>
      <span
        className="m3-body-medium"
        style={{
          flex: 1,
          fontSize: 13,
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}
      >
        {edge.trajectory}
      </span>
    </div>
  );
}
