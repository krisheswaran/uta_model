'use client';

import { useEffect, useMemo, useState } from 'react';
import type { TacticVocabulary, TacticEntry, RelationalProfile, Beat } from '@/lib/types';
import NavRail from './NavRail';
import RelationalSpacePlot from './charts/RelationalSpacePlot';
import TacticTransitionMatrix from './charts/TacticTransitionMatrix';

// ─── Tactic card ──────────────────────────────────────────────────────────────

const CATEGORY_COLORS: Record<string, string> = {
  offensive: '#f28b82',
  defensive: '#8ab4f8',
  affiliative: '#81c995',
  persuasive: '#EFB8C8',
  evasive: '#CCC2DC',
  '': '#D0BCFF',
};

function TacticCard({ tactic, highlight }: { tactic: TacticEntry; highlight?: string }) {
  const color = CATEGORY_COLORS[tactic.category] ?? CATEGORY_COLORS[''];
  const desc = tactic.description.replace(/^[a-z_]+ — /i, '');

  function hl(text: string) {
    if (!highlight) return <>{text}</>;
    const idx = text.toLowerCase().indexOf(highlight.toLowerCase());
    if (idx < 0) return <>{text}</>;
    return (
      <>
        {text.slice(0, idx)}
        <mark style={{ background: 'rgba(208,188,255,0.4)', color: 'inherit', borderRadius: 2 }}>
          {text.slice(idx, idx + highlight.length)}
        </mark>
        {text.slice(idx + highlight.length)}
      </>
    );
  }

  return (
    <div style={{
      background: 'var(--md-sys-color-surface-container)',
      borderRadius: 12, padding: '12px 14px',
      border: '1px solid var(--md-sys-color-outline-variant)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span style={{
          fontFamily: 'var(--md-sys-typescale-display-font)',
          fontSize: 13, fontWeight: 700, letterSpacing: '0.04em',
          color,
        }}>
          {hl(tactic.canonical_id)}
        </span>
        {tactic.category && (
          <span style={{
            fontSize: 10, padding: '1px 7px', borderRadius: 10,
            background: `${color}22`, color,
            border: `1px solid ${color}44`,
          }}>
            {tactic.category}
          </span>
        )}
      </div>
      <p style={{ margin: '0 0 8px', fontSize: 12, color: 'var(--md-sys-color-on-surface)', lineHeight: 1.5 }}>
        {hl(desc)}
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {tactic.members.map((m) => (
          <span key={m} className="m3-chip" style={{ fontSize: 10 }}>{hl(m)}</span>
        ))}
      </div>
    </div>
  );
}

// ─── Tactic browser ───────────────────────────────────────────────────────────

function TacticBrowser({ vocab }: { vocab: TacticVocabulary }) {
  const [query, setQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');

  const categories = useMemo(() => {
    const cats = new Set(vocab.tactics.map((t) => t.category).filter(Boolean));
    return Array.from(cats).sort();
  }, [vocab]);

  const filtered = useMemo(() => {
    return vocab.tactics.filter((t) => {
      if (categoryFilter && t.category !== categoryFilter) return false;
      if (!query) return true;
      const q = query.toLowerCase();
      return (
        t.canonical_id.toLowerCase().includes(q) ||
        t.canonical_verb.toLowerCase().includes(q) ||
        t.description.toLowerCase().includes(q) ||
        t.members.some((m) => m.toLowerCase().includes(q))
      );
    });
  }, [vocab, query, categoryFilter]);

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
        <input
          type="search"
          placeholder="Search tactics…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{
            flex: 1, minWidth: 200, maxWidth: 360,
            padding: '8px 14px', borderRadius: 24,
            border: '1px solid var(--md-sys-color-outline)',
            background: 'var(--md-sys-color-surface-container)',
            color: 'var(--md-sys-color-on-surface)',
            fontSize: 13,
            outline: 'none',
          }}
        />
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          <button
            onClick={() => setCategoryFilter('')}
            className="m3-chip"
            style={{
              background: !categoryFilter ? 'var(--md-sys-color-secondary-container)' : undefined,
              color: !categoryFilter ? 'var(--md-sys-color-on-secondary-container)' : undefined,
              cursor: 'pointer', border: 'none',
            }}
          >
            all
          </button>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setCategoryFilter(categoryFilter === cat ? '' : cat)}
              className="m3-chip"
              style={{
                background: categoryFilter === cat ? 'var(--md-sys-color-secondary-container)' : undefined,
                color: categoryFilter === cat ? 'var(--md-sys-color-on-secondary-container)' : undefined,
                cursor: 'pointer', border: 'none',
              }}
            >
              {cat}
            </button>
          ))}
        </div>
        <span style={{ fontSize: 12, color: 'var(--md-sys-color-on-surface-variant)', marginLeft: 'auto' }}>
          {filtered.length} / {vocab.tactics.length} tactics
        </span>
      </div>

      {vocab.unmapped.length > 0 && (
        <div style={{
          fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)',
          marginBottom: 12, padding: '6px 12px',
          background: 'var(--md-sys-color-surface-container-low)',
          borderRadius: 8,
        }}>
          Unmapped verbs ({vocab.unmapped.length}): {vocab.unmapped.join(', ')}
        </div>
      )}

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
        gap: 10,
      }}>
        {filtered.map((t) => (
          <TacticCard key={t.canonical_id} tactic={t} highlight={query} />
        ))}
      </div>
    </div>
  );
}

// ─── Relational + Tactic Matrix section ───────────────────────────────────────

interface PlayVocabData {
  playId: string;
  profiles: RelationalProfile[];
  beats: Beat[];
}

function PlayAnalysisSection({ data }: { data: PlayVocabData }) {
  const [focusCharacter, setFocusCharacter] = useState<string | undefined>(undefined);
  const characters = data.profiles.map((p) => p.character);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div>
        <h3 style={{ margin: '0 0 8px', fontSize: 14, fontWeight: 600, color: 'var(--md-sys-color-on-surface)' }}>
          Social Geometry — {data.playId}
        </h3>
        <p style={{ margin: '0 0 12px', fontSize: 12, color: 'var(--md-sys-color-on-surface-variant)' }}>
          Default relational position of each character. Click a character to see their partner-specific deviations.
        </p>

        {/* Character focus selector */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 12 }}>
          {characters.map((c) => (
            <button
              key={c}
              onClick={() => setFocusCharacter(focusCharacter === c ? undefined : c)}
              className="m3-chip"
              style={{
                cursor: 'pointer', border: 'none',
                background: focusCharacter === c ? 'var(--md-sys-color-secondary-container)' : undefined,
                color: focusCharacter === c ? 'var(--md-sys-color-on-secondary-container)' : undefined,
                fontSize: 11,
              }}
            >
              {c}
            </button>
          ))}
        </div>

        <RelationalSpacePlot profiles={data.profiles} focusCharacter={focusCharacter} />
      </div>

      {data.beats.length > 0 && focusCharacter && (
        <div>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, fontWeight: 600, color: 'var(--md-sys-color-on-surface)' }}>
            Tactic Transition Matrix — {focusCharacter}
          </h3>
          <p style={{ margin: '0 0 12px', fontSize: 12, color: 'var(--md-sys-color-on-surface-variant)' }}>
            P(next tactic | current tactic) from extracted beats.
          </p>
          <TacticTransitionMatrix beats={data.beats} character={focusCharacter} />
        </div>
      )}

      {data.beats.length === 0 && (
        <div style={{ fontSize: 12, color: 'var(--md-sys-color-on-surface-variant)', fontStyle: 'italic' }}>
          No beat data found for {data.playId}. Run the pipeline and sync to compute transition matrices.
        </div>
      )}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    return r.json() as Promise<T>;
  } catch {
    return null;
  }
}

export default function VocabView({ playId }: { playId?: string }) {
  const [vocab, setVocab] = useState<TacticVocabulary | null>(null);
  const [playDataMap, setPlayDataMap] = useState<Record<string, PlayVocabData>>({});
  const [activePlay, setActivePlay] = useState<string>(playId ?? '');
  const [tab, setTab] = useState<'tactics' | 'relational'>('tactics');
  const [error, setError] = useState<string | null>(null);

  // Load vocab
  useEffect(() => {
    fetch('/data/vocab/tactic_vocabulary.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<TacticVocabulary>;
      })
      .then(setVocab)
      .catch((e) => setError(e.message));
  }, []);

  // Load play index to discover available plays
  useEffect(() => {
    fetch('/data/index.json')
      .then((r) => r.ok ? r.json() as Promise<{ plays?: { id: string }[] }> : null)
      .then(async (idx) => {
        if (!idx) return;
        const plays: string[] = idx.plays?.map((p) => p.id) ?? [];
        if (plays.length > 0 && !activePlay) setActivePlay(plays[0]);

        for (const pid of plays) {
          // Fetch relational profiles
          const profiles = await fetchJson<RelationalProfile[]>(`/data/vocab/${pid}_relational_profiles.json`);
          if (!profiles) continue;

          // Fetch beats for transition matrices
          const beatData = await fetchJson<{ acts: { scenes: { beats: Beat[] }[] }[] }>(`/data/bibles/${pid}_bibles.json`);
          const beats: Beat[] = [];
          if (beatData?.acts) {
            for (const act of beatData.acts) {
              for (const scene of act.scenes ?? []) {
                for (const beat of scene.beats ?? []) {
                  beats.push(beat);
                }
              }
            }
          }

          setPlayDataMap((prev) => ({ ...prev, [pid]: { playId: pid, profiles, beats } }));
        }
      })
      .catch(() => {});
  }, []);

  const availablePlays = Object.keys(playDataMap);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--md-sys-color-background)' }}>
      <NavRail playId={playId} />

      <main
        className="ml-0 mb-[84px] md:ml-[88px] md:mb-0"
        style={{ flex: 1, padding: '24px 20px', maxWidth: 960 }}
      >
        <div style={{ marginBottom: 24 }}>
          <h1 style={{
            fontFamily: 'var(--md-sys-typescale-display-font)',
            fontSize: 28, fontWeight: 700,
            color: 'var(--md-sys-color-on-surface)',
            marginBottom: 6,
          }}>
            Tactic Vocabulary & Social Models
          </h1>
          <p style={{ fontSize: 13, color: 'var(--md-sys-color-on-surface-variant)', maxWidth: 600 }}>
            Canonical tactic ontology with {vocab ? `${vocab.tactics.length} entries` : '…'} derived from pipeline runs,
            plus relational geometry and tactic transition probabilities per character.
          </p>
        </div>

        {error && (
          <div style={{ padding: 16, borderRadius: 12, background: 'var(--md-sys-color-error-container)', color: 'var(--md-sys-color-on-error-container)', marginBottom: 16 }}>
            {error}
          </div>
        )}

        {/* Tab bar */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid var(--md-sys-color-outline-variant)' }}>
          {(['tactics', 'relational'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                color: tab === t ? 'var(--md-sys-color-primary)' : 'var(--md-sys-color-on-surface-variant)',
                borderBottom: tab === t ? '2px solid var(--md-sys-color-primary)' : '2px solid transparent',
                paddingBottom: 8, marginBottom: -1,
                fontSize: 13, fontWeight: tab === t ? 600 : 400,
                textTransform: 'capitalize',
              }}
            >
              {t === 'tactics' ? 'Tactic Ontology' : 'Relational Models'}
            </button>
          ))}
        </div>

        {tab === 'tactics' && (
          vocab
            ? <TacticBrowser vocab={vocab} />
            : <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {[1, 2, 3, 4].map((i) => <div key={i} className="skeleton" style={{ height: 80, borderRadius: 12 }} />)}
              </div>
        )}

        {tab === 'relational' && (
          <div>
            {/* Play selector */}
            {availablePlays.length > 1 && (
              <div style={{ display: 'flex', gap: 6, marginBottom: 16 }}>
                {availablePlays.map((pid) => (
                  <button
                    key={pid}
                    onClick={() => setActivePlay(pid)}
                    className="m3-chip"
                    style={{
                      cursor: 'pointer', border: 'none', fontSize: 12,
                      background: activePlay === pid ? 'var(--md-sys-color-secondary-container)' : undefined,
                      color: activePlay === pid ? 'var(--md-sys-color-on-secondary-container)' : undefined,
                    }}
                  >
                    {pid}
                  </button>
                ))}
              </div>
            )}

            {availablePlays.length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div className="skeleton" style={{ height: 420, borderRadius: 16 }} />
              </div>
            )}

            {activePlay && playDataMap[activePlay] && (
              <PlayAnalysisSection data={playDataMap[activePlay]} />
            )}
          </div>
        )}
      </main>
    </div>
  );
}
