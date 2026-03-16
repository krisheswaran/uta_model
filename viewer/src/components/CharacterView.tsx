'use client';

import { useEffect, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import NavRail from './NavRail';
import BeatStateDetail from './BeatStateDetail';
import AffectTrajectory from './charts/AffectTrajectory';
import TacticBarChart from './charts/TacticBarChart';
import BeatTimeline from './charts/BeatTimeline';
import type { Play, CharacterBible, Beat, BeatState } from '@/lib/types';
import {
  getCharacterBible,
  getAllBeatsForCharacter,
  getBeatStateForCharacter,
} from '@/lib/data';

interface Props {
  playId: string;
  character: string;
}

type TabId = 'arc' | 'affect' | 'tactics' | 'arc-by-scene';

export default function CharacterView({ playId, character }: Props) {
  const [play, setPlay] = useState<Play | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const decodedCharacter = decodeURIComponent(character);

  useEffect(() => {
    fetch(`/data/bibles/${playId}_bibles.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setPlay(data as Play);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [playId]);

  const bible = play ? getCharacterBible(play, decodedCharacter) : undefined;
  const beats = play ? getAllBeatsForCharacter(play, decodedCharacter) : [];

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--md-sys-color-background)' }}>
      <NavRail playId={playId} character={decodedCharacter} />

      <main
        style={{ flex: 1, padding: '24px 16px', marginBottom: 72 }}
        className="md:ml-[88px] md:mb-0"
      >
        {/* Breadcrumb */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 24 }}>
          <a
            href={`/plays/${playId}`}
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
            {play?.title ?? playId}
          </a>
        </div>

        {loading && <CharacterSkeleton />}
        {!loading && error && (
          <div className="m3-card" style={{ color: 'var(--md-sys-color-error)' }}>
            {error}
          </div>
        )}

        {!loading && !error && (
          <CharacterContent
            play={play!}
            playId={playId}
            character={decodedCharacter}
            bible={bible}
            beats={beats}
          />
        )}
      </main>
    </div>
  );
}

function CharacterSkeleton() {
  return (
    <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
      <div style={{ flex: '0 0 320px' }}>
        <div className="skeleton" style={{ height: 28, width: '60%', marginBottom: 12 }} />
        <div className="skeleton" style={{ height: 400 }} />
      </div>
      <div style={{ flex: 1, minWidth: 280 }}>
        <div className="skeleton" style={{ height: 40, marginBottom: 16 }} />
        <div className="skeleton" style={{ height: 300 }} />
      </div>
    </div>
  );
}

function CharacterContent({
  play,
  playId,
  character,
  bible,
  beats,
}: {
  play: Play;
  playId: string;
  character: string;
  bible: CharacterBible | undefined;
  beats: Beat[];
}) {
  const [activeTab, setActiveTab] = useState<TabId>('arc');
  const [selectedBeatId, setSelectedBeatId] = useState<string | null>(
    beats.length > 0 ? beats[0].id : null
  );

  const selectedBeat = beats.find((b) => b.id === selectedBeatId) ?? null;
  const selectedBS: BeatState | undefined = selectedBeat
    ? getBeatStateForCharacter(selectedBeat, character)
    : undefined;

  const tabs: { id: TabId; label: string }[] = [
    { id: 'arc', label: 'Arc' },
    { id: 'affect', label: 'Affect Space' },
    { id: 'tactics', label: 'Tactics' },
    { id: 'arc-by-scene', label: 'Arc by Scene' },
  ];

  return (
    <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start', maxWidth: 1200 }}>
      {/* Left column: Character Bible */}
      <div style={{ flex: '0 0 320px', minWidth: 280 }}>
        <h1
          style={{
            fontFamily: 'var(--md-sys-typescale-display-font)',
            fontSize: 26,
            fontWeight: 400,
            color: 'var(--md-sys-color-on-surface)',
            margin: '0 0 4px',
          }}
        >
          {character}
        </h1>
        <p style={{ margin: '0 0 16px', fontSize: 13, color: 'var(--md-sys-color-on-surface-variant)' }}>
          {beats.length} beats
        </p>

        {bible ? (
          <CharacterBibleCard bible={bible} />
        ) : (
          <div className="m3-card" style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14 }}>
            No character bible available for {character}.
          </div>
        )}
      </div>

      {/* Right column: Tabs */}
      <div style={{ flex: 1, minWidth: 300 }}>
        {/* Tab bar */}
        <div
          style={{
            display: 'flex',
            borderBottom: '1px solid var(--md-sys-color-outline-variant)',
            overflowX: 'auto',
            marginBottom: 20,
          }}
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`m3-tab ${activeTab === tab.id ? 'active' : ''}`}
              style={{ border: 'none', background: 'none', flexShrink: 0 }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {activeTab === 'arc' && (
          <div>
            {beats.length === 0 ? (
              <div style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14 }}>
                No beats found for {character}.
              </div>
            ) : (
              <>
                <BeatTimeline
                  beats={beats}
                  character={character}
                  selectedBeatId={selectedBeatId}
                  onSelectBeat={setSelectedBeatId}
                />
                {selectedBeat && selectedBS && (
                  <div style={{ marginTop: 16 }}>
                    <BeatStateDetail beat={selectedBeat} beatState={selectedBS} />
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {activeTab === 'affect' && (
          <div>
            <AffectTrajectory beats={beats} character={character} />
          </div>
        )}

        {activeTab === 'tactics' && (
          <div>
            {bible ? (
              <TacticBarChart distribution={bible.tactic_distribution} />
            ) : (
              <div style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14 }}>
                No tactic data available.
              </div>
            )}
          </div>
        )}

        {activeTab === 'arc-by-scene' && (
          <ArcByScene arcByScene={bible?.arc_by_scene ?? {}} playId={playId} />
        )}
      </div>
    </div>
  );
}

function CharacterBibleCard({ bible }: { bible: CharacterBible }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Superobjective */}
      <div className="m3-card-elevated">
        <p className="m3-label-medium" style={{ marginBottom: 4 }}>SUPEROBJECTIVE</p>
        <p style={{ margin: 0, fontSize: 14, color: 'var(--md-sys-color-on-surface)', lineHeight: 1.5 }}>
          {bible.superobjective}
        </p>
      </div>

      {/* Wounds / Fears / Needs */}
      {bible.wounds_fears_needs && (
        <div className="m3-card-elevated">
          <p className="m3-label-medium" style={{ marginBottom: 4 }}>WOUNDS / FEARS / NEEDS</p>
          <p style={{ margin: 0, fontSize: 13, color: 'var(--md-sys-color-on-surface-variant)', lineHeight: 1.5 }}>
            {bible.wounds_fears_needs}
          </p>
        </div>
      )}

      {/* Recurring tactics */}
      {bible.recurring_tactics.length > 0 && (
        <div>
          <p className="m3-label-medium" style={{ marginBottom: 6 }}>RECURRING TACTICS</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {bible.recurring_tactics.map((t, i) => (
              <span key={i} className="m3-chip">{t}</span>
            ))}
          </div>
        </div>
      )}

      {/* Defense mechanisms */}
      {bible.preferred_defense_mechanisms.length > 0 && (
        <div>
          <p className="m3-label-medium" style={{ marginBottom: 6 }}>DEFENSE MECHANISMS</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {bible.preferred_defense_mechanisms.map((d, i) => (
              <span key={i} className="m3-chip m3-chip-tertiary">{d}</span>
            ))}
          </div>
        </div>
      )}

      {/* Psychological contradictions */}
      {bible.psychological_contradictions.length > 0 && (
        <div className="m3-card" style={{ background: 'var(--md-sys-color-tertiary-container)' }}>
          <p className="m3-label-medium" style={{ marginBottom: 6, color: 'var(--md-sys-color-on-tertiary-container)' }}>
            PSYCHOLOGICAL CONTRADICTIONS
          </p>
          <ul style={{ margin: 0, paddingLeft: 16, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {bible.psychological_contradictions.map((c, i) => (
              <li key={i} style={{ fontSize: 13, color: 'var(--md-sys-color-on-tertiary-container)', lineHeight: 1.5 }}>{c}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Speech style */}
      {bible.speech_style && (
        <div>
          <p className="m3-label-medium" style={{ marginBottom: 4 }}>SPEECH STYLE</p>
          <p style={{ margin: 0, fontSize: 13, color: 'var(--md-sys-color-on-surface-variant)', lineHeight: 1.5 }}>
            {bible.speech_style}
          </p>
        </div>
      )}

      {/* Lexical signature */}
      {bible.lexical_signature.length > 0 && (
        <div>
          <p className="m3-label-medium" style={{ marginBottom: 6 }}>LEXICAL SIGNATURE</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {bible.lexical_signature.map((w, i) => (
              <span
                key={i}
                className="m3-chip m3-chip-primary"
                style={{ fontStyle: 'italic' }}
              >
                {w}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Rhetorical patterns */}
      {bible.rhetorical_patterns.length > 0 && (
        <div>
          <p className="m3-label-medium" style={{ marginBottom: 6 }}>RHETORICAL PATTERNS</p>
          <ul style={{ margin: 0, paddingLeft: 16, display: 'flex', flexDirection: 'column', gap: 3 }}>
            {bible.rhetorical_patterns.map((r, i) => (
              <li key={i} style={{ fontSize: 12, color: 'var(--md-sys-color-on-surface-variant)', lineHeight: 1.5 }}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Few-shot lines */}
      {bible.few_shot_lines.length > 0 && (
        <div>
          <p className="m3-label-medium" style={{ marginBottom: 6 }}>EXAMPLE LINES</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {bible.few_shot_lines.slice(0, 4).map((line, i) => (
              <blockquote
                key={i}
                style={{
                  margin: 0,
                  paddingLeft: 10,
                  borderLeft: '2px solid var(--md-sys-color-primary-container)',
                  color: 'var(--md-sys-color-on-surface)',
                  fontSize: 13,
                  lineHeight: 1.5,
                  fontStyle: 'italic',
                }}
              >
                {line}
              </blockquote>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ArcByScene({
  arcByScene,
  playId,
}: {
  arcByScene: Record<string, string>;
  playId: string;
}) {
  const entries = Object.entries(arcByScene);

  if (entries.length === 0) {
    return (
      <div style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14 }}>
        No arc-by-scene data available.
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {entries.map(([sceneKey, arcNote]) => {
        // Try to parse "A1S2" or "1-2" format for link
        const match = sceneKey.match(/[Aa]?(\d+)[Ss\-_](\d+)/);
        const linkHref = match
          ? `/plays/${playId}/scenes/${match[1]}/${match[2]}`
          : null;

        return (
          <div key={sceneKey} className="m3-card fade-in">
            <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
              <div
                style={{
                  minWidth: 48,
                  background: 'var(--md-sys-color-secondary-container)',
                  borderRadius: 6,
                  padding: '4px 6px',
                  textAlign: 'center',
                }}
              >
                {linkHref ? (
                  <a
                    href={linkHref}
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      color: 'var(--md-sys-color-on-secondary-container)',
                      textDecoration: 'none',
                    }}
                  >
                    {sceneKey}
                  </a>
                ) : (
                  <span
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      color: 'var(--md-sys-color-on-secondary-container)',
                    }}
                  >
                    {sceneKey}
                  </span>
                )}
              </div>
              <p style={{ margin: 0, fontSize: 13, color: 'var(--md-sys-color-on-surface)', lineHeight: 1.6 }}>
                {arcNote}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
