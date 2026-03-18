'use client';

import { useEffect, useState } from 'react';
import { ArrowLeft, BookOpen } from 'lucide-react';
import NavRail from './NavRail';
import FullScreenModal from './FullScreenModal';
import BeatStateDetail from './BeatStateDetail';
import SmoothedBeatStateDetail from './SmoothedBeatStateDetail';
import ViewModeSelector from './ViewModeSelector';
import type { ViewMode } from './ViewModeSelector';
import AffectTrajectory from './charts/AffectTrajectory';
import TacticBarChart from './charts/TacticBarChart';
import BeatTimeline from './charts/BeatTimeline';
import type { Play, CharacterBible, Beat, BeatState, SmoothedPlay, SmoothedBeat } from '@/lib/types';
import {
  getCharacterBible,
  getAllBeatsForCharacter,
  getBeatStateForCharacter,
  getSmoothedBeatForCharacter,
  fetchSmoothedPlay,
} from '@/lib/data';

interface Props {
  playId: string;
  character: string;
}

type TabId = 'arc' | 'affect' | 'tactics' | 'arc-by-scene';

export default function CharacterView({ playId, character }: Props) {
  const [play, setPlay] = useState<Play | null>(null);
  const [smoothedPlay, setSmoothedPlay] = useState<SmoothedPlay | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const decodedCharacter = decodeURIComponent(character);

  useEffect(() => {
    Promise.all([
      fetch(`/data/bibles/${playId}_bibles.json`)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        }),
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

  const bible = play ? getCharacterBible(play, decodedCharacter) : undefined;
  const beats = play ? getAllBeatsForCharacter(play, decodedCharacter) : [];

  return (
    <div style={{ minHeight: '100vh', background: 'var(--md-sys-color-background)' }}>
      <NavRail playId={playId} character={decodedCharacter} />

      <main
        style={{ flex: 1, padding: '24px 16px' }}
        
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
            smoothedPlay={smoothedPlay}
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
  smoothedPlay,
}: {
  play: Play;
  playId: string;
  character: string;
  bible: CharacterBible | undefined;
  beats: Beat[];
  smoothedPlay: SmoothedPlay | null;
}) {
  const [activeTab, setActiveTab] = useState<TabId>('arc');
  const [viewMode, setViewMode] = useState<ViewMode>(smoothedPlay ? 'factor-graph' : 'llm');
  const [selectedBeatId, setSelectedBeatId] = useState<string | null>(null);
  const [bibleModalOpen, setBibleModalOpen] = useState(false);
  const [beatModalOpen, setBeatModalOpen] = useState(false);

  const hasSmoothed = smoothedPlay !== null;

  // Build a Map of beat_id -> SmoothedBeat for efficient lookup
  const smoothedBeatsMap = (() => {
    if (!smoothedPlay) return undefined;
    const charData = smoothedPlay.characters[character] ?? smoothedPlay.characters[character.toUpperCase()];
    if (!charData) return undefined;
    const map = new Map<string, SmoothedBeat>();
    for (const sb of charData.beats) {
      map.set(sb.beat_id, sb);
    }
    return map;
  })();

  const smoothedCharData = smoothedPlay
    ? (smoothedPlay.characters[character] ?? smoothedPlay.characters[character.toUpperCase()] ?? null)
    : null;

  const selectedBeat = beats.find((b) => b.id === selectedBeatId) ?? null;
  const selectedBS: BeatState | undefined = selectedBeat
    ? getBeatStateForCharacter(selectedBeat, character)
    : undefined;
  const selectedSmoothedBeat: SmoothedBeat | undefined = selectedBeat && smoothedPlay
    ? getSmoothedBeatForCharacter(smoothedPlay, character, selectedBeat.id)
    : undefined;

  // Diff summary stats
  const changedCount = smoothedBeatsMap
    ? Array.from(smoothedBeatsMap.values()).filter(sb => sb.changed).length
    : 0;
  const totalSmoothed = smoothedBeatsMap ? smoothedBeatsMap.size : 0;
  const meanAffectShift = smoothedCharData?.mean_affect_shift ?? 0;

  const tabs: { id: TabId; label: string }[] = [
    { id: 'arc', label: 'Arc' },
    { id: 'affect', label: 'Affect Space' },
    { id: 'tactics', label: 'Tactics' },
    { id: 'arc-by-scene', label: 'Arc by Scene' },
  ];

  // When a beat is selected from the timeline, open the modal
  const handleSelectBeat = (beatId: string) => {
    setSelectedBeatId(beatId);
    setBeatModalOpen(true);
  };

  return (
    <div style={{ maxWidth: 1200 }}>
      {/* Character header with bible trigger */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <h1
          style={{
            fontFamily: 'var(--md-sys-typescale-display-font)',
            fontSize: 26,
            fontWeight: 400,
            color: 'var(--md-sys-color-on-surface)',
            margin: 0,
          }}
        >
          {character}
        </h1>
        <span style={{ fontSize: 13, color: 'var(--md-sys-color-on-surface-variant)' }}>
          {beats.length} beats
        </span>
        {bible && (
          <button
            onClick={() => setBibleModalOpen(true)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              background: 'var(--md-sys-color-secondary-container)',
              color: 'var(--md-sys-color-on-secondary-container)',
              border: 'none',
              borderRadius: 20,
              padding: '6px 16px',
              fontSize: 13,
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'opacity 0.2s',
            }}
          >
            <BookOpen size={14} />
            Bible
          </button>
        )}
      </div>

      {/* Bible modal */}
      <FullScreenModal
        open={bibleModalOpen}
        onClose={() => setBibleModalOpen(false)}
        title={`${character} - Character Bible`}
      >
        {bible ? (
          <div style={{ maxWidth: 600 }}>
            <CharacterBibleCard bible={bible} />
          </div>
        ) : (
          <p style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14 }}>
            No character bible available for {character}.
          </p>
        )}
      </FullScreenModal>

      {/* Beat detail modal */}
      <FullScreenModal
        open={beatModalOpen}
        onClose={() => setBeatModalOpen(false)}
        title={selectedBeat ? `Beat ${selectedBeat.index ?? selectedBeat.id}` : 'Beat Detail'}
      >
        {selectedBeat && selectedBS && (
          <div style={{ maxWidth: 700 }}>
            {viewMode === 'llm' ? (
              <BeatStateDetail beat={selectedBeat} beatState={selectedBS} playId={playId} />
            ) : (
              <SmoothedBeatStateDetail
                beat={selectedBeat}
                beatState={selectedBS}
                smoothedBeat={selectedSmoothedBeat}
                viewMode={viewMode}
                playId={playId}
              />
            )}
          </div>
        )}
      </FullScreenModal>

      {/* View mode selector (lens) */}
      <ViewModeSelector
        mode={viewMode}
        onModeChange={setViewMode}
        disabled={!hasSmoothed}
      />

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

      {/* Diff summary card */}
      {viewMode === 'diff' && smoothedCharData && (
        <div
          className="m3-card fade-in"
          style={{
            marginBottom: 16,
            background: 'var(--md-sys-color-surface-container-high)',
            display: 'flex',
            gap: 24,
            flexWrap: 'wrap',
            alignItems: 'center',
          }}
        >
          <div>
            <p style={{ margin: 0, fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.5px' }}>
              Beats changed
            </p>
            <p style={{ margin: '2px 0 0', fontSize: 20, fontWeight: 500, color: changedCount > 0 ? '#f28b82' : '#81c995' }}>
              {changedCount} <span style={{ fontSize: 13, color: 'var(--md-sys-color-on-surface-variant)', fontWeight: 400 }}>of {totalSmoothed}</span>
            </p>
          </div>
          <div>
            <p style={{ margin: 0, fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.5px' }}>
              Mean affect shift
            </p>
            <p style={{ margin: '2px 0 0', fontSize: 20, fontWeight: 500, color: 'var(--md-sys-color-on-surface)' }}>
              {meanAffectShift.toFixed(3)}
            </p>
          </div>
          <div>
            <p style={{ margin: 0, fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.5px' }}>
              Tactic changes
            </p>
            <p style={{ margin: '2px 0 0', fontSize: 20, fontWeight: 500, color: 'var(--md-sys-color-on-surface)' }}>
              {smoothedCharData.num_tactic_changes}
            </p>
          </div>
        </div>
      )}

      {/* Tab content */}
      {activeTab === 'arc' && (
        <div>
          {beats.length === 0 ? (
            <div style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14 }}>
              No beats found for {character}.
            </div>
          ) : (
            <BeatTimeline
              beats={beats}
              character={character}
              selectedBeatId={selectedBeatId}
              onSelectBeat={handleSelectBeat}
              smoothedBeats={smoothedBeatsMap}
              viewMode={viewMode}
            />
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

  // Group entries by act number
  const actGroups: { act: string; scenes: { sceneKey: string; sceneNum: string; arcNote: string; linkHref: string | null }[] }[] = [];
  const actMap = new Map<string, typeof actGroups[number]>();

  for (const [sceneKey, arcNote] of entries) {
    const match = sceneKey.match(/[Aa]?(\d+)[Ss\-_](\d+)/);
    const actNum = match ? match[1] : '?';
    const sceneNum = match ? match[2] : sceneKey;
    const linkHref = match ? `/plays/${playId}/scenes/${match[1]}/${match[2]}` : null;

    if (!actMap.has(actNum)) {
      const group = { act: actNum, scenes: [] };
      actMap.set(actNum, group);
      actGroups.push(group);
    }
    actMap.get(actNum)!.scenes.push({ sceneKey, sceneNum, arcNote, linkHref });
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {actGroups.map((group, gi) => (
        <div key={group.act}>
          {/* Act heading with separator */}
          {gi > 0 && (
            <hr style={{
              border: 'none',
              borderTop: '1px solid var(--md-sys-color-outline-variant)',
              margin: '20px 0 16px',
            }} />
          )}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            marginBottom: 12,
          }}>
            <div style={{
              background: 'var(--md-sys-color-primary-container)',
              color: 'var(--md-sys-color-on-primary-container)',
              borderRadius: 8,
              padding: '4px 12px',
              fontSize: 12,
              fontWeight: 700,
              letterSpacing: '0.5px',
            }}>
              ACT {group.act}
            </div>
            <div style={{
              flex: 1,
              height: 1,
              background: 'var(--md-sys-color-outline-variant)',
            }} />
          </div>

          {/* Scenes within this act */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, paddingLeft: 4 }}>
            {group.scenes.map(({ sceneKey, sceneNum, arcNote, linkHref }) => (
              <div key={sceneKey} className="m3-card fade-in" style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <div style={{
                  minWidth: 40,
                  background: 'var(--md-sys-color-secondary-container)',
                  borderRadius: 6,
                  padding: '4px 8px',
                  textAlign: 'center',
                  flexShrink: 0,
                }}>
                  {linkHref ? (
                    <a
                      href={linkHref}
                      style={{
                        fontSize: 13,
                        fontWeight: 600,
                        color: 'var(--md-sys-color-on-secondary-container)',
                        textDecoration: 'none',
                      }}
                    >
                      {sceneNum}
                    </a>
                  ) : (
                    <span style={{
                      fontSize: 13,
                      fontWeight: 600,
                      color: 'var(--md-sys-color-on-secondary-container)',
                    }}>
                      {sceneNum}
                    </span>
                  )}
                </div>
                <p style={{
                  margin: 0,
                  fontSize: 13,
                  color: 'var(--md-sys-color-on-surface)',
                  lineHeight: 1.6,
                  wordWrap: 'break-word',
                  overflowWrap: 'break-word',
                  minWidth: 0,
                }}>
                  {arcNote}
                </p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
