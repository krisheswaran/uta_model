'use client';

import { useEffect, useState } from 'react';
import { ArrowLeft, ChevronDown, ChevronRight } from 'lucide-react';
import NavRail from './NavRail';
import BeatStateDetail from './BeatStateDetail';
import type { Play, Beat, BeatState, Utterance } from '@/lib/types';
import { getScene, getSceneBible, confidenceClass, confidenceLabel } from '@/lib/data';

interface Props {
  playId: string;
  act: number;
  scene: number;
}

export default function SceneView({ playId, act, scene }: Props) {
  const [play, setPlay] = useState<Play | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  const sceneData = play ? getScene(play, act, scene) : undefined;
  const sceneBible = play ? getSceneBible(play, act, scene) : undefined;

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--md-sys-color-background)' }}>
      <NavRail playId={playId} act={act} scene={scene} />

      <main
        style={{ flex: 1, padding: '24px 16px' }}
        className="ml-0 mb-[72px] md:ml-[88px] md:mb-0"
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
          <span style={{ color: 'var(--md-sys-color-outline)', fontSize: 14 }}>/</span>
          <span style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14 }}>
            Act {act}, Scene {scene}
          </span>
        </div>

        {loading && <SceneSkeleton />}
        {!loading && error && (
          <div className="m3-card" style={{ color: 'var(--md-sys-color-error)' }}>{error}</div>
        )}

        {!loading && !error && (
          <SceneContent
            play={play!}
            playId={playId}
            act={act}
            scene={scene}
            sceneData={sceneData}
            sceneBible={sceneBible}
          />
        )}
      </main>
    </div>
  );
}

function SceneSkeleton() {
  return (
    <div>
      <div className="skeleton" style={{ height: 120, marginBottom: 16 }} />
      <div className="skeleton" style={{ height: 200, marginBottom: 16 }} />
      <div className="skeleton" style={{ height: 200 }} />
    </div>
  );
}

function SceneContent({
  play,
  playId,
  act,
  scene,
  sceneData,
  sceneBible,
}: {
  play: Play;
  playId: string;
  act: number;
  scene: number;
  sceneData: ReturnType<typeof getScene>;
  sceneBible: ReturnType<typeof getSceneBible>;
}) {
  if (!sceneData) {
    return (
      <div className="m3-card" style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14 }}>
        Scene {act}.{scene} not found in play data.
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 900 }}>
      {/* Scene header */}
      <header>
        <h1
          style={{
            fontFamily: 'var(--md-sys-typescale-display-font)',
            fontSize: 26,
            fontWeight: 400,
            margin: '0 0 4px',
            color: 'var(--md-sys-color-on-surface)',
          }}
        >
          Act {act}, Scene {scene}
        </h1>
        <p style={{ margin: 0, fontSize: 13, color: 'var(--md-sys-color-on-surface-variant)' }}>
          {sceneData.beats.length} beats
        </p>
      </header>

      {/* Scene Bible card */}
      {sceneBible && (
        <div className="m3-card-elevated fade-in">
          <p className="m3-label-medium" style={{ marginBottom: 12 }}>SCENE BIBLE</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {sceneBible.dramatic_pressure && (
              <div>
                <p className="m3-label-medium" style={{ marginBottom: 4, fontSize: 11 }}>DRAMATIC PRESSURE</p>
                <p style={{ margin: 0, fontSize: 14, color: 'var(--md-sys-color-on-surface)', lineHeight: 1.6 }}>
                  {sceneBible.dramatic_pressure}
                </p>
              </div>
            )}
            {sceneBible.what_changes && (
              <div>
                <p className="m3-label-medium" style={{ marginBottom: 4, fontSize: 11 }}>WHAT CHANGES</p>
                <p style={{ margin: 0, fontSize: 14, color: 'var(--md-sys-color-on-surface)', lineHeight: 1.6 }}>
                  {sceneBible.what_changes}
                </p>
              </div>
            )}
            {sceneBible.hidden_tensions && (
              <div>
                <p className="m3-label-medium" style={{ marginBottom: 4, fontSize: 11 }}>HIDDEN TENSIONS</p>
                <p style={{ margin: 0, fontSize: 14, color: 'var(--md-sys-color-tertiary)', lineHeight: 1.6 }}>
                  {sceneBible.hidden_tensions}
                </p>
              </div>
            )}
            {sceneBible.beat_map && (
              <div>
                <p className="m3-label-medium" style={{ marginBottom: 4, fontSize: 11 }}>BEAT MAP</p>
                <p
                  style={{
                    margin: 0,
                    fontSize: 13,
                    color: 'var(--md-sys-color-on-surface-variant)',
                    lineHeight: 1.6,
                    fontStyle: 'italic',
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  {sceneBible.beat_map}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Beat cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {sceneData.beats.map((beat) => (
          <BeatCard key={beat.id} beat={beat} playId={playId} />
        ))}
      </div>
    </div>
  );
}

function BeatCard({ beat, playId }: { beat: Beat; playId: string }) {
  const [expanded, setExpanded] = useState(false);

  const characters = beat.characters_present ?? [
    ...new Set(beat.beat_states.map((bs) => bs.character)),
  ];

  return (
    <div className="m3-card fade-in" style={{ border: '1px solid var(--md-sys-color-outline-variant)' }}>
      {/* Beat header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          marginBottom: 12,
          cursor: 'pointer',
          flexWrap: 'wrap',
        }}
        onClick={() => setExpanded((v) => !v)}
      >
        <div
          style={{
            background: 'var(--md-sys-color-primary-container)',
            color: 'var(--md-sys-color-on-primary-container)',
            borderRadius: 6,
            padding: '3px 10px',
            fontSize: 12,
            fontWeight: 600,
            flexShrink: 0,
          }}
        >
          {beat.index != null ? `Beat ${beat.index}` : beat.id}
        </div>

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', flex: 1 }}>
          {characters.map((c) => (
            <a
              key={c}
              href={`/plays/${playId}/characters/${encodeURIComponent(c)}`}
              onClick={(e) => e.stopPropagation()}
              style={{ textDecoration: 'none' }}
            >
              <span className="m3-chip" style={{ fontSize: 11, padding: '2px 8px' }}>
                {c}
              </span>
            </a>
          ))}
        </div>

        <div style={{ marginLeft: 'auto', color: 'var(--md-sys-color-on-surface-variant)' }}>
          {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </div>
      </div>

      {/* Beat summary */}
      {beat.beat_summary && (
        <p
          style={{
            margin: '0 0 12px',
            fontSize: 13,
            color: 'var(--md-sys-color-on-surface-variant)',
            lineHeight: 1.5,
            fontStyle: 'italic',
          }}
        >
          {beat.beat_summary}
        </p>
      )}

      {/* Utterances */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
        {beat.utterances.map((u) => (
          <UtteranceRow key={u.id} utterance={u} />
        ))}
      </div>

      {/* BeatState details (expandable) */}
      {expanded && beat.beat_states.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 12 }}>
          <hr className="m3-divider" />
          {beat.beat_states.map((bs) => (
            <BeatStateCompact key={bs.character} beat={beat} beatState={bs} />
          ))}
        </div>
      )}
    </div>
  );
}

function UtteranceRow({ utterance: u }: { utterance: Utterance }) {
  return (
    <div>
      {u.stage_direction && (
        <p
          style={{
            margin: '0 0 4px',
            fontSize: 12,
            color: 'var(--md-sys-color-on-surface-variant)',
            fontStyle: 'italic',
          }}
        >
          [{u.stage_direction}]
        </p>
      )}
      {u.text && (
        <div style={{ display: 'flex', gap: 8 }}>
          <span
            style={{
              flexShrink: 0,
              fontSize: 12,
              fontWeight: 700,
              color: 'var(--md-sys-color-primary)',
              minWidth: 80,
              paddingTop: 1,
            }}
          >
            {u.speaker}
          </span>
          <p
            style={{
              margin: 0,
              fontSize: 14,
              color: 'var(--md-sys-color-on-surface)',
              lineHeight: 1.6,
              fontFamily: 'var(--md-sys-typescale-display-font)',
            }}
          >
            {u.text}
          </p>
        </div>
      )}
    </div>
  );
}

function BeatStateCompact({
  beat,
  beatState: bs,
}: {
  beat: Beat;
  beatState: BeatState;
}) {
  const [showFull, setShowFull] = useState(false);

  if (showFull) {
    return (
      <div>
        <button
          onClick={() => setShowFull(false)}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: 'var(--md-sys-color-on-surface-variant)',
            fontSize: 12,
            marginBottom: 8,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            padding: 0,
          }}
        >
          <ChevronDown size={14} /> Collapse
        </button>
        <BeatStateDetail beat={beat} beatState={bs} />
      </div>
    );
  }

  return (
    <div
      style={{
        background: 'var(--md-sys-color-surface-container-high)',
        borderRadius: 8,
        padding: '10px 12px',
        display: 'flex',
        gap: 10,
        alignItems: 'flex-start',
        flexWrap: 'wrap',
        cursor: 'pointer',
      }}
      onClick={() => setShowFull(true)}
    >
      {/* Character + confidence */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 120 }}>
        <span
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: 'var(--md-sys-color-primary)',
          }}
        >
          {bs.character}
        </span>
        <span className={`confidence-badge ${confidenceClass(bs.confidence)}`} style={{ width: 24, height: 24, fontSize: 9 }}>
          {confidenceLabel(bs.confidence)}
        </span>
      </div>

      {/* Desire → Tactic */}
      <div style={{ flex: 1, minWidth: 200 }}>
        <p style={{ margin: '0 0 2px', fontSize: 12, color: 'var(--md-sys-color-on-surface)' }}>
          <span style={{ color: 'var(--md-sys-color-on-surface-variant)' }}>Wants: </span>
          {bs.desire_state}
        </p>
        <p style={{ margin: 0, fontSize: 12, color: 'var(--md-sys-color-on-surface)' }}>
          <span style={{ color: 'var(--md-sys-color-on-surface-variant)' }}>Tactic: </span>
          {bs.tactic_state}
        </p>
        {bs.obstacle && (
          <p style={{ margin: '2px 0 0', fontSize: 12, color: 'var(--md-sys-color-tertiary)' }}>
            <span style={{ color: 'var(--md-sys-color-on-surface-variant)' }}>Obstacle: </span>
            {bs.obstacle}
          </p>
        )}
      </div>

      {/* Mini affect bars */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <MiniAffectBar label="V" value={bs.affect_state.valence} />
        <MiniAffectBar label="A" value={bs.affect_state.arousal} />
      </div>

      {/* Low confidence warning */}
      {bs.confidence < 0.7 && bs.alternative_hypothesis && (
        <div
          style={{
            width: '100%',
            padding: '4px 8px',
            background: 'var(--md-sys-color-tertiary-container)',
            borderRadius: 4,
            marginTop: 4,
          }}
          onClick={(e) => {
            e.stopPropagation();
            setShowFull(true);
          }}
        >
          <p style={{ margin: 0, fontSize: 11, color: 'var(--md-sys-color-on-tertiary-container)' }}>
            Alt: {bs.alternative_hypothesis}
          </p>
        </div>
      )}

      <ChevronRight size={14} style={{ color: 'var(--md-sys-color-on-surface-variant)', marginLeft: 'auto' }} />
    </div>
  );
}

function MiniAffectBar({ label, value }: { label: string; value: number }) {
  const pos = value >= 0;
  const width = Math.abs(value) * 24;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={{ fontSize: 9, color: 'var(--md-sys-color-on-surface-variant)', fontWeight: 600 }}>{label}</span>
      <div style={{ position: 'relative', width: 48, height: 6, background: 'var(--md-sys-color-surface-container-highest)', borderRadius: 3, overflow: 'hidden' }}>
        {pos ? (
          <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width, background: 'var(--affect-positive)', borderRadius: '0 3px 3px 0' }} />
        ) : (
          <div style={{ position: 'absolute', right: '50%', top: 0, bottom: 0, width, background: 'var(--affect-negative)', borderRadius: '3px 0 0 3px' }} />
        )}
        <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'var(--md-sys-color-outline)' }} />
      </div>
    </div>
  );
}
