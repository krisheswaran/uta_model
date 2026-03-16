'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { X } from 'lucide-react';
import type { Play, Beat, Utterance, BeatState } from '@/lib/types';

// beats.json shape: { [sceneId: string]: number[] }
// The array contains utterance indices where each beat *starts*.
// Beat k spans utterances [indices[k], indices[k+1]) — the last beat runs to the scene end.
type BeatsData = Record<string, number[]>;

interface SceneSegmentation {
  sceneId: string;
  label: string;
  beatCount: number;
  beatLengths: number[];
  totalUtterances: number;
}

interface SelectedBeat {
  sceneId: string;
  sceneLabel: string;
  beatIndex: number;     // 0-based index into the scene's beats array
  beatLength: number;    // utterance count
  beat: Beat | null;     // null if that scene isn't in bibles yet
}

interface Props {
  playId: string;
  play: Play | null;
}

export default function BeatSegmentationChart({ playId, play }: Props) {
  const [data, setData] = useState<BeatsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SelectedBeat | null>(null);

  useEffect(() => {
    fetch(`/data/beats/${playId}_beats.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<BeatsData>;
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, [playId]);

  const handleBeatClick = useCallback(
    (sceneId: string, sceneLabel: string, beatIndex: number, beatLength: number) => {
      const beat = play ? findBeat(play, sceneId, beatIndex) : null;
      setSelected({ sceneId, sceneLabel, beatIndex, beatLength, beat });
    },
    [play],
  );

  if (error) return null;
  if (!data) return (
    <div style={{ height: 80, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--md-sys-color-on-surface-variant)', fontSize: 13 }}>
      Loading segmentation…
    </div>
  );

  const scenes = buildSceneSegmentations(data);
  if (scenes.length === 0) return null;

  const allLengths = scenes.flatMap((s) => s.beatLengths);

  return (
    <>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        {/* Per-scene rhythm strips */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {scenes.map((scene) => (
            <SceneRhythmRow
              key={scene.sceneId}
              scene={scene}
              onBeatClick={handleBeatClick}
            />
          ))}
        </div>

        {/* Beat-length histogram */}
        <BeatLengthHistogram lengths={allLengths} />

        {/* Legend */}
        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', fontSize: 12, color: 'var(--md-sys-color-on-surface-variant)' }}>
          <span>Tap a beat to inspect its utterances · Width ∝ utterance count</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ display: 'inline-block', width: 28, height: 10, background: 'var(--md-sys-color-primary)', opacity: 0.7, borderRadius: 2 }} />
            odd beats
            <span style={{ display: 'inline-block', width: 28, height: 10, background: 'var(--md-sys-color-tertiary)', opacity: 0.5, borderRadius: 2, marginLeft: 8 }} />
            even beats
          </span>
        </div>
      </div>

      {/* Beat detail modal */}
      {selected && (
        <BeatModal selected={selected} onClose={() => setSelected(null)} />
      )}
    </>
  );
}

// ── Scene rhythm strip ─────────────────────────────────────────────────────

interface SceneRhythmRowProps {
  scene: SceneSegmentation;
  onBeatClick: (sceneId: string, sceneLabel: string, beatIndex: number, beatLength: number) => void;
}

function SceneRhythmRow({ scene, onBeatClick }: SceneRhythmRowProps) {
  const [hovered, setHovered] = useState<number | null>(null);
  const total = scene.totalUtterances;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      {/* Scene label */}
      <div style={{
        minWidth: 80,
        fontSize: 11,
        color: 'var(--md-sys-color-on-surface-variant)',
        textAlign: 'right',
        lineHeight: 1.3,
        flexShrink: 0,
      }}>
        {scene.label}
        <div style={{ fontSize: 10, opacity: 0.7 }}>{scene.beatCount} beats</div>
      </div>

      {/* Rhythm strip */}
      <div style={{
        flex: 1,
        height: 28,
        display: 'flex',
        borderRadius: 4,
        overflow: 'visible',
        position: 'relative',
        cursor: 'pointer',
      }}>
        {scene.beatLengths.map((len, i) => {
          const widthPct = (len / total) * 100;
          const isOdd = i % 2 === 0;
          const isHovered = hovered === i;
          return (
            <div
              key={i}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => onBeatClick(scene.sceneId, scene.label, i, len)}
              title={`Beat ${i + 1}: ${len} utterance${len !== 1 ? 's' : ''} — click to inspect`}
              style={{
                width: `${widthPct}%`,
                height: '100%',
                background: isOdd
                  ? 'var(--md-sys-color-primary)'
                  : 'var(--md-sys-color-tertiary)',
                opacity: isHovered ? 1 : (isOdd ? 0.65 : 0.45),
                borderRight: i < scene.beatLengths.length - 1
                  ? '1px solid var(--md-sys-color-background)'
                  : 'none',
                transition: 'opacity 0.12s',
                flexShrink: 0,
                transform: isHovered ? 'scaleY(1.15)' : 'scaleY(1)',
                transformOrigin: 'bottom',
              }}
            />
          );
        })}
      </div>

      {/* Utterance count */}
      <div style={{ minWidth: 36, fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', textAlign: 'left', flexShrink: 0 }}>
        {total}u
      </div>
    </div>
  );
}

// ── Beat detail modal ──────────────────────────────────────────────────────

function BeatModal({ selected, onClose }: { selected: SelectedBeat; onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement>(null);

  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const { beat, sceneLabel, beatIndex, beatLength } = selected;

  return (
    /* Backdrop — click outside panel to close */
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0,
        background: 'rgba(0,0,0,0.55)',
        backdropFilter: 'blur(2px)',
        zIndex: 200,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
      }}
    >
      {/* Panel — stop propagation so clicking inside doesn't close */}
      <div
        ref={panelRef}
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--md-sys-color-surface-container)',
          border: '1px solid var(--md-sys-color-outline-variant)',
          borderRadius: 16,
          width: '100%',
          maxWidth: 560,
          maxHeight: '80vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          padding: '16px 20px 12px',
          borderBottom: '1px solid var(--md-sys-color-outline-variant)',
          flexShrink: 0,
        }}>
          <div>
            <div style={{
              fontFamily: 'var(--md-sys-typescale-display-font)',
              fontSize: 17,
              fontWeight: 500,
              color: 'var(--md-sys-color-on-surface)',
              lineHeight: 1.3,
            }}>
              {sceneLabel} — Beat {beatIndex + 1}
            </div>
            <div style={{ fontSize: 12, color: 'var(--md-sys-color-on-surface-variant)', marginTop: 2 }}>
              {beatLength} utterance{beatLength !== 1 ? 's' : ''}
              {beat?.beat_states?.[0]?.tactic_state && (
                <span style={{
                  marginLeft: 10,
                  background: 'var(--md-sys-color-primary-container)',
                  color: 'var(--md-sys-color-on-primary-container)',
                  padding: '1px 8px',
                  borderRadius: 6,
                  fontSize: 11,
                  fontWeight: 500,
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                }}>
                  {beat.beat_states[0].tactic_state}
                </span>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--md-sys-color-on-surface-variant)',
              padding: 4,
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* BeatState summary (if available) */}
        {beat?.beat_states && beat.beat_states.length > 0 && (
          <div style={{
            padding: '10px 20px',
            borderBottom: '1px solid var(--md-sys-color-outline-variant)',
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
            flexShrink: 0,
          }}>
            {beat.beat_states.map((bs) => (
              <BeatStateSummaryRow key={bs.character} bs={bs} />
            ))}
          </div>
        )}

        {/* Utterances */}
        <div style={{ overflowY: 'auto', padding: '12px 20px 20px', flex: 1 }}>
          {beat ? (
            beat.utterances.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {beat.utterances.map((u, i) => (
                  <UtteranceRow key={u.id ?? i} u={u} />
                ))}
              </div>
            ) : (
              <EmptyNote>No utterances in this beat.</EmptyNote>
            )
          ) : (
            <EmptyNote>
              Utterances for this act haven't been extracted yet.<br />
              Run the analysis pipeline on this act to see content here.
            </EmptyNote>
          )}
        </div>
      </div>
    </div>
  );
}

// ── BeatState one-liner per character ─────────────────────────────────────

function BeatStateSummaryRow({ bs }: { bs: BeatState }) {
  const v = bs.affect_state?.valence ?? 0;
  const valenceColor = v > 0.15 ? '#81c995' : v < -0.15 ? '#f28b82' : '#8ab4f8';
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, fontSize: 12 }}>
      <span style={{
        minWidth: 90,
        fontWeight: 600,
        color: 'var(--md-sys-color-on-surface)',
        fontSize: 11,
        letterSpacing: '0.03em',
      }}>
        {bs.character}
      </span>
      <span style={{
        background: 'var(--md-sys-color-secondary-container)',
        color: 'var(--md-sys-color-on-secondary-container)',
        padding: '1px 7px',
        borderRadius: 5,
        fontSize: 11,
        fontWeight: 500,
      }}>
        {bs.tactic_state}
      </span>
      <span style={{ color: 'var(--md-sys-color-on-surface-variant)', flex: 1, fontStyle: 'italic', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {bs.desire_state}
      </span>
      <span style={{
        width: 8, height: 8, borderRadius: '50%',
        background: valenceColor,
        flexShrink: 0,
        title: `valence ${v.toFixed(2)}`,
      }} />
    </div>
  );
}

// ── Utterance row ──────────────────────────────────────────────────────────

function UtteranceRow({ u }: { u: Utterance }) {
  const isStageDir = !u.speaker || u.speaker === u.speaker.toLowerCase();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {u.stage_direction && (
        <div style={{
          fontSize: 12,
          fontStyle: 'italic',
          color: 'var(--md-sys-color-on-surface-variant)',
          paddingLeft: 4,
          opacity: 0.8,
        }}>
          [{u.stage_direction}]
        </div>
      )}
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
        <span style={{
          minWidth: 90,
          fontSize: 12,
          fontWeight: 600,
          color: 'var(--md-sys-color-primary)',
          letterSpacing: '0.02em',
          flexShrink: 0,
          lineHeight: 1.5,
        }}>
          {u.speaker}
        </span>
        <span style={{
          fontSize: 14,
          color: 'var(--md-sys-color-on-surface)',
          lineHeight: 1.6,
        }}>
          {u.text}
        </span>
      </div>
    </div>
  );
}

function EmptyNote({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      textAlign: 'center',
      padding: '32px 16px',
      color: 'var(--md-sys-color-on-surface-variant)',
      fontSize: 13,
      lineHeight: 1.6,
    }}>
      {children}
    </div>
  );
}

// ── Beat-length histogram ──────────────────────────────────────────────────

function BeatLengthHistogram({ lengths }: { lengths: number[] }) {
  const buckets: { label: string; min: number; max: number }[] = [
    { label: '1', min: 1, max: 1 },
    { label: '2', min: 2, max: 2 },
    { label: '3', min: 3, max: 3 },
    { label: '4', min: 4, max: 4 },
    { label: '5–7', min: 5, max: 7 },
    { label: '8–12', min: 8, max: 12 },
    { label: '13+', min: 13, max: Infinity },
  ];

  const counts = buckets.map(({ min, max }) =>
    lengths.filter((l) => l >= min && l <= max).length
  );
  const maxCount = Math.max(...counts, 1);

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 12, color: 'var(--md-sys-color-on-surface-variant)', marginBottom: 8 }}>
        Beat length distribution (utterances per beat, n={lengths.length})
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: 64 }}>
        {buckets.map((b, i) => {
          const heightPct = (counts[i] / maxCount) * 100;
          return (
            <div
              key={b.label}
              style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, gap: 4 }}
              title={`${b.label} utterances: ${counts[i]} beat${counts[i] !== 1 ? 's' : ''}`}
            >
              <div style={{ fontSize: 10, color: 'var(--md-sys-color-on-surface-variant)', opacity: counts[i] > 0 ? 1 : 0.3 }}>
                {counts[i] > 0 ? counts[i] : ''}
              </div>
              <div style={{
                width: '100%',
                height: `${heightPct}%`,
                minHeight: counts[i] > 0 ? 3 : 0,
                background: 'var(--md-sys-color-primary)',
                opacity: 0.75,
                borderRadius: '3px 3px 0 0',
                transition: 'height 0.3s',
              }} />
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
        {buckets.map((b) => (
          <div key={b.label} style={{ flex: 1, textAlign: 'center', fontSize: 10, color: 'var(--md-sys-color-on-surface-variant)' }}>
            {b.label}
          </div>
        ))}
      </div>
      <div style={{ textAlign: 'center', fontSize: 10, color: 'var(--md-sys-color-on-surface-variant)', marginTop: 2 }}>
        utterances / beat
      </div>
    </div>
  );
}

// ── Helpers ────────────────────────────────────────────────────────────────

function findBeat(play: Play, sceneId: string, beatIndex: number): Beat | null {
  const parts = sceneId.split('_');
  const sceneNum = parseInt(parts[parts.length - 1]);
  const actNum = parseInt(parts[parts.length - 2]);
  const act = play.acts.find((a) => a.number === actNum);
  if (!act) return null;
  const scene = act.scenes.find((s) => s.scene === sceneNum);
  if (!scene) return null;
  return scene.beats[beatIndex] ?? null;
}

function buildSceneSegmentations(data: BeatsData): SceneSegmentation[] {
  return Object.entries(data)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([sceneId, indices]) => {
      const parts = sceneId.split('_');
      const scene = parts[parts.length - 1];
      const act = parts[parts.length - 2];
      const label = `Act ${act}, Sc ${scene}`;

      const beatLengths: number[] = [];
      for (let i = 0; i < indices.length - 1; i++) {
        beatLengths.push(indices[i + 1] - indices[i]);
      }
      // Last beat length: use median of the rest as estimate
      const median = beatLengths.length > 0
        ? beatLengths.slice().sort((a, b) => a - b)[Math.floor(beatLengths.length / 2)]
        : 1;
      beatLengths.push(median);

      const totalUtterances = beatLengths.reduce((a, b) => a + b, 0);
      return { sceneId, label, beatCount: indices.length, beatLengths, totalUtterances };
    });
}
