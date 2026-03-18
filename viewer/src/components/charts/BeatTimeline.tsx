'use client';

import { useRef } from 'react';
import type { Beat, BeatState, SmoothedBeat } from '@/lib/types';
import { getBeatStateForCharacter, valenceColor } from '@/lib/data';
import type { ViewMode } from '../ViewModeSelector';

interface Props {
  beats: Beat[];
  character: string;
  selectedBeatId: string | null;
  onSelectBeat: (beatId: string) => void;
  smoothedBeats?: Map<string, SmoothedBeat>;
  viewMode?: ViewMode;
}

const BEAT_WIDTH = 64;
const BEAT_HEIGHT = 120;

export default function BeatTimeline({ beats, character, selectedBeatId, onSelectBeat, smoothedBeats, viewMode = 'llm' }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  if (beats.length === 0) {
    return (
      <div style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14, padding: 16 }}>
        No beats available for this character.
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      style={{
        overflowX: 'auto',
        overflowY: 'visible',
        paddingBottom: 8,
        cursor: 'grab',
      }}
    >
      <svg
        width={beats.length * BEAT_WIDTH + 2}
        height={BEAT_HEIGHT + 20}
        style={{ display: 'block' }}
      >
        {beats.map((beat, i) => {
          const bs: BeatState | undefined = getBeatStateForCharacter(beat, character);
          const sb = smoothedBeats?.get(beat.id);
          const useSmoothed = (viewMode === 'factor-graph' || viewMode === 'diff') && sb;

          const valence = useSmoothed
            ? (sb.affect_trans_mean[0] ?? 0)
            : (bs?.affect_state.valence ?? 0);
          const arousal = useSmoothed
            ? sb.arousal
            : (bs?.affect_state.arousal ?? 0);
          const confidence = useSmoothed
            ? sb.smoothed_tactic_prob
            : (bs?.confidence ?? 0.5);
          const tactic = useSmoothed
            ? sb.smoothed_tactic
            : (bs?.tactic_state ?? '');

          // Background color from valence
          const bgColor = valenceColor(valence);

          // Arousal bar height
          const arousalNorm = useSmoothed ? arousal : (arousal + 1) / 2;
          const arousalH = Math.round(Math.max(0, Math.min(1, arousalNorm)) * 8);
          const arousalColor =
            arousalNorm > 0.6 ? '#f28b82' : arousalNorm < 0.3 ? '#8ab4f8' : '#fbbc04';

          const isSelected = beat.id === selectedBeatId;
          const isChanged = sb?.changed ?? false;
          const x = i * BEAT_WIDTH;

          return (
            <g
              key={beat.id}
              onClick={() => onSelectBeat(beat.id)}
              style={{ cursor: 'pointer' }}
            >
              {/* Background rect */}
              <rect
                x={x + 1}
                y={0}
                width={BEAT_WIDTH - 2}
                height={BEAT_HEIGHT}
                fill={bgColor}
                opacity={0.3}
                rx={4}
              />

              {/* Base surface */}
              <rect
                x={x + 1}
                y={0}
                width={BEAT_WIDTH - 2}
                height={BEAT_HEIGHT}
                fill="var(--md-sys-color-surface-container)"
                opacity={0.85}
                rx={4}
              />

              {/* Valence overlay */}
              <rect
                x={x + 1}
                y={0}
                width={BEAT_WIDTH - 2}
                height={BEAT_HEIGHT}
                fill={bgColor}
                opacity={0.2}
                rx={4}
              />

              {/* Diff mode: changed beat highlight */}
              {viewMode === 'diff' && isChanged && (
                <rect
                  x={x + 1}
                  y={0}
                  width={BEAT_WIDTH - 2}
                  height={BEAT_HEIGHT}
                  fill="none"
                  stroke="#f28b82"
                  strokeWidth={2}
                  strokeDasharray="4,2"
                  rx={4}
                />
              )}

              {/* Selected highlight */}
              {isSelected && (
                <rect
                  x={x + 1}
                  y={0}
                  width={BEAT_WIDTH - 2}
                  height={BEAT_HEIGHT}
                  fill="none"
                  stroke="var(--md-sys-color-primary)"
                  strokeWidth={2}
                  rx={4}
                />
              )}

              {/* Beat index */}
              <text
                x={x + BEAT_WIDTH / 2}
                y={14}
                textAnchor="middle"
                fontSize={10}
                fill="var(--md-sys-color-on-surface-variant)"
                fontWeight={500}
              >
                {beat.index ?? i + 1}
              </text>

              {/* Diff mode: changed indicator dot at top-right */}
              {viewMode === 'diff' && isChanged && (
                <circle
                  cx={x + BEAT_WIDTH - 8}
                  cy={8}
                  r={4}
                  fill="#f28b82"
                  opacity={0.9}
                />
              )}

              {/* Tactic label (rotated) */}
              {tactic && (
                <text
                  x={x + BEAT_WIDTH / 2}
                  y={BEAT_HEIGHT / 2 + 4}
                  textAnchor="middle"
                  fontSize={10}
                  fill="var(--md-sys-color-on-surface)"
                  opacity={0.4 + confidence * 0.6}
                  transform={`rotate(-45, ${x + BEAT_WIDTH / 2}, ${BEAT_HEIGHT / 2 + 4})`}
                  style={{ userSelect: 'none' }}
                >
                  {tactic.length > 12 ? tactic.slice(0, 11) + '...' : tactic}
                </text>
              )}

              {/* Confidence dot at bottom */}
              <circle
                cx={x + BEAT_WIDTH / 2}
                cy={BEAT_HEIGHT - 12}
                r={3}
                fill={
                  confidence >= 0.7
                    ? '#81c995'
                    : confidence >= 0.4
                    ? '#fbbc04'
                    : '#f28b82'
                }
                opacity={0.9}
              />

              {/* Arousal bar at very bottom */}
              {arousalH > 0 && (
                <rect
                  x={x + 2}
                  y={BEAT_HEIGHT - arousalH}
                  width={BEAT_WIDTH - 4}
                  height={arousalH}
                  fill={arousalColor}
                  opacity={0.5}
                  rx={2}
                />
              )}

              {/* Divider */}
              <line
                x1={x + BEAT_WIDTH - 1}
                y1={4}
                x2={x + BEAT_WIDTH - 1}
                y2={BEAT_HEIGHT - 4}
                stroke="var(--md-sys-color-outline-variant)"
                strokeWidth={0.5}
                opacity={0.5}
              />
            </g>
          );
        })}

        {/* Beat number axis at bottom */}
        {beats.map((beat, i) => (
          <text
            key={beat.id + '-label'}
            x={i * BEAT_WIDTH + BEAT_WIDTH / 2}
            y={BEAT_HEIGHT + 14}
            textAnchor="middle"
            fontSize={9}
            fill="var(--md-sys-color-on-surface-variant)"
            opacity={0.5}
          >
            {beat.act}.{beat.scene}
          </text>
        ))}
      </svg>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, marginTop: 8, flexWrap: 'wrap', paddingLeft: 4 }}>
        <LegendSwatch color="var(--affect-positive)" label={viewMode === 'factor-graph' ? 'Positive eigenspace' : 'Positive valence'} />
        <LegendSwatch color="var(--affect-negative)" label={viewMode === 'factor-graph' ? 'Negative eigenspace' : 'Negative valence'} />
        <span style={{ fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: '#81c995' }} />
          {viewMode === 'factor-graph' ? 'High tactic prob' : 'High confidence'}
        </span>
        <span style={{ fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 8, height: 8, background: '#8ab4f8', borderRadius: 2 }} />
          Arousal (bottom bar)
        </span>
        {viewMode === 'diff' && (
          <span style={{ fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: '#f28b82' }} />
            Changed by smoother
          </span>
        )}
      </div>
    </div>
  );
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)' }}>
      <span style={{ display: 'inline-block', width: 14, height: 14, background: color, borderRadius: 3, opacity: 0.7 }} />
      {label}
    </span>
  );
}
