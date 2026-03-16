'use client';

import { useRef } from 'react';
import type { Beat, BeatState } from '@/lib/types';
import { getBeatStateForCharacter, valenceColor } from '@/lib/data';

interface Props {
  beats: Beat[];
  character: string;
  selectedBeatId: string | null;
  onSelectBeat: (beatId: string) => void;
}

const BEAT_WIDTH = 64;
const BEAT_HEIGHT = 120;

export default function BeatTimeline({ beats, character, selectedBeatId, onSelectBeat }: Props) {
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
          const valence = bs?.affect_state.valence ?? 0;
          const arousal = bs?.affect_state.arousal ?? 0;
          const confidence = bs?.confidence ?? 0.5;
          const tactic = bs?.tactic_state ?? '';

          // Background color from valence
          const bgColor = valenceColor(valence);

          // Arousal bar height: 0–8px at bottom of cell
          const arousalH = Math.round(((arousal + 1) / 2) * 8);
          const arousalColor =
            arousal > 0.3 ? '#f28b82' : arousal < -0.3 ? '#8ab4f8' : '#fbbc04';

          const isSelected = beat.id === selectedBeatId;
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
                  {tactic.length > 12 ? tactic.slice(0, 11) + '…' : tactic}
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
        <LegendSwatch color="var(--affect-positive)" label="Positive valence" />
        <LegendSwatch color="var(--affect-negative)" label="Negative valence" />
        <span style={{ fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: '#81c995' }} />
          High confidence
        </span>
        <span style={{ fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 8, height: 8, background: '#8ab4f8', borderRadius: 2 }} />
          Arousal (bottom bar)
        </span>
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
