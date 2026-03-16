'use client';

import { useState } from 'react';
import type { Beat, BeatState } from '@/lib/types';
import { getBeatStateForCharacter, beatProgressColor } from '@/lib/data';

interface Props {
  beats: Beat[];
  character: string;
}

interface PlotPoint {
  x: number;
  y: number;
  beatId: string;
  beatIndex: number;
  desireState: string;
  tacticState: string;
  confidence: number;
  color: string;
}

export default function AffectTrajectory({ beats, character }: Props) {
  const [hovered, setHovered] = useState<PlotPoint | null>(null);

  const SVG_SIZE = 400;
  const CENTER = SVG_SIZE / 2;
  const RADIUS = 170;

  // Build data points
  const points: PlotPoint[] = [];
  beats.forEach((beat, i) => {
    const bs: BeatState | undefined = getBeatStateForCharacter(beat, character);
    if (!bs) return;
    const { valence, arousal } = bs.affect_state;
    // valence maps to x-axis: -1 = left, +1 = right
    // arousal maps to y-axis: -1 = bottom, +1 = top (SVG y is inverted)
    const x = CENTER + valence * RADIUS;
    const y = CENTER - arousal * RADIUS;
    points.push({
      x,
      y,
      beatId: beat.id,
      beatIndex: i,
      desireState: bs.desire_state,
      tacticState: bs.tactic_state,
      confidence: bs.confidence,
      color: beatProgressColor(i, beats.length),
    });
  });

  const polylinePoints = points.map((p) => `${p.x},${p.y}`).join(' ');

  return (
    <div style={{ position: 'relative', width: '100%', maxWidth: 460 }}>
      <svg
        viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`}
        style={{ width: '100%', height: 'auto', display: 'block' }}
        role="img"
        aria-label="Russell Circumplex affect trajectory"
      >
        {/* Quadrant backgrounds */}
        {/* Q1: Excited (top-right, valence+, arousal+) */}
        <rect x={CENTER} y={0} width={CENTER} height={CENTER} fill="rgba(129,201,149,0.07)" />
        {/* Q2: Tense (top-left, valence-, arousal+) */}
        <rect x={0} y={0} width={CENTER} height={CENTER} fill="rgba(242,139,130,0.07)" />
        {/* Q3: Depressed (bottom-left, valence-, arousal-) */}
        <rect x={0} y={CENTER} width={CENTER} height={CENTER} fill="rgba(100,100,120,0.07)" />
        {/* Q4: Calm (bottom-right, valence+, arousal-) */}
        <rect x={CENTER} y={CENTER} width={CENTER} height={CENTER} fill="rgba(138,180,248,0.07)" />

        {/* Quadrant labels */}
        <text x={CENTER + 8} y={20} fontSize={11} fill="rgba(129,201,149,0.7)" fontWeight={600}>EXCITED</text>
        <text x={8} y={20} fontSize={11} fill="rgba(242,139,130,0.7)" fontWeight={600}>TENSE</text>
        <text x={8} y={SVG_SIZE - 8} fontSize={11} fill="rgba(150,150,180,0.7)" fontWeight={600}>DEPRESSED</text>
        <text x={CENTER + 8} y={SVG_SIZE - 8} fontSize={11} fill="rgba(138,180,248,0.7)" fontWeight={600}>CALM</text>

        {/* Axes */}
        <line x1={0} y1={CENTER} x2={SVG_SIZE} y2={CENTER} stroke="rgba(147,143,153,0.3)" strokeWidth={1} />
        <line x1={CENTER} y1={0} x2={CENTER} y2={SVG_SIZE} stroke="rgba(147,143,153,0.3)" strokeWidth={1} />

        {/* Axis labels */}
        <text x={SVG_SIZE - 4} y={CENTER - 6} fontSize={10} fill="rgba(202,196,208,0.6)" textAnchor="end">POSITIVE →</text>
        <text x={4} y={CENTER - 6} fontSize={10} fill="rgba(202,196,208,0.6)" textAnchor="start">← NEGATIVE</text>
        <text x={CENTER} y={12} fontSize={10} fill="rgba(202,196,208,0.6)" textAnchor="middle">HIGH AROUSAL ↑</text>
        <text x={CENTER} y={SVG_SIZE - 4} fontSize={10} fill="rgba(202,196,208,0.6)" textAnchor="middle">↓ LOW AROUSAL</text>

        {/* Circle boundary */}
        <circle
          cx={CENTER}
          cy={CENTER}
          r={RADIUS}
          fill="none"
          stroke="rgba(147,143,153,0.15)"
          strokeWidth={1}
          strokeDasharray="4,4"
        />

        {/* Trajectory line */}
        {points.length > 1 && (
          <polyline
            points={polylinePoints}
            fill="none"
            stroke="rgba(208,188,255,0.35)"
            strokeWidth={1.5}
            strokeLinejoin="round"
          />
        )}

        {/* Beat points */}
        {points.map((p, i) => {
          const r = 4 + p.confidence * 4; // 4–8px
          const isHovered = hovered?.beatId === p.beatId;
          return (
            <g key={p.beatId}>
              {isHovered && (
                <circle cx={p.x} cy={p.y} r={r + 4} fill={p.color} opacity={0.2} />
              )}
              <circle
                cx={p.x}
                cy={p.y}
                r={r}
                fill={p.color}
                opacity={0.85}
                stroke={isHovered ? 'white' : 'none'}
                strokeWidth={1.5}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHovered(p)}
                onMouseLeave={() => setHovered(null)}
              />
              {/* Beat index label for first + last */}
              {(i === 0 || i === points.length - 1) && (
                <text
                  x={p.x + r + 3}
                  y={p.y + 4}
                  fontSize={9}
                  fill={p.color}
                  opacity={0.8}
                >
                  {i === 0 ? 'start' : 'end'}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Tooltip */}
      {hovered && (
        <div
          style={{
            position: 'absolute',
            top: 8,
            right: 8,
            background: 'var(--md-sys-color-surface-container-highest)',
            border: '1px solid var(--md-sys-color-outline-variant)',
            borderRadius: 8,
            padding: '8px 12px',
            maxWidth: 200,
            pointerEvents: 'none',
          }}
        >
          <p style={{ margin: '0 0 4px', fontSize: 11, color: 'var(--md-sys-color-primary)', fontWeight: 600 }}>
            {hovered.beatId}
          </p>
          {hovered.desireState && (
            <p style={{ margin: '0 0 2px', fontSize: 11, color: 'var(--md-sys-color-on-surface)' }}>
              <span style={{ color: 'var(--md-sys-color-on-surface-variant)' }}>Want: </span>
              {hovered.desireState}
            </p>
          )}
          {hovered.tacticState && (
            <p style={{ margin: 0, fontSize: 11, color: 'var(--md-sys-color-on-surface)' }}>
              <span style={{ color: 'var(--md-sys-color-on-surface-variant)' }}>Tactic: </span>
              {hovered.tacticState}
            </p>
          )}
          <p style={{ margin: '4px 0 0', fontSize: 10, color: 'var(--md-sys-color-on-surface-variant)' }}>
            confidence {Math.round(hovered.confidence * 100)}%
          </p>
        </div>
      )}

      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, marginTop: 8, justifyContent: 'center' }}>
        <LegendItem color="#8ab4f8" label="Early beats" />
        <LegendItem color="#f9ab00" label="Later beats" />
        <LegendItem color="rgba(208,188,255,0.5)" label="Trajectory" line />
      </div>
    </div>
  );
}

function LegendItem({ color, label, line }: { color: string; label: string; line?: boolean }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)' }}>
      {line ? (
        <span style={{ display: 'inline-block', width: 20, height: 2, background: color }} />
      ) : (
        <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: color }} />
      )}
      {label}
    </span>
  );
}
