'use client';

import { useState } from 'react';
import type { RevisionTrace, RevisionScores } from '@/lib/types';

const AXES: { key: keyof RevisionScores; label: string; color: string }[] = [
  { key: 'voice_fidelity',                    label: 'Voice',        color: '#D0BCFF' },
  { key: 'tactic_fidelity',                   label: 'Tactic',       color: '#EFB8C8' },
  { key: 'knowledge_fidelity',                label: 'Knowledge',    color: '#81c995' },
  { key: 'relationship_fidelity',             label: 'Relationship', color: '#8ab4f8' },
  { key: 'subtext_richness',                  label: 'Subtext',      color: '#CCC2DC' },
  { key: 'emotional_transition_plausibility', label: 'Emotion',      color: '#f28b82' },
];

interface Props {
  /** All revision traces for a single turn */
  traces: RevisionTrace[];
  /** Final scored values (after last revision) */
  finalScores: RevisionScores;
}

const W = 320;
const H = 140;
const PAD_L = 28;
const PAD_R = 8;
const PAD_T = 8;
const PAD_B = 24;
const INNER_W = W - PAD_L - PAD_R;
const INNER_H = H - PAD_T - PAD_B;

export default function ScoreEvolutionChart({ traces, finalScores }: Props) {
  const [hiddenAxes, setHiddenAxes] = useState<Set<string>>(new Set());

  // Build series: rounds 0..N where round N = finalScores
  const allRounds = [...traces.map((t) => t.scores), finalScores];
  const roundCount = allRounds.length;

  function toX(i: number) {
    return PAD_L + (roundCount <= 1 ? INNER_W / 2 : (i / (roundCount - 1)) * INNER_W);
  }
  function toY(score: number) {
    return PAD_T + INNER_H - ((score / 5) * INNER_H);
  }

  function toggle(key: string) {
    setHiddenAxes((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, display: 'block' }}>
        {/* Y gridlines */}
        {[1, 2, 3, 4, 5].map((v) => (
          <g key={v}>
            <line x1={PAD_L} y1={toY(v)} x2={PAD_L + INNER_W} y2={toY(v)}
              stroke="var(--md-sys-color-outline-variant)" strokeWidth={0.5} strokeDasharray="2 3" />
            <text x={PAD_L - 4} y={toY(v) + 3} textAnchor="end" fontSize={8}
              fill="var(--md-sys-color-on-surface-variant)">{v}</text>
          </g>
        ))}

        {/* X axis */}
        <line x1={PAD_L} y1={PAD_T + INNER_H} x2={PAD_L + INNER_W} y2={PAD_T + INNER_H}
          stroke="var(--md-sys-color-outline-variant)" strokeWidth={1} />

        {/* X tick labels */}
        {allRounds.map((_, i) => (
          <text key={i} x={toX(i)} y={H - 6} textAnchor="middle" fontSize={8}
            fill="var(--md-sys-color-on-surface-variant)">
            {i < traces.length ? `r${i + 1}` : 'final'}
          </text>
        ))}

        {/* Score lines */}
        {AXES.map(({ key, color }) => {
          if (hiddenAxes.has(key)) return null;
          const points = allRounds.map((r, i) => `${toX(i)},${toY(r[key])}`).join(' ');
          return (
            <g key={key}>
              <polyline points={points} fill="none" stroke={color}
                strokeWidth={1.8} strokeLinejoin="round" strokeLinecap="round" />
              {allRounds.map((r, i) => (
                <circle key={i} cx={toX(i)} cy={toY(r[key])} r={2.5} fill={color} />
              ))}
            </g>
          );
        })}
      </svg>

      {/* Legend / toggles */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {AXES.map(({ key, label, color }) => {
          const hidden = hiddenAxes.has(key);
          const finalVal = finalScores[key];
          return (
            <button
              key={key}
              onClick={() => toggle(key)}
              style={{
                display: 'flex', alignItems: 'center', gap: 5,
                background: 'none', border: '1px solid var(--md-sys-color-outline-variant)',
                borderRadius: 6, padding: '2px 8px', cursor: 'pointer',
                opacity: hidden ? 0.35 : 1,
                fontSize: 11,
                color: 'var(--md-sys-color-on-surface-variant)',
              }}
            >
              <span style={{ width: 10, height: 3, borderRadius: 2, background: color, display: 'inline-block', flexShrink: 0 }} />
              {label}
              <span style={{ fontWeight: 600, color: 'var(--md-sys-color-on-surface)', marginLeft: 2 }}>
                {finalVal.toFixed(1)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
