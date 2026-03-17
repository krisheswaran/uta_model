'use client';

import { useState } from 'react';
import type { RelationalProfile } from '@/lib/types';

interface Props {
  profiles: RelationalProfile[];
  /** If set, highlight one character and show all their partner lines */
  focusCharacter?: string;
}

const W = 420;
const H = 420;
const PAD = 44;
const INNER_W = W - PAD * 2;
const INNER_H = H - PAD * 2;

function toSvg(warmth: number, status: number) {
  // warmth: -1 (left) → +1 (right), status: -1 (bottom) → +1 (top)
  const x = PAD + ((warmth + 1) / 2) * INNER_W;
  const y = PAD + ((1 - status) / 2) * INNER_H; // flip y
  return { x, y };
}

const QUADRANT_LABELS = [
  { warmth: 0.5,  status: 0.82, text: 'High Status' },
  { warmth: 0.5,  status: -0.82, text: 'Low Status' },
  { warmth: 0.82, status: 0,    text: 'Warm →' },
  { warmth: -0.82,status: 0,    text: '← Cold' },
];

export default function RelationalSpacePlot({ profiles, focusCharacter }: Props) {
  const [hovered, setHovered] = useState<string | null>(null);

  const focused = focusCharacter
    ? profiles.find((p) => p.character === focusCharacter)
    : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ fontSize: 12, color: 'var(--md-sys-color-on-surface-variant)' }}>
        Social geometry — default relational position of each character (warmth × status).
        {focusCharacter && ' Lines show per-partner deviations.'}
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: '100%', maxWidth: W, display: 'block', overflow: 'visible' }}
        role="img"
        aria-label="Relational space plot"
      >
        {/* Quadrant backgrounds */}
        <rect x={PAD} y={PAD} width={INNER_W / 2} height={INNER_H / 2}
          fill="#4F378B" opacity={0.07} />
        <rect x={PAD + INNER_W / 2} y={PAD} width={INNER_W / 2} height={INNER_H / 2}
          fill="#81c995" opacity={0.07} />
        <rect x={PAD} y={PAD + INNER_H / 2} width={INNER_W / 2} height={INNER_H / 2}
          fill="#f28b82" opacity={0.07} />
        <rect x={PAD + INNER_W / 2} y={PAD + INNER_H / 2} width={INNER_W / 2} height={INNER_H / 2}
          fill="#8ab4f8" opacity={0.07} />

        {/* Axes */}
        <line x1={PAD} y1={PAD + INNER_H / 2} x2={PAD + INNER_W} y2={PAD + INNER_H / 2}
          stroke="var(--md-sys-color-outline-variant)" strokeWidth={1} />
        <line x1={PAD + INNER_W / 2} y1={PAD} x2={PAD + INNER_W / 2} y2={PAD + INNER_H}
          stroke="var(--md-sys-color-outline-variant)" strokeWidth={1} />

        {/* Axis tick labels */}
        {[-1, -0.5, 0, 0.5, 1].map((v) => {
          const { x } = toSvg(v, 0);
          const { y } = toSvg(0, v);
          return (
            <g key={v}>
              <text x={x} y={PAD + INNER_H + 14} textAnchor="middle"
                fontSize={9} fill="var(--md-sys-color-on-surface-variant)">{v}</text>
              {v !== 0 && (
                <text x={PAD - 6} y={y + 3} textAnchor="end"
                  fontSize={9} fill="var(--md-sys-color-on-surface-variant)">{v}</text>
              )}
            </g>
          );
        })}

        {/* Quadrant corner labels */}
        {QUADRANT_LABELS.map(({ warmth, status, text }) => {
          const { x, y } = toSvg(warmth, status);
          return (
            <text key={text} x={x} y={y} textAnchor="middle" fontSize={9}
              fill="var(--md-sys-color-on-surface-variant)" opacity={0.5}>{text}</text>
          );
        })}

        {/* Axis labels */}
        <text x={PAD + INNER_W / 2} y={H - 4} textAnchor="middle"
          fontSize={10} fill="var(--md-sys-color-on-surface-variant)">warmth</text>
        <text x={8} y={PAD + INNER_H / 2} textAnchor="middle" fontSize={10}
          fill="var(--md-sys-color-on-surface-variant)"
          transform={`rotate(-90, 8, ${PAD + INNER_H / 2})`}>status</text>

        {/* Partner deviation lines for focused character */}
        {focused && Object.entries(focused.partner_deviations).map(([partner, dev]) => {
          const base = toSvg(focused.default_warmth, focused.default_status_claim);
          const end = toSvg(
            focused.default_warmth + dev.warmth_delta,
            focused.default_status_claim + dev.status_delta,
          );
          return (
            <g key={partner}>
              <line x1={base.x} y1={base.y} x2={end.x} y2={end.y}
                stroke="var(--md-sys-color-primary)" strokeWidth={1.5}
                strokeDasharray="3 2" opacity={0.6} markerEnd="url(#arrow)" />
              <text x={end.x + 4} y={end.y - 4} fontSize={9}
                fill="var(--md-sys-color-primary)" opacity={0.85}>{partner}</text>
            </g>
          );
        })}

        {/* Arrow marker */}
        <defs>
          <marker id="arrow" markerWidth={6} markerHeight={6} refX={5} refY={3} orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--md-sys-color-primary)" opacity={0.7} />
          </marker>
        </defs>

        {/* Character dots */}
        {profiles.map((p) => {
          const { x, y } = toSvg(p.default_warmth, p.default_status_claim);
          const isHovered = hovered === p.character;
          const isFocused = focusCharacter === p.character;
          const r = isFocused ? 7 : 5;
          return (
            <g key={p.character}
              onMouseEnter={() => setHovered(p.character)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: 'default' }}
            >
              <circle cx={x} cy={y} r={r + 4} fill="transparent" />
              <circle cx={x} cy={y} r={r}
                fill={isFocused ? 'var(--md-sys-color-primary)' : 'var(--md-sys-color-secondary)'}
                opacity={isHovered || isFocused ? 1 : 0.75}
                stroke={isHovered ? 'var(--md-sys-color-on-surface)' : 'none'}
                strokeWidth={1.5}
              />
              {(isHovered || isFocused || profiles.length <= 12) && (
                <text x={x} y={y - r - 3} textAnchor="middle" fontSize={9}
                  fill="var(--md-sys-color-on-surface)" fontWeight={isFocused ? 600 : 400}>
                  {p.character.length > 12 ? p.character.slice(0, 10) + '…' : p.character}
                </text>
              )}
              {isHovered && (
                <title>{`${p.character}\nwarmth: ${p.default_warmth.toFixed(2)}, status: ${p.default_status_claim.toFixed(2)}\nσ_w: ${p.warmth_variance.toFixed(3)}, σ_s: ${p.status_variance.toFixed(3)}`}</title>
              )}
            </g>
          );
        })}
      </svg>

      {/* Hover detail */}
      {hovered && (() => {
        const p = profiles.find((x) => x.character === hovered);
        if (!p) return null;
        return (
          <div style={{
            display: 'flex', gap: 16, flexWrap: 'wrap',
            padding: '8px 12px',
            background: 'var(--md-sys-color-surface-container-high)',
            borderRadius: 8, fontSize: 12,
          }}>
            <span style={{ fontWeight: 600, color: 'var(--md-sys-color-on-surface)' }}>{p.character}</span>
            <span style={{ color: 'var(--md-sys-color-on-surface-variant)' }}>warmth {p.default_warmth.toFixed(3)}</span>
            <span style={{ color: 'var(--md-sys-color-on-surface-variant)' }}>status {p.default_status_claim.toFixed(3)}</span>
            <span style={{ color: 'var(--md-sys-color-on-surface-variant)' }}>σ_w {p.warmth_variance.toFixed(4)}</span>
            <span style={{ color: 'var(--md-sys-color-on-surface-variant)' }}>σ_s {p.status_variance.toFixed(4)}</span>
            <span style={{ color: 'var(--md-sys-color-on-surface-variant)' }}>
              {Object.keys(p.partner_deviations).length} partner deviations
            </span>
          </div>
        );
      })()}
    </div>
  );
}
