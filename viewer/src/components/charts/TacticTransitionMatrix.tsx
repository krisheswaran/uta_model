'use client';

import { useMemo, useState } from 'react';
import type { Beat } from '@/lib/types';

interface Props {
  beats: Beat[];
  character: string;
  /** Optional canonical vocab mapping: raw tactic → canonical_id */
  canonicalMap?: Record<string, string>;
  /** Max tactics to show (by frequency) */
  maxTactics?: number;
}

export default function TacticTransitionMatrix({ beats, character, canonicalMap, maxTactics = 12 }: Props) {
  const [hoveredCell, setHoveredCell] = useState<{ from: string; to: string } | null>(null);

  const { tactics, matrix, counts } = useMemo(() => {
    // Extract ordered tactic sequence for this character across beats
    const sequence: string[] = [];
    for (const beat of beats) {
      const bs = beat.beat_states.find((b) => b.character === character);
      if (bs?.tactic_state) {
        const raw = bs.tactic_state.toLowerCase().trim();
        const label = canonicalMap?.[raw] ?? raw;
        sequence.push(label);
      }
    }

    // Frequency counts
    const freq: Record<string, number> = {};
    for (const t of sequence) freq[t] = (freq[t] ?? 0) + 1;

    // Top N tactics by frequency
    const tactics = Object.entries(freq)
      .sort(([, a], [, b]) => b - a)
      .slice(0, maxTactics)
      .map(([t]) => t);

    const tacticSet = new Set(tactics);

    // Transition counts (only between top tactics)
    const raw: Record<string, Record<string, number>> = {};
    for (const t of tactics) raw[t] = {};

    for (let i = 0; i < sequence.length - 1; i++) {
      const from = sequence[i];
      const to = sequence[i + 1];
      if (tacticSet.has(from) && tacticSet.has(to)) {
        raw[from][to] = (raw[from][to] ?? 0) + 1;
      }
    }

    // Normalize rows → P(to | from)
    const matrix: Record<string, Record<string, number>> = {};
    for (const from of tactics) {
      const rowTotal = Object.values(raw[from]).reduce((a, b) => a + b, 0);
      matrix[from] = {};
      for (const to of tactics) {
        matrix[from][to] = rowTotal > 0 ? (raw[from][to] ?? 0) / rowTotal : 0;
      }
    }

    return { tactics, matrix, counts: freq };
  }, [beats, character, canonicalMap, maxTactics]);

  if (tactics.length < 2) {
    return (
      <div style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 13 }}>
        Not enough tactic data to compute transitions.
      </div>
    );
  }

  const CELL = Math.max(28, Math.floor(340 / tactics.length));
  const LABEL_W = 96;

  function cellColor(p: number): string {
    // 0 → transparent, 1 → primary at full opacity
    if (p === 0) return 'transparent';
    const opacity = 0.12 + p * 0.83;
    return `rgba(208, 188, 255, ${opacity.toFixed(2)})`; // primary color tinted
  }

  const hovered = hoveredCell
    ? matrix[hoveredCell.from]?.[hoveredCell.to]
    : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, overflowX: 'auto' }}>
      <div style={{ fontSize: 12, color: 'var(--md-sys-color-on-surface-variant)' }}>
        Tactic transition matrix — P(next tactic | current tactic). Rows = from, columns = to.
        Showing top {tactics.length} tactics by frequency.
      </div>

      <div style={{ display: 'flex', alignItems: 'flex-start' }}>
        {/* Row labels */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {/* Empty corner */}
          <div style={{ height: LABEL_W, width: LABEL_W }} />
          {tactics.map((from) => (
            <div key={from} style={{
              height: CELL,
              width: LABEL_W,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              paddingRight: 8,
              fontSize: 10,
              color: hoveredCell?.from === from
                ? 'var(--md-sys-color-primary)'
                : 'var(--md-sys-color-on-surface-variant)',
              fontWeight: hoveredCell?.from === from ? 600 : 400,
              overflow: 'hidden',
              whiteSpace: 'nowrap',
              textOverflow: 'ellipsis',
            }}>
              {from} <span style={{ marginLeft: 4, opacity: 0.5 }}>({counts[from]})</span>
            </div>
          ))}
        </div>

        {/* Matrix */}
        <div>
          {/* Column headers (rotated) */}
          <div style={{ display: 'flex', height: LABEL_W, alignItems: 'flex-end', paddingBottom: 4 }}>
            {tactics.map((to) => (
              <div key={to} style={{
                width: CELL,
                height: LABEL_W,
                display: 'flex',
                alignItems: 'flex-end',
                justifyContent: 'center',
                overflow: 'hidden',
              }}>
                <div style={{
                  transformOrigin: 'bottom center',
                  transform: 'rotate(-55deg)',
                  fontSize: 10,
                  whiteSpace: 'nowrap',
                  color: hoveredCell?.to === to
                    ? 'var(--md-sys-color-primary)'
                    : 'var(--md-sys-color-on-surface-variant)',
                  fontWeight: hoveredCell?.to === to ? 600 : 400,
                  paddingBottom: 2,
                }}>
                  {to}
                </div>
              </div>
            ))}
          </div>

          {/* Cells */}
          {tactics.map((from) => (
            <div key={from} style={{ display: 'flex' }}>
              {tactics.map((to) => {
                const p = matrix[from]?.[to] ?? 0;
                const isHovered = hoveredCell?.from === from && hoveredCell?.to === to;
                return (
                  <div
                    key={to}
                    onMouseEnter={() => setHoveredCell({ from, to })}
                    onMouseLeave={() => setHoveredCell(null)}
                    style={{
                      width: CELL,
                      height: CELL,
                      background: cellColor(p),
                      border: isHovered
                        ? '1px solid var(--md-sys-color-primary)'
                        : '1px solid var(--md-sys-color-outline-variant)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'default',
                      transition: 'background 0.1s',
                      boxSizing: 'border-box',
                    }}
                  >
                    {p > 0 && (
                      <span style={{
                        fontSize: CELL < 34 ? 8 : 10,
                        color: p > 0.6 ? 'var(--md-sys-color-on-primary-container)' : 'var(--md-sys-color-on-surface)',
                        fontWeight: p > 0.4 ? 600 : 400,
                      }}>
                        {p < 0.01 ? '<1%' : `${Math.round(p * 100)}%`}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Hover tooltip */}
      {hoveredCell && hovered !== null && (
        <div style={{
          padding: '6px 12px',
          background: 'var(--md-sys-color-surface-container-high)',
          borderRadius: 8, fontSize: 12,
          color: 'var(--md-sys-color-on-surface)',
        }}>
          <strong>{hoveredCell.from}</strong>
          {' → '}
          <strong>{hoveredCell.to}</strong>
          {': '}
          {hovered === 0
            ? 'never observed'
            : `${(hovered * 100).toFixed(1)}% of transitions from ${hoveredCell.from}`}
        </div>
      )}
    </div>
  );
}
