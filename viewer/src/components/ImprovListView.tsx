'use client';

import { useEffect, useState } from 'react';
import type { ImprovIndex, ImprovSessionMeta } from '@/lib/types';
import NavRail from './NavRail';

function ScorePip({ score }: { score: number }) {
  const pct = (score / 5) * 100;
  const color = score >= 4.5 ? '#81c995' : score >= 3.5 ? '#8ab4f8' : score >= 2.5 ? '#EFB8C8' : '#f28b82';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ flex: 1, height: 4, borderRadius: 2, background: 'var(--md-sys-color-surface-container-highest)' }}>
        <div style={{ width: `${pct}%`, height: '100%', borderRadius: 2, background: color, transition: 'width 0.3s' }} />
      </div>
      <span style={{ fontSize: 11, fontWeight: 600, color, minWidth: 28, textAlign: 'right' }}>
        {score.toFixed(2)}
      </span>
    </div>
  );
}

function SessionCard({ meta }: { meta: ImprovSessionMeta }) {
  const ts = new Date(meta.timestamp);
  const dateStr = ts.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  const timeStr = ts.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });

  return (
    <a
      href={`/improv/${meta.scene_id}`}
      style={{
        display: 'block',
        textDecoration: 'none',
        background: 'var(--md-sys-color-surface-container)',
        borderRadius: 16,
        padding: '16px 20px',
        border: '1px solid var(--md-sys-color-outline-variant)',
        transition: 'background 0.15s, border-color 0.15s',
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.background = 'var(--md-sys-color-surface-container-high)';
        (e.currentTarget as HTMLElement).style.borderColor = 'var(--md-sys-color-primary)';
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.background = 'var(--md-sys-color-surface-container)';
        (e.currentTarget as HTMLElement).style.borderColor = 'var(--md-sys-color-outline-variant)';
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, marginBottom: 8 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{
              fontSize: 13, fontWeight: 700,
              color: 'var(--md-sys-color-primary)',
              fontFamily: 'var(--md-sys-typescale-display-font)',
              letterSpacing: '0.02em',
            }}>
              {meta.character}
            </span>
            <span style={{ fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', opacity: 0.7 }}>
              {meta.play_id}
            </span>
          </div>
          <div style={{ fontSize: 13, color: 'var(--md-sys-color-on-surface)', fontStyle: 'italic', marginBottom: 4 }}>
            {meta.setting}
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            <span className="m3-chip">{meta.mode}</span>
            <span className="m3-chip">{meta.turn_count} turns</span>
          </div>
        </div>
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: meta.mean_score >= 4 ? '#81c995' : meta.mean_score >= 3 ? '#8ab4f8' : '#f28b82' }}>
            {meta.mean_score.toFixed(2)}
          </div>
          <div style={{ fontSize: 10, color: 'var(--md-sys-color-on-surface-variant)' }}>mean score</div>
          <div style={{ fontSize: 10, color: 'var(--md-sys-color-on-surface-variant)', marginTop: 4 }}>
            {dateStr}<br />{timeStr}
          </div>
        </div>
      </div>
    </a>
  );
}

export default function ImprovListView() {
  const [index, setIndex] = useState<ImprovIndex | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/data/improv-index.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<ImprovIndex>;
      })
      .then(setIndex)
      .catch((e) => setError(e.message));
  }, []);

  const sessions = index?.sessions ?? [];

  return (
    <div style={{ minHeight: '100vh', background: 'var(--md-sys-color-background)' }}>
      <NavRail />

      <main
        
        style={{ flex: 1, padding: '24px 20px', maxWidth: 800 }}
      >
        <div style={{ marginBottom: 24 }}>
          <h1 style={{
            fontFamily: 'var(--md-sys-typescale-display-font)',
            fontSize: 28, fontWeight: 700,
            color: 'var(--md-sys-color-on-surface)',
            marginBottom: 6,
          }}>
            Improvisation Sessions
          </h1>
          <p style={{ fontSize: 13, color: 'var(--md-sys-color-on-surface-variant)', maxWidth: 560 }}>
            Character-grounded improvised scenes produced by the pipeline's iterative revision loop.
            Each session shows the director's critiques and how each line evolved.
          </p>
        </div>

        {error && (
          <div style={{ padding: 16, borderRadius: 12, background: 'var(--md-sys-color-error-container)', color: 'var(--md-sys-color-on-error-container)', marginBottom: 16 }}>
            Could not load session index: {error}
          </div>
        )}

        {!index && !error && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {[1, 2, 3].map((i) => (
              <div key={i} className="skeleton" style={{ height: 120, borderRadius: 16 }} />
            ))}
          </div>
        )}

        {sessions.length === 0 && index && (
          <div style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14, padding: 24 }}>
            No sessions found. Run the improvisation pipeline and then <code>npm run sync</code>.
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {sessions.map((meta) => (
            <SessionCard key={meta.scene_id} meta={meta} />
          ))}
        </div>
      </main>
    </div>
  );
}
