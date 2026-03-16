'use client';

import { useEffect, useState } from 'react';
import { BookOpen, Users, Layers, AlertCircle } from 'lucide-react';
import NavRail from './NavRail';
import type { PlayIndexEntry } from '@/lib/types';

export default function HomeView() {
  const [plays, setPlays] = useState<PlayIndexEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/data/index.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        const d = data as { plays?: PlayIndexEntry[] };
        setPlays(d.plays ?? []);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--md-sys-color-background)' }}>
      <NavRail />

      {/* Main content offset for desktop nav rail */}
      <main
        style={{ flex: 1, padding: '24px 16px' }}
        className="ml-0 mb-[72px] md:ml-[88px] md:mb-0"
      >
        {/* Top App Bar */}
        <header style={{ marginBottom: 32 }}>
          <h1
            style={{
              fontFamily: 'var(--md-sys-typescale-display-font)',
              fontSize: 28,
              fontWeight: 400,
              color: 'var(--md-sys-color-on-background)',
              margin: 0,
            }}
          >
            UTA Model Viewer
          </h1>
          <p style={{ color: 'var(--md-sys-color-on-surface-variant)', marginTop: 6, fontSize: 14 }}>
            Theatrical AI — beat analysis, character bibles, scene bibles
          </p>
        </header>

        {/* Loading skeletons */}
        {loading && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: 16,
            }}
          >
            {[1, 2, 3].map((i) => (
              <div key={i} className="m3-card-elevated" style={{ height: 160 }}>
                <div className="skeleton" style={{ height: 20, width: '70%', marginBottom: 12 }} />
                <div className="skeleton" style={{ height: 14, width: '40%', marginBottom: 24 }} />
                <div className="skeleton" style={{ height: 12, width: '55%', marginBottom: 8 }} />
                <div className="skeleton" style={{ height: 12, width: '45%' }} />
              </div>
            ))}
          </div>
        )}

        {/* Error state */}
        {!loading && error && (
          <div
            className="m3-card"
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 12,
              padding: 32,
              maxWidth: 480,
              margin: '0 auto',
              textAlign: 'center',
            }}
          >
            <AlertCircle size={40} color="var(--md-sys-color-error)" />
            <h2 style={{ margin: 0, color: 'var(--md-sys-color-error)', fontSize: 18 }}>
              Could not load plays
            </h2>
            <p style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14, margin: 0 }}>
              {error}
            </p>
            <div
              className="m3-card"
              style={{ background: 'var(--md-sys-color-surface-container-high)', padding: 16, width: '100%', textAlign: 'left' }}
            >
              <p style={{ margin: 0, fontSize: 13, color: 'var(--md-sys-color-on-surface-variant)', lineHeight: 1.6 }}>
                Run <code style={{ color: 'var(--md-sys-color-primary)', background: 'var(--md-sys-color-surface-container-highest)', padding: '2px 6px', borderRadius: 4 }}>npm run sync</code> from
                the <code style={{ color: 'var(--md-sys-color-primary)', background: 'var(--md-sys-color-surface-container-highest)', padding: '2px 6px', borderRadius: 4 }}>viewer/</code> directory
                to copy play data from <code style={{ color: 'var(--md-sys-color-primary)', background: 'var(--md-sys-color-surface-container-highest)', padding: '2px 6px', borderRadius: 4 }}>../data/bibles/</code>
                into <code style={{ color: 'var(--md-sys-color-primary)', background: 'var(--md-sys-color-surface-container-highest)', padding: '2px 6px', borderRadius: 4 }}>public/data/</code>.
              </p>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!loading && !error && plays.length === 0 && (
          <div
            className="m3-card"
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 12,
              padding: 48,
              maxWidth: 480,
              margin: '0 auto',
              textAlign: 'center',
            }}
          >
            <BookOpen size={48} color="var(--md-sys-color-on-surface-variant)" />
            <h2 style={{ margin: 0, fontSize: 20, color: 'var(--md-sys-color-on-surface)' }}>
              No plays found
            </h2>
            <p style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14, margin: 0 }}>
              Run the sync script to import play data.
            </p>
            <code
              style={{
                background: 'var(--md-sys-color-surface-container-highest)',
                color: 'var(--md-sys-color-primary)',
                padding: '8px 16px',
                borderRadius: 8,
                fontSize: 13,
              }}
            >
              npm run sync
            </code>
          </div>
        )}

        {/* Play grid */}
        {!loading && !error && plays.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: 16,
            }}
          >
            {plays.map((play) => (
              <PlayCard key={play.id} play={play} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

function PlayCard({ play }: { play: PlayIndexEntry }) {
  return (
    <a
      href={`/plays/${play.id}`}
      style={{
        textDecoration: 'none',
        display: 'block',
      }}
    >
      <div
        className="m3-card-elevated fade-in"
        style={{
          cursor: 'pointer',
          transition: 'background 0.2s, transform 0.15s',
          height: '100%',
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLDivElement).style.background =
            'var(--md-sys-color-surface-container)';
          (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-2px)';
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLDivElement).style.background =
            'var(--md-sys-color-surface-container-low)';
          (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)';
        }}
      >
        {/* Title + Author */}
        <div style={{ marginBottom: 16 }}>
          <h2
            style={{
              fontFamily: 'var(--md-sys-typescale-display-font)',
              fontSize: 20,
              fontWeight: 500,
              color: 'var(--md-sys-color-on-surface)',
              margin: 0,
              marginBottom: 4,
              lineHeight: 1.3,
            }}
          >
            {play.title}
          </h2>
          <p
            style={{
              fontSize: 13,
              color: 'var(--md-sys-color-on-surface-variant)',
              margin: 0,
            }}
          >
            {play.author}
          </p>
        </div>

        {/* Stats */}
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <StatChip
            icon={<Users size={14} />}
            label={`${play.characters.length} characters`}
          />
          <StatChip
            icon={<Layers size={14} />}
            label={`${play.actCount} act${play.actCount !== 1 ? 's' : ''}`}
          />
        </div>

        {/* Character preview */}
        {play.characters.length > 0 && (
          <div style={{ marginTop: 12, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {play.characters.slice(0, 5).map((c) => (
              <span key={c} className="m3-chip" style={{ fontSize: 11, padding: '2px 8px' }}>
                {c}
              </span>
            ))}
            {play.characters.length > 5 && (
              <span className="m3-chip" style={{ fontSize: 11, padding: '2px 8px', opacity: 0.6 }}>
                +{play.characters.length - 5}
              </span>
            )}
          </div>
        )}
      </div>
    </a>
  );
}

function StatChip({
  icon,
  label,
}: {
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        fontSize: 12,
        color: 'var(--md-sys-color-on-surface-variant)',
      }}
    >
      {icon}
      {label}
    </span>
  );
}
