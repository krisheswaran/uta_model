'use client';

import type { Beat, BeatState } from '@/lib/types';
import type { SmoothedBeat } from '@/lib/types';
import type { ViewMode } from './ViewModeSelector';

interface Props {
  beat: Beat;
  beatState: BeatState;
  smoothedBeat?: SmoothedBeat;
  viewMode: ViewMode;
}

const EIGENSPACE_LABELS = ['Disempowerment', 'Blissful Ignorance', 'Burdened Power'];

export default function SmoothedBeatStateDetail({ beat, beatState: bs, smoothedBeat: sb, viewMode }: Props) {
  if (!sb) {
    return (
      <div className="m3-card" style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14 }}>
        No smoothed data available for this beat.
      </div>
    );
  }

  const isDiff = viewMode === 'diff';

  return (
    <div
      className="m3-card fade-in"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
        border: isDiff && sb.changed
          ? '2px solid var(--md-sys-color-error)'
          : '1px solid var(--md-sys-color-outline-variant)',
      }}
    >
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <p
            style={{
              fontFamily: 'var(--md-sys-typescale-display-font)',
              fontSize: 15,
              fontWeight: 500,
              color: 'var(--md-sys-color-primary)',
              margin: '0 0 2px',
            }}
          >
            {bs.character}
          </p>
          <p style={{ margin: 0, fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)' }}>
            Beat {beat.id}
          </p>
        </div>
        <span
          style={{
            padding: '4px 10px',
            borderRadius: 12,
            fontSize: 11,
            fontWeight: 600,
            background: sb.smoothed_tactic_prob >= 0.7
              ? 'var(--affect-positive-bg, rgba(129, 201, 149, 0.2))'
              : sb.smoothed_tactic_prob >= 0.4
              ? 'rgba(251, 188, 4, 0.2)'
              : 'rgba(242, 139, 130, 0.2)',
            color: sb.smoothed_tactic_prob >= 0.7
              ? '#81c995'
              : sb.smoothed_tactic_prob >= 0.4
              ? '#fbbc04'
              : '#f28b82',
          }}
        >
          {Math.round(sb.smoothed_tactic_prob * 100)}%
        </span>
      </div>

      {/* Tactic: smoothed vs LLM */}
      <div>
        <p className="m3-label-medium" style={{ marginBottom: 8 }}>
          {isDiff ? 'TACTIC (LLM vs SMOOTHED)' : 'SMOOTHED TACTIC'}
        </p>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
          {isDiff && (
            <TacticPill
              label="LLM"
              value={sb.llm_tactic}
              color={sb.changed ? 'var(--md-sys-color-error)' : 'var(--md-sys-color-on-surface-variant)'}
              bgColor={sb.changed ? 'rgba(242, 139, 130, 0.15)' : 'var(--md-sys-color-surface-container-high)'}
            />
          )}
          <TacticPill
            label={isDiff ? 'Smoothed' : ''}
            value={sb.smoothed_tactic}
            color="var(--md-sys-color-on-secondary-container)"
            bgColor="var(--md-sys-color-secondary-container)"
          />
        </div>
        <TacticDistribution distribution={sb.tactic_distribution} />
      </div>

      {/* Eigenspace Affect */}
      <div>
        <p className="m3-label-medium" style={{ marginBottom: 8 }}>EIGENSPACE AFFECT (TRANSITION)</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {EIGENSPACE_LABELS.map((label, i) => (
            <EigenBar
              key={label}
              label={label}
              mean={sb.affect_trans_mean[i] ?? 0}
              std={sb.affect_trans_std[i] ?? 0}
            />
          ))}
        </div>
      </div>

      {/* Arousal */}
      <div>
        <p className="m3-label-medium" style={{ marginBottom: 8 }}>AROUSAL (EMISSION)</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ width: 90, fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', flexShrink: 0 }}>
            Arousal
          </span>
          <div
            style={{
              flex: 1,
              height: 8,
              borderRadius: 4,
              background: 'var(--md-sys-color-surface-container-highest)',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: `${Math.max(0, Math.min(100, sb.arousal * 100))}%`,
                background: sb.arousal > 0.6 ? '#f28b82' : sb.arousal > 0.3 ? '#fbbc04' : '#8ab4f8',
                borderRadius: 4,
              }}
            />
          </div>
          <span style={{ width: 36, fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)', textAlign: 'right', flexShrink: 0 }}>
            {sb.arousal.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Social State with uncertainty */}
      <div>
        <p className="m3-label-medium" style={{ marginBottom: 8 }}>
          {isDiff ? 'SOCIAL STATE (LLM vs SMOOTHED)' : 'SOCIAL STATE'}
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <EigenBar
            label="Status"
            mean={sb.social_mean[0] ?? 0}
            std={sb.social_std[0] ?? 0}
          />
          <EigenBar
            label="Warmth"
            mean={sb.social_mean[1] ?? 0}
            std={sb.social_std[1] ?? 0}
          />
        </div>
        {isDiff && (
          <div style={{ marginTop: 6, fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)' }}>
            <span>LLM: Status {bs.social_state.status.toFixed(2)}, Warmth {bs.social_state.warmth.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Diff mode: change indicator */}
      {isDiff && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 12px',
            borderRadius: 8,
            background: sb.changed
              ? 'rgba(242, 139, 130, 0.1)'
              : 'rgba(129, 201, 149, 0.1)',
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: sb.changed ? '#f28b82' : '#81c995',
              flexShrink: 0,
            }}
          />
          <span style={{ fontSize: 12, color: 'var(--md-sys-color-on-surface)' }}>
            {sb.changed
              ? `Smoother disagreed: LLM said "${sb.llm_tactic}", smoothed to "${sb.smoothed_tactic}"`
              : `Smoother agreed with LLM tactic: "${sb.smoothed_tactic}"`}
          </span>
        </div>
      )}
    </div>
  );
}

function TacticPill({
  label,
  value,
  color,
  bgColor,
}: {
  label: string;
  value: string;
  color: string;
  bgColor: string;
}) {
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '6px 12px',
        borderRadius: 8,
        background: bgColor,
      }}
    >
      {label && (
        <span style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color, opacity: 0.7 }}>
          {label}
        </span>
      )}
      <span style={{ fontSize: 13, fontWeight: 500, color }}>{value}</span>
    </div>
  );
}

function TacticDistribution({ distribution }: { distribution: Record<string, number> }) {
  const sorted = Object.entries(distribution)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  if (sorted.length === 0) return null;

  const maxProb = sorted[0][1];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {sorted.map(([tactic, prob]) => (
        <div key={tactic} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              width: 80,
              fontSize: 10,
              color: 'var(--md-sys-color-on-surface-variant)',
              textAlign: 'right',
              flexShrink: 0,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {tactic}
          </span>
          <div
            style={{
              flex: 1,
              height: 6,
              borderRadius: 3,
              background: 'var(--md-sys-color-surface-container-highest)',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${(prob / maxProb) * 100}%`,
                background: 'var(--md-sys-color-secondary)',
                borderRadius: 3,
                opacity: 0.5 + (prob / maxProb) * 0.5,
              }}
            />
          </div>
          <span
            style={{
              width: 32,
              fontSize: 10,
              color: 'var(--md-sys-color-on-surface-variant)',
              textAlign: 'right',
              flexShrink: 0,
            }}
          >
            {(prob * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  );
}

function EigenBar({
  label,
  mean,
  std,
}: {
  label: string;
  mean: number;
  std: number;
}) {
  // Display on a [-1, 1] range centered at 0
  const clampedMean = Math.max(-1, Math.min(1, mean));
  const centerPct = 50;
  const meanPct = clampedMean * 50; // -50 to 50

  // Uncertainty band
  const stdPct = Math.min(std, 1) * 50;
  const bandLeft = centerPct + meanPct - stdPct;
  const bandRight = centerPct + meanPct + stdPct;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span
        style={{
          width: 120,
          fontSize: 11,
          color: 'var(--md-sys-color-on-surface-variant)',
          flexShrink: 0,
        }}
      >
        {label}
      </span>
      <div
        style={{
          flex: 1,
          height: 10,
          borderRadius: 5,
          background: 'var(--md-sys-color-surface-container-highest)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Uncertainty band */}
        <div
          style={{
            position: 'absolute',
            left: `${Math.max(0, bandLeft)}%`,
            width: `${Math.min(100, bandRight) - Math.max(0, bandLeft)}%`,
            top: 0,
            bottom: 0,
            background: 'var(--md-sys-color-secondary)',
            opacity: 0.15,
          }}
        />
        {/* Mean indicator */}
        {clampedMean < 0 ? (
          <div
            style={{
              position: 'absolute',
              right: `${centerPct}%`,
              top: 0,
              bottom: 0,
              width: `${Math.abs(meanPct)}%`,
              background: 'var(--affect-negative)',
              borderRadius: '5px 0 0 5px',
            }}
          />
        ) : (
          <div
            style={{
              position: 'absolute',
              left: `${centerPct}%`,
              top: 0,
              bottom: 0,
              width: `${meanPct}%`,
              background: 'var(--affect-positive)',
              borderRadius: '0 5px 5px 0',
            }}
          />
        )}
        {/* Center line */}
        <div
          style={{
            position: 'absolute',
            left: '50%',
            top: 0,
            bottom: 0,
            width: 1,
            background: 'var(--md-sys-color-outline)',
          }}
        />
      </div>
      <span
        style={{
          width: 60,
          fontSize: 10,
          color: 'var(--md-sys-color-on-surface-variant)',
          textAlign: 'right',
          flexShrink: 0,
        }}
      >
        {mean.toFixed(2)} +/- {std.toFixed(2)}
      </span>
    </div>
  );
}
