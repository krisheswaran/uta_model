'use client';

import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { Beat, BeatState } from '@/lib/types';
import { confidenceClass, confidenceLabel } from '@/lib/data';

interface Props {
  beat: Beat;
  beatState: BeatState;
}

export default function BeatStateDetail({ beat, beatState: bs }: Props) {
  const [epistemicOpen, setEpistemicOpen] = useState(false);

  return (
    <div
      className="m3-card fade-in"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
        border: '1px solid var(--md-sys-color-outline-variant)',
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
        <span className={`confidence-badge ${confidenceClass(bs.confidence)}`}>
          {confidenceLabel(bs.confidence)}
        </span>
      </div>

      {/* Desire → Tactic → Obstacle */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
          gap: 10,
        }}
      >
        <StateField label="Desire" value={bs.desire_state} accent="primary" />
        <StateField label="Tactic" value={bs.tactic_state} accent="secondary" />
        <StateField label="Obstacle" value={bs.obstacle} accent="tertiary" />
      </div>

      {/* Affect State */}
      <div>
        <p className="m3-label-medium" style={{ marginBottom: 8 }}>AFFECT STATE</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <AffectBar label="Valence" value={bs.affect_state.valence} min={-1} max={1} />
          <AffectBar label="Arousal" value={bs.affect_state.arousal} min={-1} max={1} />
          <AffectBar label="Certainty" value={bs.affect_state.certainty} min={-1} max={1} />
          <AffectBar label="Control" value={bs.affect_state.control} min={-1} max={1} />
          <AffectBar label="Vulnerability" value={bs.affect_state.vulnerability} min={0} max={1} />
        </div>
        {bs.affect_state.rationale && (
          <p
            className="m3-body-medium"
            style={{ margin: '8px 0 0', fontSize: 12, fontStyle: 'italic' }}
          >
            {bs.affect_state.rationale}
          </p>
        )}
      </div>

      {/* Social State */}
      <div>
        <p className="m3-label-medium" style={{ marginBottom: 8 }}>SOCIAL STATE</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <AffectBar label="Status" value={bs.social_state.status} min={-1} max={1} />
          <AffectBar label="Warmth" value={bs.social_state.warmth} min={-1} max={1} />
        </div>
        {bs.social_state.rationale && (
          <p
            className="m3-body-medium"
            style={{ margin: '8px 0 0', fontSize: 12, fontStyle: 'italic' }}
          >
            {bs.social_state.rationale}
          </p>
        )}
      </div>

      {/* Defense + Psychological contradiction */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
          gap: 10,
        }}
      >
        {bs.defense_state && (
          <StateField label="Defense Mechanism" value={bs.defense_state} accent="secondary" />
        )}
        {bs.psychological_contradiction && (
          <StateField label="Psychological Contradiction" value={bs.psychological_contradiction} accent="tertiary" />
        )}
      </div>

      {/* Alternative hypothesis (flagged when confidence is low) */}
      {bs.confidence < 0.7 && bs.alternative_hypothesis && (
        <div
          style={{
            background: 'var(--md-sys-color-tertiary-container)',
            borderRadius: 8,
            padding: '10px 12px',
          }}
        >
          <p
            className="m3-label-medium"
            style={{
              marginBottom: 4,
              color: 'var(--md-sys-color-on-tertiary-container)',
            }}
          >
            ALTERNATIVE HYPOTHESIS
          </p>
          <p
            style={{
              margin: 0,
              fontSize: 13,
              color: 'var(--md-sys-color-on-tertiary-container)',
              lineHeight: 1.5,
            }}
          >
            {bs.alternative_hypothesis}
          </p>
        </div>
      )}

      {/* Superobjective reminder */}
      {bs.superobjective_reminder && (
        <p
          className="m3-body-medium"
          style={{
            margin: 0,
            fontSize: 12,
            paddingLeft: 10,
            borderLeft: '2px solid var(--md-sys-color-primary-container)',
          }}
        >
          <span style={{ color: 'var(--md-sys-color-on-surface-variant)' }}>Superobjective: </span>
          {bs.superobjective_reminder}
        </p>
      )}

      {/* Epistemic State (collapsible) */}
      {(bs.epistemic_state.known_facts.length > 0 ||
        bs.epistemic_state.hidden_secrets.length > 0 ||
        bs.epistemic_state.false_beliefs.length > 0) && (
        <div>
          <button
            onClick={() => setEpistemicOpen((v) => !v)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'var(--md-sys-color-on-surface-variant)',
              fontSize: 12,
              fontWeight: 500,
              letterSpacing: '0.5px',
              padding: 0,
              textTransform: 'uppercase',
            }}
          >
            {epistemicOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            Epistemic State
          </button>

          {epistemicOpen && (
            <div
              style={{
                marginTop: 10,
                display: 'flex',
                flexDirection: 'column',
                gap: 10,
              }}
            >
              {bs.epistemic_state.known_facts.length > 0 && (
                <EpistemicList
                  label="Known Facts"
                  items={bs.epistemic_state.known_facts}
                  color="var(--affect-positive)"
                />
              )}
              {bs.epistemic_state.hidden_secrets.length > 0 && (
                <EpistemicList
                  label="Hidden Secrets"
                  items={bs.epistemic_state.hidden_secrets}
                  color="var(--affect-negative)"
                />
              )}
              {bs.epistemic_state.false_beliefs.length > 0 && (
                <EpistemicList
                  label="False Beliefs"
                  items={bs.epistemic_state.false_beliefs}
                  color="var(--affect-neutral)"
                />
              )}
              {bs.epistemic_state.rationale && (
                <p
                  className="m3-body-medium"
                  style={{ margin: 0, fontSize: 12, fontStyle: 'italic' }}
                >
                  {bs.epistemic_state.rationale}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StateField({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: 'primary' | 'secondary' | 'tertiary';
}) {
  const colorMap = {
    primary: {
      bg: 'var(--md-sys-color-primary-container)',
      fg: 'var(--md-sys-color-on-primary-container)',
    },
    secondary: {
      bg: 'var(--md-sys-color-secondary-container)',
      fg: 'var(--md-sys-color-on-secondary-container)',
    },
    tertiary: {
      bg: 'var(--md-sys-color-tertiary-container)',
      fg: 'var(--md-sys-color-on-tertiary-container)',
    },
  };
  const { bg, fg } = colorMap[accent];

  return (
    <div
      style={{
        background: bg,
        borderRadius: 8,
        padding: '8px 10px',
      }}
    >
      <p style={{ margin: '0 0 3px', fontSize: 10, fontWeight: 600, letterSpacing: '0.5px', color: fg, opacity: 0.7, textTransform: 'uppercase' }}>
        {label}
      </p>
      <p style={{ margin: 0, fontSize: 13, color: fg, lineHeight: 1.4 }}>{value}</p>
    </div>
  );
}

function AffectBar({
  label,
  value,
  min,
  max,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
}) {
  // Normalize to [0, 1]
  const normalized = (value - min) / (max - min);
  const percentage = `${Math.round(normalized * 100)}%`;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span
        style={{
          width: 90,
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
          height: 8,
          borderRadius: 4,
          background: 'var(--md-sys-color-surface-container-highest)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* For bipolar bars (min=-1), use split bar */}
        {min < 0 ? (
          <>
            {value < 0 ? (
              <div
                style={{
                  position: 'absolute',
                  right: '50%',
                  top: 0,
                  bottom: 0,
                  width: `${Math.abs(value) * 50}%`,
                  background: 'var(--affect-negative)',
                  borderRadius: '4px 0 0 4px',
                }}
              />
            ) : (
              <div
                style={{
                  position: 'absolute',
                  left: '50%',
                  top: 0,
                  bottom: 0,
                  width: `${value * 50}%`,
                  background: 'var(--affect-positive)',
                  borderRadius: '0 4px 4px 0',
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
          </>
        ) : (
          <div
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              bottom: 0,
              width: percentage,
              background: 'var(--affect-neutral)',
              borderRadius: 4,
            }}
          />
        )}
      </div>
      <span
        style={{
          width: 36,
          fontSize: 11,
          color: 'var(--md-sys-color-on-surface-variant)',
          textAlign: 'right',
          flexShrink: 0,
        }}
      >
        {value.toFixed(2)}
      </span>
    </div>
  );
}

function EpistemicList({
  label,
  items,
  color,
}: {
  label: string;
  items: string[];
  color: string;
}) {
  return (
    <div>
      <p
        style={{
          margin: '0 0 4px',
          fontSize: 11,
          fontWeight: 600,
          color,
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
        }}
      >
        {label}
      </p>
      <ul style={{ margin: 0, paddingLeft: 16, display: 'flex', flexDirection: 'column', gap: 3 }}>
        {items.map((item, i) => (
          <li
            key={i}
            style={{
              fontSize: 12,
              color: 'var(--md-sys-color-on-surface)',
              lineHeight: 1.5,
            }}
          >
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
