'use client';

import { useEffect, useRef, useState } from 'react';
import type { ImprovSession, ImprovTurn, RevisionTrace, RevisionScores } from '@/lib/types';
import NavRail from './NavRail';
import ScoreEvolutionChart from './charts/ScoreEvolutionChart';

// ─── Score bar ────────────────────────────────────────────────────────────────

const SCORE_AXES: { key: keyof RevisionScores; short: string }[] = [
  { key: 'voice_fidelity', short: 'Voice' },
  { key: 'tactic_fidelity', short: 'Tactic' },
  { key: 'knowledge_fidelity', short: 'Know.' },
  { key: 'relationship_fidelity', short: 'Rel.' },
  { key: 'subtext_richness', short: 'Sub.' },
  { key: 'emotional_transition_plausibility', short: 'Emo.' },
];

function scoreColor(v: number) {
  if (v >= 4.5) return '#81c995';
  if (v >= 3.5) return '#8ab4f8';
  if (v >= 2.5) return '#EFB8C8';
  return '#f28b82';
}

function MiniScoreRow({ scores, mean }: { scores: RevisionScores; mean?: number }) {
  const avg = mean ?? (Object.values(scores).reduce((a, b) => a + (b as number), 0) / 6);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
      {SCORE_AXES.map(({ key, short }) => {
        const v = scores[key];
        return (
          <div key={key} title={key.replace(/_/g, ' ')}
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <div style={{
              width: 28, height: 28, borderRadius: 6,
              background: scoreColor(v),
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 11, fontWeight: 700,
              color: '#1a1a1a',
            }}>
              {v.toFixed(0)}
            </div>
            <span style={{ fontSize: 9, color: 'var(--md-sys-color-on-surface-variant)' }}>{short}</span>
          </div>
        );
      })}
      <div style={{
        marginLeft: 4,
        fontSize: 18, fontWeight: 700,
        color: scoreColor(avg),
      }}>
        {avg.toFixed(2)}
      </div>
    </div>
  );
}

// ─── Director notes block ─────────────────────────────────────────────────────

function DirectorNotes({ notes, label }: { notes: string[]; label?: string }) {
  if (notes.length === 0) return null;
  return (
    <div style={{ marginBottom: 10 }}>
      {label && (
        <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase',
          color: 'var(--md-sys-color-tertiary)', marginBottom: 4 }}>
          {label}
        </div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {notes.map((fb, i) => (
          <div key={i} style={{
            fontSize: 12, color: 'var(--md-sys-color-on-surface)',
            background: 'var(--md-sys-color-tertiary-container)',
            borderRadius: 8, padding: '7px 11px',
            lineHeight: 1.55,
            borderLeft: '3px solid var(--md-sys-color-tertiary)',
          }}>
            {fb}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Candidate block ─────────────────────────────────────────────────────────

function CandidateBlock({ label, text, scores, isFinal }: {
  label: string; text: string; scores: RevisionScores; isFinal?: boolean;
}) {
  return (
    <div style={{
      borderLeft: `3px solid ${isFinal ? 'var(--md-sys-color-primary)' : 'var(--md-sys-color-outline-variant)'}`,
      paddingLeft: 12, marginBottom: 10,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span style={{
          fontSize: 11, fontWeight: 600,
          color: isFinal ? 'var(--md-sys-color-primary)' : 'var(--md-sys-color-on-surface-variant)',
        }}>
          {label}
        </span>
        <MiniScoreRow scores={scores} />
      </div>
      <blockquote style={{
        margin: 0,
        fontSize: 13, lineHeight: 1.65,
        color: 'var(--md-sys-color-on-surface)',
        fontStyle: 'italic',
        background: 'var(--md-sys-color-surface-container)',
        borderRadius: 8, padding: '8px 12px',
      }}>
        {text}
      </blockquote>
    </div>
  );
}

// ─── Turn card ────────────────────────────────────────────────────────────────

function TurnCard({ turn }: { turn: ImprovTurn }) {
  const [expanded, setExpanded] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  const mean = (
    turn.scored_line.voice_fidelity +
    turn.scored_line.tactic_fidelity +
    turn.scored_line.knowledge_fidelity +
    turn.scored_line.relationship_fidelity +
    turn.scored_line.subtext_richness +
    turn.scored_line.emotional_transition_plausibility
  ) / 6;

  const finalScores: RevisionScores = {
    voice_fidelity: turn.scored_line.voice_fidelity,
    tactic_fidelity: turn.scored_line.tactic_fidelity,
    knowledge_fidelity: turn.scored_line.knowledge_fidelity,
    relationship_fidelity: turn.scored_line.relationship_fidelity,
    subtext_richness: turn.scored_line.subtext_richness,
    emotional_transition_plausibility: turn.scored_line.emotional_transition_plausibility,
  };

  function toggle() {
    const opening = !expanded;
    setExpanded(opening);
    if (opening) {
      requestAnimationFrame(() => {
        contentRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      });
    }
  }

  return (
    <div style={{
      background: 'var(--md-sys-color-surface-container)',
      borderRadius: 16, overflow: 'hidden',
      border: '1px solid var(--md-sys-color-outline-variant)',
    }}>
      {/* Header */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--md-sys-color-outline-variant)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <span style={{
            fontSize: 11, fontWeight: 700,
            color: 'var(--md-sys-color-on-surface-variant)',
            background: 'var(--md-sys-color-surface-container-highest)',
            borderRadius: 6, padding: '2px 7px',
          }}>
            Turn {turn.turn_index}
          </span>
          <span style={{ fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)' }}>
            {turn.context.character}
          </span>
          <span className="m3-chip" style={{ marginLeft: 'auto' }}>
            {turn.revisions} revision{turn.revisions !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* Partner line (if any) */}
      {turn.context.partner_line && (
        <div style={{
          padding: '10px 16px',
          background: 'var(--md-sys-color-surface-container-low)',
          fontSize: 13, color: 'var(--md-sys-color-on-surface-variant)',
          borderBottom: '1px solid var(--md-sys-color-outline-variant)',
        }}>
          <span style={{ fontWeight: 600, fontSize: 11, display: 'block', marginBottom: 2, color: 'var(--md-sys-color-outline)' }}>
            PARTNER
          </span>
          <span style={{ fontStyle: 'italic' }}>{turn.context.partner_line}</span>
        </div>
      )}

      {/* Final line */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--md-sys-color-outline-variant)' }}>
        <span style={{
          fontWeight: 600, fontSize: 11, display: 'block', marginBottom: 4,
          color: 'var(--md-sys-color-primary)',
        }}>
          {turn.context.character}
        </span>
        <p style={{
          margin: 0, fontSize: 14, lineHeight: 1.7,
          color: 'var(--md-sys-color-on-surface)',
          fontFamily: 'var(--md-sys-typescale-display-font)',
        }}>
          {turn.final_line}
        </p>

        {turn.scored_line.candidate.internal_reasoning && (
          <p style={{
            marginTop: 8, marginBottom: 0,
            fontSize: 11, color: 'var(--md-sys-color-on-surface-variant)',
            fontStyle: 'italic', lineHeight: 1.5,
          }}>
            ✦ {turn.scored_line.candidate.internal_reasoning}
          </p>
        )}
      </div>

      {/* Score summary */}
      <div style={{ padding: '10px 16px', borderBottom: expanded ? '1px solid var(--md-sys-color-outline-variant)' : 'none' }}>
        <MiniScoreRow scores={finalScores} mean={mean} />
      </div>

      {/* Expand button */}
      <button
        onClick={toggle}
        style={{
          width: '100%', padding: '8px 16px',
          background: 'none', border: 'none', cursor: 'pointer',
          fontSize: 12, color: 'var(--md-sys-color-primary)',
          textAlign: 'left',
          borderTop: expanded ? '1px solid var(--md-sys-color-outline-variant)' : 'none',
        }}
      >
        {expanded ? '▲ Hide revision trace' : `▼ Show revision trace (${turn.revision_trace.length} rounds)`}
      </button>

      {/* Expanded content */}
      {expanded && (
        <div ref={contentRef} style={{ padding: '12px 16px 16px' }}>
          {/* Score evolution chart */}
          {turn.revision_trace.length > 1 && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--md-sys-color-on-surface-variant)', marginBottom: 8 }}>
                SCORE EVOLUTION
              </div>
              <ScoreEvolutionChart traces={turn.revision_trace} finalScores={finalScores} />
            </div>
          )}

          {/* Interleaved revision trace:
              trace[i].feedback was given to produce trace[i], so show it BEFORE trace[i].
              trace[0].feedback is always empty (no prior feedback for the first draft). */}
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--md-sys-color-on-surface-variant)', marginBottom: 10 }}>
            REVISION TRACE
          </div>
          {turn.revision_trace.map((trace, i) => (
            <div key={trace.round}>
              {/* Director's notes that produced this round */}
              <DirectorNotes
                notes={trace.feedback}
                label={i === 0 ? undefined : `Director's notes → round ${trace.round}`}
              />
              <CandidateBlock
                label={`round ${trace.round}`}
                text={trace.candidate_text}
                scores={trace.scores}
                isFinal={i === turn.revision_trace.length - 1}
              />
            </div>
          ))}

          {/* Director's notes on the final accepted line */}
          <DirectorNotes notes={turn.scored_line.feedback} label="Director's notes (on final)" />
        </div>
      )}
    </div>
  );
}

// ─── Session header ───────────────────────────────────────────────────────────

function SessionHeader({ session }: { session: ImprovSession }) {
  const ts = new Date(session.timestamp);
  const meanScore = session.transcript
    .filter((t) => t.mean_score != null)
    .reduce((sum, t, _, arr) => sum + (t.mean_score ?? 0) / arr.length, 0);

  return (
    <div style={{
      background: 'var(--md-sys-color-surface-container)',
      borderRadius: 16, padding: '16px 20px',
      border: '1px solid var(--md-sys-color-outline-variant)',
      marginBottom: 20,
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, marginBottom: 12 }}>
        <div>
          <h2 style={{
            margin: 0, marginBottom: 4,
            fontFamily: 'var(--md-sys-typescale-display-font)',
            fontSize: 20, fontWeight: 700,
            color: 'var(--md-sys-color-on-surface)',
          }}>
            {session.characters.map((c) => c.character).join(' & ')}
          </h2>
          <p style={{ margin: 0, fontSize: 13, color: 'var(--md-sys-color-on-surface-variant)', fontStyle: 'italic' }}>
            {session.setting}
          </p>
        </div>
        {meanScore > 0 && (
          <div style={{ textAlign: 'center', flexShrink: 0 }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: scoreColor(meanScore) }}>
              {meanScore.toFixed(2)}
            </div>
            <div style={{ fontSize: 10, color: 'var(--md-sys-color-on-surface-variant)' }}>avg score</div>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
        <span className="m3-chip">{session.mode}</span>
        <span className="m3-chip">{session.dramatic_register}</span>
        <span className="m3-chip">{session.constraint.replace(/_/g, ' ')}</span>
        <span className="m3-chip">{session.turns.length} turns</span>
        <span className="m3-chip">{ts.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ fontSize: 12 }}>
          <span style={{ color: 'var(--md-sys-color-on-surface-variant)', fontWeight: 600 }}>Stakes: </span>
          <span style={{ color: 'var(--md-sys-color-on-surface)' }}>{session.stakes}</span>
        </div>
        {session.prior_events && session.prior_events !== 'The scene begins.' && (
          <div style={{ fontSize: 12 }}>
            <span style={{ color: 'var(--md-sys-color-on-surface-variant)', fontWeight: 600 }}>Prior events: </span>
            <span style={{ color: 'var(--md-sys-color-on-surface)' }}>{session.prior_events}</span>
          </div>
        )}
      </div>

      {/* Character info */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 12 }}>
        {session.characters.map((c) => (
          <a
            key={c.character}
            href={`/plays/${c.play_id}/characters/${c.character}`}
            style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '6px 12px',
              background: 'var(--md-sys-color-surface-container-high)',
              borderRadius: 10, textDecoration: 'none',
              border: '1px solid var(--md-sys-color-outline-variant)',
            }}
          >
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--md-sys-color-primary)' }}>{c.character}</span>
            <span style={{ fontSize: 10, color: 'var(--md-sys-color-on-surface-variant)' }}>{c.play_id}</span>
            {c.top_tactic && (
              <span style={{ fontSize: 10, color: 'var(--md-sys-color-on-surface-variant)' }}>
                top: {c.top_tactic} ({c.top_tactic_pct.toFixed(0)}%)
              </span>
            )}
          </a>
        ))}
      </div>

      {/* Pipeline config */}
      <div style={{
        marginTop: 12, fontSize: 11,
        color: 'var(--md-sys-color-on-surface-variant)',
        display: 'flex', gap: 12, flexWrap: 'wrap',
      }}>
        {Object.entries(session.config.model_configs).map(([role, cfg]) => (
          <span key={role}>{role}: {cfg.model}</span>
        ))}
        <span>max revisions: {session.config.pipeline_params.max_revision_rounds}</span>
        <span>threshold: {session.config.pipeline_params.score_threshold}</span>
      </div>
    </div>
  );
}

// ─── Transcript panel ─────────────────────────────────────────────────────────

function TranscriptPanel({ session }: { session: ImprovSession }) {
  return (
    <div style={{
      background: 'var(--md-sys-color-surface-container)',
      borderRadius: 16, padding: '16px 20px',
      border: '1px solid var(--md-sys-color-outline-variant)',
      marginBottom: 20,
    }}>
      <h3 style={{ margin: '0 0 12px', fontSize: 14, fontWeight: 600, color: 'var(--md-sys-color-on-surface)' }}>
        Full Transcript
      </h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {session.transcript.map((line, i) => {
          const isCharacter = line.speaker !== 'PARTNER';
          return (
            <div key={i} style={{
              display: 'flex', alignItems: 'flex-start', gap: 10,
              flexDirection: isCharacter ? 'row-reverse' : 'row',
            }}>
              <div style={{
                maxWidth: '70%',
                background: isCharacter
                  ? 'var(--md-sys-color-primary-container)'
                  : 'var(--md-sys-color-surface-container-high)',
                borderRadius: isCharacter ? '16px 4px 16px 16px' : '4px 16px 16px 16px',
                padding: '8px 12px',
              }}>
                <div style={{
                  fontSize: 10, fontWeight: 600,
                  color: isCharacter ? 'var(--md-sys-color-primary)' : 'var(--md-sys-color-on-surface-variant)',
                  marginBottom: 2,
                }}>
                  {line.speaker}
                  {line.tactic && (
                    <span style={{ marginLeft: 6, fontWeight: 400, opacity: 0.7 }}>· {line.tactic}</span>
                  )}
                  {line.mean_score != null && (
                    <span style={{ marginLeft: 6, fontWeight: 700, color: scoreColor(line.mean_score) }}>
                      {line.mean_score.toFixed(2)}
                    </span>
                  )}
                </div>
                <p style={{
                  margin: 0, fontSize: 13, lineHeight: 1.6,
                  color: isCharacter ? 'var(--md-sys-color-on-primary-container)' : 'var(--md-sys-color-on-surface)',
                }}>
                  {line.line}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function ImprovSessionView({ sessionId }: { sessionId: string }) {
  const [session, setSession] = useState<ImprovSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<'turns' | 'transcript'>('transcript');

  useEffect(() => {
    fetch(`/data/improv/${sessionId}.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<ImprovSession>;
      })
      .then(setSession)
      .catch((e) => setError(e.message));
  }, [sessionId]);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--md-sys-color-background)' }}>
      <NavRail />

      <main
        className="ml-0 mb-[84px] md:ml-[88px] md:mb-0"
        style={{ flex: 1, padding: '24px 20px', maxWidth: 860 }}
      >
        <a
          href="/improv"
          style={{ fontSize: 12, color: 'var(--md-sys-color-primary)', textDecoration: 'none', display: 'inline-block', marginBottom: 16 }}
        >
          ← All sessions
        </a>

        {error && (
          <div style={{ padding: 16, borderRadius: 12, background: 'var(--md-sys-color-error-container)', color: 'var(--md-sys-color-on-error-container)' }}>
            {error}
          </div>
        )}

        {!session && !error && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="skeleton" style={{ height: 180, borderRadius: 16 }} />
            <div className="skeleton" style={{ height: 120, borderRadius: 16 }} />
            <div className="skeleton" style={{ height: 120, borderRadius: 16 }} />
          </div>
        )}

        {session && (
          <>
            <SessionHeader session={session} />

            {/* Tabs */}
            <div style={{ display: 'flex', gap: 4, marginBottom: 16, borderBottom: '1px solid var(--md-sys-color-outline-variant)', paddingBottom: 0 }}>
              {(['transcript', 'turns'] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className="m3-tab"
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: tab === t ? 'var(--md-sys-color-primary)' : 'var(--md-sys-color-on-surface-variant)',
                    borderBottom: tab === t ? '2px solid var(--md-sys-color-primary)' : '2px solid transparent',
                    paddingBottom: 8, marginBottom: -1,
                    fontSize: 13, fontWeight: tab === t ? 600 : 400,
                    textTransform: 'capitalize',
                  }}
                >
                  {t === 'turns' ? `Revision Detail (${session.turns.length})` : 'Transcript'}
                </button>
              ))}
            </div>

            {tab === 'transcript' && <TranscriptPanel session={session} />}

            {tab === 'turns' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {session.turns.map((turn) => (
                  <TurnCard key={turn.turn_index} turn={turn} />
                ))}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
