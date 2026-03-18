'use client';

import type { Beat, BeatState, SmoothedBeat } from '@/lib/types';
import { getBeatStateForCharacter, valenceColor } from '@/lib/data';
import type { ViewMode } from '../ViewModeSelector';

interface Props {
  beats: Beat[];
  character: string;
  selectedBeatId: string | null;
  onSelectBeat: (beatId: string) => void;
  smoothedBeats?: Map<string, SmoothedBeat>;
  viewMode?: ViewMode;
}

export default function BeatTimeline({ beats, character, selectedBeatId, onSelectBeat, smoothedBeats, viewMode = 'factor-graph' }: Props) {
  if (beats.length === 0) {
    return (
      <div style={{ color: 'var(--md-sys-color-on-surface-variant)', fontSize: 14, padding: 16 }}>
        No beats available for this character.
      </div>
    );
  }

  // Group beats by act, then scene
  const groups: { act: number; scene: number; beats: Beat[] }[] = [];
  const groupMap = new Map<string, typeof groups[number]>();

  for (const beat of beats) {
    const key = `${beat.act}-${beat.scene}`;
    if (!groupMap.has(key)) {
      const group = { act: beat.act, scene: beat.scene, beats: [] };
      groupMap.set(key, group);
      groups.push(group);
    }
    groupMap.get(key)!.beats.push(beat);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {/* Legend */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap', paddingLeft: 4 }}>
        <LegendSwatch color="var(--affect-positive)" label={viewMode === 'factor-graph' ? 'Positive eigenspace' : 'Positive valence'} />
        <LegendSwatch color="var(--affect-negative)" label={viewMode === 'factor-graph' ? 'Negative eigenspace' : 'Negative valence'} />
        <LegendDot color="#81c995" label={viewMode === 'factor-graph' ? 'High tactic prob' : 'High confidence'} />
        {viewMode === 'diff' && (
          <LegendDot color="#f28b82" label="Changed by smoother" />
        )}
      </div>

      {groups.map((group, gi) => (
        <div key={`${group.act}-${group.scene}`}>
          {/* Act/Scene heading */}
          {(gi === 0 || group.act !== groups[gi - 1].act) && (
            <>
              {gi > 0 && (
                <hr style={{
                  border: 'none',
                  borderTop: '2px solid var(--md-sys-color-outline-variant)',
                  margin: '16px 0 12px',
                }} />
              )}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                marginBottom: 8,
                marginTop: gi > 0 ? 0 : 0,
              }}>
                <span style={{
                  background: 'var(--md-sys-color-primary-container)',
                  color: 'var(--md-sys-color-on-primary-container)',
                  borderRadius: 8,
                  padding: '3px 10px',
                  fontSize: 11,
                  fontWeight: 700,
                  letterSpacing: '0.5px',
                  flexShrink: 0,
                }}>
                  ACT {group.act}
                </span>
                <div style={{ flex: 1, height: 1, background: 'var(--md-sys-color-outline-variant)' }} />
              </div>
            </>
          )}

          {/* Scene sub-heading */}
          <div style={{
            fontSize: 11,
            fontWeight: 600,
            color: 'var(--md-sys-color-on-surface-variant)',
            marginBottom: 6,
            marginLeft: 4,
          }}>
            Scene {group.scene}
          </div>

          {/* Beat rows */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 12 }}>
            {group.beats.map((beat) => {
              const bs: BeatState | undefined = getBeatStateForCharacter(beat, character);
              const sb = smoothedBeats?.get(beat.id);
              const useSmoothed = (viewMode === 'factor-graph' || viewMode === 'diff') && sb;

              const valence = useSmoothed
                ? (sb.affect_trans_mean[0] ?? 0)
                : (bs?.affect_state.valence ?? 0);
              const arousal = useSmoothed
                ? sb.arousal
                : (bs?.affect_state.arousal ?? 0);
              const confidence = useSmoothed
                ? sb.smoothed_tactic_prob
                : (bs?.confidence ?? 0.5);
              const tactic = useSmoothed
                ? sb.smoothed_tactic
                : (bs?.tactic_state ?? '');

              const bgColor = valenceColor(valence);
              const arousalNorm = useSmoothed ? arousal : (arousal + 1) / 2;
              const isSelected = beat.id === selectedBeatId;
              const isChanged = sb?.changed ?? false;

              return (
                <div
                  key={beat.id}
                  onClick={() => onSelectBeat(beat.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    padding: '8px 10px',
                    borderRadius: 8,
                    cursor: 'pointer',
                    background: isSelected
                      ? 'var(--md-sys-color-secondary-container)'
                      : 'var(--md-sys-color-surface-container)',
                    border: isSelected
                      ? '2px solid var(--md-sys-color-primary)'
                      : viewMode === 'diff' && isChanged
                        ? '2px dashed #f28b82'
                        : '1px solid var(--md-sys-color-outline-variant)',
                    transition: 'background 0.15s, border-color 0.15s',
                    minWidth: 0,
                  }}
                >
                  {/* Beat index */}
                  <span style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: 'var(--md-sys-color-on-surface-variant)',
                    minWidth: 20,
                    textAlign: 'center',
                    flexShrink: 0,
                  }}>
                    {beat.index ?? '?'}
                  </span>

                  {/* Valence bar */}
                  <div style={{
                    width: 6,
                    height: 28,
                    borderRadius: 3,
                    background: bgColor,
                    opacity: 0.7,
                    flexShrink: 0,
                  }} />

                  {/* Tactic label */}
                  <span style={{
                    fontSize: 13,
                    color: 'var(--md-sys-color-on-surface)',
                    fontWeight: 500,
                    flex: 1,
                    minWidth: 0,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    opacity: 0.5 + confidence * 0.5,
                  }}>
                    {tactic || '—'}
                  </span>

                  {/* Arousal indicator */}
                  <div style={{
                    width: 32,
                    height: 6,
                    borderRadius: 3,
                    background: 'var(--md-sys-color-outline-variant)',
                    flexShrink: 0,
                    overflow: 'hidden',
                  }}>
                    <div style={{
                      width: `${Math.round(Math.max(0, Math.min(1, arousalNorm)) * 100)}%`,
                      height: '100%',
                      borderRadius: 3,
                      background: arousalNorm > 0.6 ? '#f28b82' : arousalNorm < 0.3 ? '#8ab4f8' : '#fbbc04',
                    }} />
                  </div>

                  {/* Confidence dot */}
                  <span style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: confidence >= 0.7 ? '#81c995' : confidence >= 0.4 ? '#fbbc04' : '#f28b82',
                    flexShrink: 0,
                  }} />

                  {/* Diff changed indicator */}
                  {viewMode === 'diff' && isChanged && (
                    <span style={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      background: '#f28b82',
                      flexShrink: 0,
                    }} />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--md-sys-color-on-surface-variant)' }}>
      <span style={{ display: 'inline-block', width: 12, height: 12, background: color, borderRadius: 3, opacity: 0.7 }} />
      {label}
    </span>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--md-sys-color-on-surface-variant)' }}>
      <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: color }} />
      {label}
    </span>
  );
}
