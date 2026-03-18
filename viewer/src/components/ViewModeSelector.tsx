'use client';

export type ViewMode = 'llm' | 'factor-graph' | 'diff';

interface Props {
  mode: ViewMode;
  onModeChange: (mode: ViewMode) => void;
  disabled?: boolean;
}

const modes: { id: ViewMode; label: string }[] = [
  { id: 'llm', label: 'LLM' },
  { id: 'factor-graph', label: 'Factor Graph' },
  { id: 'diff', label: 'Diff' },
];

export default function ViewModeSelector({ mode, onModeChange, disabled }: Props) {
  return (
    <div
      style={{
        display: 'inline-flex',
        gap: 4,
        padding: 4,
        borderRadius: 20,
        background: 'var(--md-sys-color-surface-container-high)',
        marginBottom: 12,
      }}
    >
      {modes.map((m) => {
        const isActive = mode === m.id;
        const isDisabled = disabled && m.id !== 'llm';
        return (
          <button
            key={m.id}
            onClick={() => !isDisabled && onModeChange(m.id)}
            disabled={isDisabled}
            style={{
              padding: '6px 16px',
              borderRadius: 16,
              border: 'none',
              fontSize: 12,
              fontWeight: 500,
              letterSpacing: '0.3px',
              cursor: isDisabled ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s',
              background: isActive
                ? 'var(--md-sys-color-secondary-container)'
                : 'transparent',
              color: isActive
                ? 'var(--md-sys-color-on-secondary-container)'
                : isDisabled
                ? 'var(--md-sys-color-on-surface-variant)'
                : 'var(--md-sys-color-on-surface)',
              opacity: isDisabled ? 0.4 : 1,
            }}
          >
            {m.label}
          </button>
        );
      })}
    </div>
  );
}
