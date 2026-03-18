'use client';

import { useState } from 'react';
import FullScreenModal from './FullScreenModal';

interface Props {
  text: string;
  truncate?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

export default function TruncatedChip({ text, truncate = false, className = 'm3-chip', style }: Props) {
  const [modalOpen, setModalOpen] = useState(false);
  const shouldTruncate = truncate && text.length > 25;

  if (!shouldTruncate) {
    return (
      <span className={className} style={style}>
        {text}
      </span>
    );
  }

  return (
    <>
      <span
        className={className}
        style={{
          ...style,
          maxWidth: 200,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          cursor: 'pointer',
        }}
        onClick={() => setModalOpen(true)}
        title={text}
      >
        {text}
      </span>
      <FullScreenModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        title="Full Text"
      >
        <p
          style={{
            margin: 0,
            fontSize: 15,
            color: 'var(--md-sys-color-on-surface)',
            lineHeight: 1.6,
            wordWrap: 'break-word',
            overflowWrap: 'break-word',
          }}
        >
          {text}
        </p>
      </FullScreenModal>
    </>
  );
}
