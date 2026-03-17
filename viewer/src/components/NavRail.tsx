'use client';

import { useState } from 'react';
import { Home, BookOpen, User, LayoutList, Globe, ChevronDown, ChevronUp, Library, Drama } from 'lucide-react';

interface NavItem {
  label: string;
  href: string;
  icon: React.ReactNode;
  exact?: boolean;
}

interface NavRailProps {
  playId?: string;
  character?: string;
  act?: number;
  scene?: number;
}

export default function NavRail({ playId, character, act, scene }: NavRailProps) {
  const path = typeof window !== 'undefined' ? window.location.pathname : '';
  const [bottomHidden, setBottomHidden] = useState(false);

  const items: NavItem[] = [
    { label: 'Plays', href: '/', icon: <Home size={22} />, exact: true },
  ];

  if (playId) {
    items.push({ label: 'Overview', href: `/plays/${playId}`, icon: <BookOpen size={22} />, exact: true });
  }
  if (playId && character) {
    items.push({ label: 'Character', href: `/plays/${playId}/characters/${character}`, icon: <User size={22} />, exact: true });
  }
  if (playId && act != null && scene != null) {
    items.push({ label: 'Scene', href: `/plays/${playId}/scenes/${act}/${scene}`, icon: <LayoutList size={22} />, exact: true });
  }
  if (playId) {
    items.push({ label: 'World', href: `/plays/${playId}#world`, icon: <Globe size={22} /> });
  }

  const globalItems: NavItem[] = [
    { label: 'Vocab', href: '/vocab', icon: <Library size={22} /> },
    { label: 'Improv', href: '/improv', icon: <Drama size={22} /> },
  ];

  function isActive(item: NavItem): boolean {
    if (item.exact) return path === item.href;
    return path.startsWith(item.href);
  }

  return (
    <>
      {/* ── Desktop: Left navigation rail ─────────────────────────────── */}
      <nav
        style={{
          position: 'fixed',
          top: 0, left: 0, bottom: 0,
          width: 88,
          background: 'var(--md-sys-color-surface-container-low)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          paddingTop: 16,
          paddingBottom: 16,
          gap: 4,
          zIndex: 100,
          borderRight: '1px solid var(--md-sys-color-outline-variant)',
        }}
        className="hidden md:flex"
      >
        <div style={{ width: 56, height: 56, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 16 }}>
          <span style={{ fontFamily: 'var(--md-sys-typescale-display-font)', fontSize: 24, color: 'var(--md-sys-color-primary)', fontWeight: 700, letterSpacing: '-0.5px' }}>
            UTA
          </span>
        </div>

        {items.map((item) => {
          const active = isActive(item);
          return (
            <a
              key={item.href}
              href={item.href}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 4,
                width: 72,
                padding: '12px 8px',
                borderRadius: 16,
                textDecoration: 'none',
                background: active ? 'var(--md-sys-color-secondary-container)' : 'transparent',
                color: active ? 'var(--md-sys-color-on-secondary-container)' : 'var(--md-sys-color-on-surface-variant)',
                transition: 'background 0.2s, color 0.2s',
              }}
            >
              {item.icon}
              <span style={{ fontSize: 11, fontWeight: 500 }}>{item.label}</span>
            </a>
          );
        })}

        <div style={{ flex: 1 }} />
        <div style={{ width: 40, height: 1, background: 'var(--md-sys-color-outline-variant)', margin: '4px 0' }} />

        {globalItems.map((item) => {
          const active = path.startsWith(item.href);
          return (
            <a
              key={item.href}
              href={item.href}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 4,
                width: 72,
                padding: '12px 8px',
                borderRadius: 16,
                textDecoration: 'none',
                background: active ? 'var(--md-sys-color-secondary-container)' : 'transparent',
                color: active ? 'var(--md-sys-color-on-secondary-container)' : 'var(--md-sys-color-on-surface-variant)',
                transition: 'background 0.2s, color 0.2s',
              }}
            >
              {item.icon}
              <span style={{ fontSize: 11, fontWeight: 500 }}>{item.label}</span>
            </a>
          );
        })}
      </nav>

      {/* ── Mobile: Bottom navigation bar (collapsible) ────────────────── */}
      <div
        style={{
          position: 'fixed',
          bottom: 0, left: 0, right: 0,
          zIndex: 100,
          transform: bottomHidden ? 'translateY(calc(100% - 20px))' : 'translateY(0)',
          transition: 'transform 0.28s cubic-bezier(0.4, 0, 0.2, 1)',
        }}
        className="flex flex-col md:hidden"
      >
        {/* Drag handle / toggle strip */}
        <button
          onClick={() => setBottomHidden((v) => !v)}
          aria-label={bottomHidden ? 'Show navigation' : 'Hide navigation'}
          style={{
            width: '100%',
            height: 20,
            background: 'var(--md-sys-color-surface-container-low)',
            borderTop: '1px solid var(--md-sys-color-outline-variant)',
            borderBottom: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 6,
            padding: 0,
          }}
        >
          <div style={{ width: 32, height: 3, borderRadius: 2, background: 'var(--md-sys-color-outline)' }} />
          {bottomHidden
            ? <ChevronUp size={12} color="var(--md-sys-color-on-surface-variant)" />
            : <ChevronDown size={12} color="var(--md-sys-color-on-surface-variant)" />}
        </button>

        {/* Nav items */}
        <nav
          style={{
            background: 'var(--md-sys-color-surface-container-low)',
            borderTop: '1px solid var(--md-sys-color-outline-variant)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-around',
            height: 64,
            paddingBottom: 'env(safe-area-inset-bottom)',
          }}
        >
          {[...items, ...globalItems].map((item) => {
            const active = isActive(item);
            return (
              <a
                key={item.href}
                href={item.href}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 4,
                  padding: '8px 12px',
                  borderRadius: 12,
                  textDecoration: 'none',
                  color: active ? 'var(--md-sys-color-primary)' : 'var(--md-sys-color-on-surface-variant)',
                  transition: 'color 0.2s',
                }}
              >
                {item.icon}
                <span style={{ fontSize: 10, fontWeight: 500 }}>{item.label}</span>
              </a>
            );
          })}
        </nav>
      </div>
    </>
  );
}
