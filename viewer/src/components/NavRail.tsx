'use client';

import { useState, useEffect, useRef } from 'react';
import { Home, BookOpen, User, LayoutList, Globe, Library, Drama, Menu, X } from 'lucide-react';

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
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

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

  const allItems = [...items, ...globalItems];

  function isActive(item: NavItem): boolean {
    if (item.exact) return path === item.href;
    return path.startsWith(item.href);
  }

  // Close menu on outside click or Escape
  useEffect(() => {
    if (!menuOpen) return;
    function handleClick(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    function handleEscape(e: KeyboardEvent) {
      if (e.key === 'Escape') setMenuOpen(false);
    }
    document.addEventListener('mousedown', handleClick);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleClick);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [menuOpen]);

  const activePage = allItems.find((item) => isActive(item));

  return (
    <div style={{ position: 'sticky', top: 0, left: 0, right: 0, zIndex: 200 }}>
      {/* Top app bar */}
      <header
        style={{
          width: '100%',
          height: 56,
          background: 'var(--md-sys-color-surface-container)',
          borderBottom: '1px solid var(--md-sys-color-outline-variant)',
          display: 'flex',
          alignItems: 'center',
          padding: '0 8px 0 16px',
          gap: 12,
        }}
      >
        {/* Brand */}
        <a href="/" style={{ textDecoration: 'none' }}>
          <span style={{
            fontFamily: 'var(--md-sys-typescale-display-font)',
            fontSize: 20,
            color: 'var(--md-sys-color-primary)',
            fontWeight: 700,
            letterSpacing: '-0.5px',
          }}>
            Uta
          </span>
        </a>

        {/* Current page indicator */}
        {activePage && (
          <>
            <span style={{ color: 'var(--md-sys-color-outline)', fontSize: 14 }}>/</span>
            <span style={{
              fontSize: 14,
              fontWeight: 500,
              color: 'var(--md-sys-color-on-surface)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              flex: 1,
              minWidth: 0,
            }}>
              {activePage.label}
            </span>
          </>
        )}

        <div style={{ flex: 1 }} />

        {/* FAB menu button */}
        <div ref={menuRef} style={{ position: 'relative' }}>
          <button
            onClick={() => setMenuOpen((v) => !v)}
            aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            style={{
              width: 44,
              height: 44,
              borderRadius: 14,
              border: 'none',
              background: menuOpen
                ? 'var(--md-sys-color-primary)'
                : 'var(--md-sys-color-primary-container)',
              color: menuOpen
                ? 'var(--md-sys-color-on-primary)'
                : 'var(--md-sys-color-on-primary-container)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              transition: 'background 0.2s, color 0.2s, transform 0.2s',
              transform: menuOpen ? 'rotate(90deg)' : 'rotate(0deg)',
              boxShadow: menuOpen ? 'none' : '0 2px 6px rgba(0,0,0,0.15)',
            }}
          >
            {menuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>

          {/* Dropdown menu */}
          {menuOpen && (
            <div
              style={{
                position: 'absolute',
                top: 52,
                right: 0,
                minWidth: 220,
                background: 'var(--md-sys-color-surface-container-high)',
                borderRadius: 16,
                boxShadow: '0 4px 16px rgba(0,0,0,0.2), 0 1px 4px rgba(0,0,0,0.1)',
                padding: '8px 0',
                animation: 'menuDropdown 0.2s cubic-bezier(0.2, 0, 0, 1)',
                transformOrigin: 'top right',
                overflow: 'hidden',
              }}
            >
              {items.map((item) => {
                const active = isActive(item);
                return (
                  <a
                    key={item.href}
                    href={item.href}
                    onClick={() => setMenuOpen(false)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      padding: '12px 20px',
                      textDecoration: 'none',
                      color: active
                        ? 'var(--md-sys-color-primary)'
                        : 'var(--md-sys-color-on-surface)',
                      background: active
                        ? 'var(--md-sys-color-primary-container)'
                        : 'transparent',
                      fontWeight: active ? 600 : 400,
                      fontSize: 14,
                      transition: 'background 0.15s',
                    }}
                  >
                    <span style={{ opacity: active ? 1 : 0.7 }}>{item.icon}</span>
                    {item.label}
                  </a>
                );
              })}

              {/* Divider */}
              <div style={{
                height: 1,
                background: 'var(--md-sys-color-outline-variant)',
                margin: '4px 16px',
              }} />

              {globalItems.map((item) => {
                const active = path.startsWith(item.href);
                return (
                  <a
                    key={item.href}
                    href={item.href}
                    onClick={() => setMenuOpen(false)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      padding: '12px 20px',
                      textDecoration: 'none',
                      color: active
                        ? 'var(--md-sys-color-primary)'
                        : 'var(--md-sys-color-on-surface)',
                      background: active
                        ? 'var(--md-sys-color-primary-container)'
                        : 'transparent',
                      fontWeight: active ? 600 : 400,
                      fontSize: 14,
                      transition: 'background 0.15s',
                    }}
                  >
                    <span style={{ opacity: active ? 1 : 0.7 }}>{item.icon}</span>
                    {item.label}
                  </a>
                );
              })}
            </div>
          )}
        </div>
      </header>
    </div>
  );
}
