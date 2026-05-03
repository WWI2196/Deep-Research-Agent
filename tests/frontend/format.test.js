import { describe, it, expect } from 'vitest';

// Pure utility functions from format.js (no DOM dependencies)
function formatNumber(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

function formatDuration(ms) {
  if (!ms || ms < 0) return '0s';
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}h ${m % 60}m ${s % 60}s`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

function getDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

function truncate(str, len) {
  if (!str) return '';
  if (str.length <= len) return str;
  return str.slice(0, len - 1) + '…';
}

function timeAgo(ts) {
  const now = Date.now();
  const diff = now - ts * 1000;
  const mins = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);
  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (mins > 0) return `${mins}min ago`;
  return 'just now';
}

describe('formatNumber', () => {
  it('formats millions', () => {
    expect(formatNumber(1_500_000)).toBe('1.5M');
  });
  it('formats thousands', () => {
    expect(formatNumber(2_500)).toBe('2.5K');
  });
  it('returns string for small numbers', () => {
    expect(formatNumber(42)).toBe('42');
  });
  it('handles zero', () => {
    expect(formatNumber(0)).toBe('0');
  });
});

describe('formatDuration', () => {
  it('formats seconds', () => {
    expect(formatDuration(45000)).toBe('45s');
  });
  it('formats minutes and seconds', () => {
    expect(formatDuration(125000)).toBe('2m 5s');
  });
  it('formats hours', () => {
    expect(formatDuration(3723000)).toBe('1h 2m 3s');
  });
  it('handles zero', () => {
    expect(formatDuration(0)).toBe('0s');
  });
  it('handles negative', () => {
    expect(formatDuration(-1)).toBe('0s');
  });
});

describe('getDomain', () => {
  it('extracts domain from URL', () => {
    expect(getDomain('https://www.example.com/path')).toBe('example.com');
  });
  it('handles URL without www', () => {
    expect(getDomain('https://github.com/user/repo')).toBe('github.com');
  });
  it('returns original for invalid URL', () => {
    expect(getDomain('not-a-url')).toBe('not-a-url');
  });
});

describe('truncate', () => {
  it('returns short strings as-is', () => {
    expect(truncate('hello', 10)).toBe('hello');
  });
  it('truncates long strings', () => {
    expect(truncate('hello world this is long', 10)).toBe('hello wor…');
  });
  it('handles empty string', () => {
    expect(truncate('', 10)).toBe('');
  });
  it('handles null/undefined', () => {
    expect(truncate(null, 10)).toBe('');
  });
});

describe('timeAgo', () => {
  it('returns just now for recent timestamps', () => {
    const ts = Math.floor(Date.now() / 1000) - 5;
    expect(timeAgo(ts)).toBe('just now');
  });
  it('returns minutes for older timestamps', () => {
    const ts = Math.floor(Date.now() / 1000) - 180;
    expect(timeAgo(ts)).toBe('3min ago');
  });
  it('returns hours', () => {
    const ts = Math.floor(Date.now() / 1000) - 7200;
    expect(timeAgo(ts)).toBe('2h ago');
  });
  it('returns days', () => {
    const ts = Math.floor(Date.now() / 1000) - 172800;
    expect(timeAgo(ts)).toBe('2d ago');
  });
});
