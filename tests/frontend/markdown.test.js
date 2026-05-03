import { describe, it, expect } from 'vitest';

// Pure utility functions from markdown.js (no DOM / marked.js dependencies)

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function stripBracketTags(text) {
  if (!text) return '';
  return text.replace(/\[([a-z0-9_]+(?:_[a-z0-9_]+)+)\]\s*/g, '');
}

function cleanFencedCode(text) {
  if (!text) return '';
  return text
    .replace(/```(\w+)?\n```/g, '')
    .replace(/```\n\s*```/g, '');
}

describe('escapeHtml', () => {
  it('escapes ampersand', () => {
    expect(escapeHtml('a & b')).toBe('a &amp; b');
  });
  it('escapes angle brackets', () => {
    expect(escapeHtml('<script>')).toBe('&lt;script&gt;');
  });
  it('escapes quotes', () => {
    expect(escapeHtml('"hello"')).toBe('&quot;hello&quot;');
  });
  it('handles empty string', () => {
    expect(escapeHtml('')).toBe('');
  });
  it('handles null/undefined', () => {
    expect(escapeHtml(null)).toBe('');
  });
  it('leaves safe text unchanged', () => {
    expect(escapeHtml('hello world')).toBe('hello world');
  });
});

describe('stripBracketTags', () => {
  it('removes bracket tags', () => {
    expect(stripBracketTags('Text [some_tag] more text')).toBe('Text more text');
  });
  it('removes tags with underscores', () => {
    expect(stripBracketTags('[complex_tag_name] Hello')).toBe('Hello');
  });
  it('handles no tags', () => {
    expect(stripBracketTags('Plain text')).toBe('Plain text');
  });
  it('handles empty string', () => {
    expect(stripBracketTags('')).toBe('');
  });
});

describe('cleanFencedCode', () => {
  it('removes empty code blocks', () => {
    expect(cleanFencedCode('```python\n```')).toBe('');
  });
  it('preserves non-empty code blocks', () => {
    expect(cleanFencedCode('```python\nprint("hi")\n```'))
      .toBe('```python\nprint("hi")\n```');
  });
  it('handles empty string', () => {
    expect(cleanFencedCode('')).toBe('');
  });
});
