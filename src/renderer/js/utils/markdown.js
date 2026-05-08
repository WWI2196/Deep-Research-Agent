// ── Markdown rendering ───────────────────────────────────────────

function renderMarkdown(text) {
  if (!text) return '';
  if (typeof marked === 'undefined') return escapeHtml(text);

  let content = text.trim();

  // Strip bracket tags like [task_id_name]
  content = content.replace(/\[([a-z0-9_]+(?:_[a-z0-9_]+)+)\]\s*/g, '');

  // Strip ```json/markdown fences if whole content is wrapped
  if (content.startsWith('```') && content.endsWith('```')) {
    content = content.replace(/^```(?:\w+)?\s*\n?/, '').replace(/\n?```\s*$/, '');
  }

  // Escape footnote markers so marked.js doesn't treat them as reference links
  // and swallow the definition lines from the References section.
  content = content.replace(/^\[\^(\d+)\]:\s+/gm, '**[$1]** ');
  content = content.replace(/\[\^(\d+)\](?!:)/g, '<sup>[$1]</sup>');

  try {
    return marked.parse(content);
  } catch {
    return escapeHtml(content);
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
