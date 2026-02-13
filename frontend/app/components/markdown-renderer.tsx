"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function MarkdownRenderer({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => (
          <h1 className="text-2xl font-bold mt-6 mb-3 text-[var(--foreground)]">{children}</h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-xl font-semibold mt-5 mb-2 text-[var(--foreground)]">{children}</h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-lg font-medium mt-4 mb-2 text-[var(--foreground)]">{children}</h3>
        ),
        p: ({ children }) => (
          <p className="mb-3 leading-relaxed text-[var(--foreground)]">{children}</p>
        ),
        ul: ({ children }) => (
          <ul className="list-disc pl-5 mb-3 space-y-1">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal pl-5 mb-3 space-y-1">{children}</ol>
        ),
        li: ({ children }) => (
          <li className="text-[var(--foreground)] leading-relaxed">{children}</li>
        ),
        a: ({ href, children }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--primary)] hover:underline font-medium"
          >
            {children}
          </a>
        ),
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-[var(--primary)] pl-4 my-3 italic text-[var(--muted-foreground)]">
            {children}
          </blockquote>
        ),
        code: ({ children, className }) => {
          const isInline = !className;
          if (isInline) {
            return (
              <code className="bg-[var(--muted)] px-1.5 py-0.5 rounded text-sm font-mono text-[var(--accent)]">
                {children}
              </code>
            );
          }
          return (
            <pre className="bg-[var(--muted)] rounded-lg p-4 overflow-x-auto my-3">
              <code className="text-sm font-mono">{children}</code>
            </pre>
          );
        },
        table: ({ children }) => (
          <div className="overflow-x-auto my-4">
            <table className="min-w-full border border-[var(--border)] rounded-lg overflow-hidden">
              {children}
            </table>
          </div>
        ),
        th: ({ children }) => (
          <th className="bg-[var(--muted)] px-4 py-2 text-left text-sm font-semibold border-b border-[var(--border)]">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="px-4 py-2 text-sm border-b border-[var(--border)]">{children}</td>
        ),
        hr: () => <hr className="my-6 border-[var(--border)]" />,
        strong: ({ children }) => (
          <strong className="font-semibold text-[var(--foreground)]">{children}</strong>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
