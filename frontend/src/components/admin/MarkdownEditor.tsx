import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import rehypeSlug from 'rehype-slug'
import 'highlight.js/styles/github-dark.css'

interface MarkdownEditorProps {
  value: string
  onChange: (value: string) => void
  rows?: number
}

export function MarkdownEditor({ value, onChange, rows = 20 }: MarkdownEditorProps) {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={rows}
        placeholder="Write markdown here…"
        className="w-full resize-none rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-4 font-mono text-sm outline-none focus:border-[var(--accent)]"
      />
      <div className="prose prose-neutral max-w-none overflow-y-auto rounded-xl border border-[var(--border)] p-4 dark:prose-invert" style={{ maxHeight: rows * 28 }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSlug, rehypeHighlight]}>
          {value || '*Nothing to preview yet.*'}
        </ReactMarkdown>
      </div>
    </div>
  )
}
