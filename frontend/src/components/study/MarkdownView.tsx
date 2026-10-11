import { memo, useRef, useState, type ComponentProps, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeSlug from 'rehype-slug'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import { Check, Copy, Link2 } from 'lucide-react'
import { resolveAsset, resolveLink } from '@/lib/study'
import { useSmoothScroll } from '@/components/providers/SmoothScroll'
import 'highlight.js/styles/github-dark.css'
import 'katex/dist/katex.min.css'

/* ---- GitHub-style callouts: > [!NOTE] / [!TIP] / [!IMPORTANT] / [!WARNING] / [!CAUTION] ---- */

type HNode = { type: string; tagName?: string; value?: string; properties?: Record<string, unknown>; children?: HNode[] }

const CALLOUT = /^\s*\[!(note|tip|important|warning|caution)\]\s*/i

function rehypeCallouts() {
  const walk = (node: HNode) => {
    if (node.type === 'element' && node.tagName === 'blockquote') {
      const p = node.children?.find((c) => c.type === 'element')
      const text = p?.tagName === 'p' ? p.children?.[0] : undefined
      const m = text?.type === 'text' ? text.value?.match(CALLOUT) : null
      if (m && text && p) {
        const kind = m[1].toLowerCase()
        text.value = text.value!.slice(m[0].length)
        // Drop the line break GitHub puts after the marker.
        if (!text.value && p.children?.[1]?.type === 'element' && p.children[1].tagName === 'br') p.children.splice(1, 1)
        node.properties = { ...node.properties, className: ['callout', `callout-${kind}`], dataCallout: kind }
      }
    }
    node.children?.forEach(walk)
  }
  return (tree: HNode) => walk(tree)
}

/* ---- Pieces ---- */

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textOf).join('')
  if (node && typeof node === 'object' && 'props' in node) return textOf((node.props as { children?: ReactNode }).children)
  return ''
}

function CodeBlock({ children, ...props }: ComponentProps<'pre'>) {
  const ref = useRef<HTMLPreElement>(null)
  const [copied, setCopied] = useState(false)
  const codeEl = Array.isArray(children) ? children[0] : children
  const className = (codeEl as { props?: { className?: string } })?.props?.className ?? ''
  const lang = className.match(/language-([\w+#-]+)/)?.[1]

  async function copy() {
    try {
      await navigator.clipboard.writeText(ref.current?.innerText ?? textOf(children))
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable */
    }
  }

  return (
    <div className="code-block group">
      <div className="code-block__bar">
        <span>{lang ?? 'text'}</span>
        <button onClick={copy} aria-label="Copy code" className="code-block__copy">
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre ref={ref} {...props} data-lenis-prevent>
        {children}
      </pre>
    </div>
  )
}

function Heading({ level, id, children, ...props }: ComponentProps<'h2'> & { level: 2 | 3 | 4 }) {
  const Tag = `h${level}` as const
  return (
    <Tag id={id} {...props} className="group/h relative">
      {children}
      {id && (
        <a href={`#${id}`} aria-label="Link to this section" className="heading-anchor">
          <Link2 size={16} />
        </a>
      )}
    </Tag>
  )
}

/* ---- Main ---- */

export const MarkdownView = memo(function MarkdownView({ source, file }: { source: string; file: string }) {
  const { scrollTo } = useSmoothScroll()

  const components: Components = {
    a({ href = '', children, node: _node, ...props }) {
      const link = resolveLink(file, href)
      if (link.to) return <Link to={link.to}>{children}</Link>
      if (href.startsWith('#')) {
        return (
          <a
            href={href}
            onClick={(e) => {
              e.preventDefault()
              const el = document.getElementById(decodeURIComponent(href.slice(1)))
              if (el) scrollTo(el, { offset: -110 })
              history.replaceState(null, '', href)
            }}
            {...props}
          >
            {children}
          </a>
        )
      }
      return (
        <a href={link.href} {...(link.external ? { target: '_blank', rel: 'noreferrer' } : {})} {...props}>
          {children}
        </a>
      )
    },
    img({ src = '', alt = '', node: _node, ...props }) {
      return (
        <span className="md-figure">
          <img src={resolveAsset(file, String(src))} alt={alt} loading="lazy" {...props} />
          {alt && <span className="md-caption">{alt}</span>}
        </span>
      )
    },
    pre: ({ node: _node, ...props }) => <CodeBlock {...props} />,
    table: ({ node: _node, ...props }) => (
      <div className="table-wrap" data-lenis-prevent>
        <table {...props} />
      </div>
    ),
    h2: ({ node: _node, ...props }) => <Heading level={2} {...props} />,
    h3: ({ node: _node, ...props }) => <Heading level={3} {...props} />,
    h4: ({ node: _node, ...props }) => <Heading level={4} {...props} />,
  }

  return (
    <div className="md-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeSlug, [rehypeHighlight, { detect: false, ignoreMissing: true }], rehypeKatex, rehypeCallouts]}
        components={components}
      >
        {source}
      </ReactMarkdown>
    </div>
  )
})
