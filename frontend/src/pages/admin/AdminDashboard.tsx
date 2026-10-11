import { useState } from 'react'
import { Helmet } from 'react-helmet-async'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { LogOut, Plus } from 'lucide-react'
import { MarkdownEditor } from '@/components/admin/MarkdownEditor'
import { MagneticButton } from '@/components/animations/MagneticButton'
import { useAuth } from '@/context/AuthContext'
import { api } from '@/lib/api'
import type { BlogPost, OnlineStatus } from '@/types'

type Tab = 'blog' | 'study' | 'status'

export default function AdminDashboard() {
  const { logout, user } = useAuth()
  const [tab, setTab] = useState<Tab>('blog')

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <Helmet>
        <title>Admin Dashboard</title>
      </Helmet>

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Admin</h1>
          <p className="text-sm text-[var(--fg-muted)]">Signed in as {user?.email}</p>
        </div>
        <button
          onClick={logout}
          className="flex items-center gap-2 rounded-full border border-[var(--border)] px-4 py-2 text-sm font-medium hover:bg-[var(--bg-elevated)]"
        >
          <LogOut size={14} /> Logout
        </button>
      </div>

      <div className="mt-8 flex gap-2 border-b border-[var(--border)]">
        {(['blog', 'study', 'status'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`border-b-2 px-4 py-2 text-sm font-medium capitalize transition-colors ${
              tab === t ? 'border-[var(--accent)] text-[var(--fg)]' : 'border-transparent text-[var(--fg-muted)]'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="mt-8">
        {tab === 'blog' && <BlogEditorPanel />}
        {tab === 'study' && <StudyEditorPanel />}
        {tab === 'status' && <StatusPanel />}
      </div>
    </div>
  )
}

function BlogEditorPanel() {
  const queryClient = useQueryClient()
  const [title, setTitle] = useState('')
  const [excerpt, setExcerpt] = useState('')
  const [tags, setTags] = useState('')
  const [content, setContent] = useState('')

  const { data: posts } = useQuery<BlogPost[]>({
    queryKey: ['admin-blog'],
    queryFn: async () => (await api.get('/blog')).data,
  })

  const createPost = useMutation({
    mutationFn: () =>
      api.post('/admin/blog', {
        title,
        excerpt,
        content,
        tags: tags.split(',').map((t) => t.trim()).filter(Boolean),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-blog'] })
      setTitle('')
      setExcerpt('')
      setTags('')
      setContent('')
    },
  })

  const deletePost = useMutation({
    mutationFn: (id: string) => api.delete(`/admin/blog/${id}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['admin-blog'] }),
  })

  return (
    <div className="space-y-8">
      <div className="space-y-3 rounded-2xl border border-[var(--border)] p-5">
        <h2 className="font-semibold">New Post</h2>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Title"
          className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-2.5 text-sm outline-none focus:border-[var(--accent)]"
        />
        <input
          value={excerpt}
          onChange={(e) => setExcerpt(e.target.value)}
          placeholder="Excerpt"
          className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-2.5 text-sm outline-none focus:border-[var(--accent)]"
        />
        <input
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          placeholder="Tags (comma separated)"
          className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-2.5 text-sm outline-none focus:border-[var(--accent)]"
        />
        <MarkdownEditor value={content} onChange={setContent} rows={12} />
        <MagneticButton
          onClick={() => createPost.mutate()}
          disabled={createPost.isPending || !title || !content}
          className="flex items-center gap-2 rounded-full bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-[var(--accent-fg)] disabled:opacity-60"
        >
          <Plus size={14} /> Publish Post
        </MagneticButton>
      </div>

      <div className="space-y-2">
        <h2 className="font-semibold">Existing Posts</h2>
        {posts?.map((post) => (
          <div key={post.id} className="flex items-center justify-between rounded-xl border border-[var(--border)] px-4 py-3">
            <span className="text-sm">{post.title}</span>
            <button
              onClick={() => deletePost.mutate(post.id)}
              className="text-xs font-medium text-red-500 hover:underline"
            >
              Delete
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

function StudyEditorPanel() {
  const queryClient = useQueryClient()
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [parentId, setParentId] = useState('')

  const createPage = useMutation({
    mutationFn: () => api.post('/admin/study', { title, content, parentId: parentId || null }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['study-tree'] })
      setTitle('')
      setContent('')
    },
  })

  return (
    <div className="space-y-3 rounded-2xl border border-[var(--border)] p-5">
      <h2 className="font-semibold">New Study Page</h2>
      <input
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Page title"
        className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-2.5 text-sm outline-none focus:border-[var(--accent)]"
      />
      <input
        value={parentId}
        onChange={(e) => setParentId(e.target.value)}
        placeholder="Parent page ID (optional, for nesting)"
        className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-2.5 text-sm outline-none focus:border-[var(--accent)]"
      />
      <MarkdownEditor value={content} onChange={setContent} rows={12} />
      <MagneticButton
        onClick={() => createPage.mutate()}
        disabled={createPage.isPending || !title}
        className="flex items-center gap-2 rounded-full bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-[var(--accent-fg)] disabled:opacity-60"
      >
        <Plus size={14} /> Save Page
      </MagneticButton>
    </div>
  )
}

function StatusPanel() {
  const queryClient = useQueryClient()
  const [activity, setActivity] = useState('')
  const [status, setStatus] = useState<OnlineStatus['status']>('coding')

  const updateStatus = useMutation({
    mutationFn: () => api.put('/admin/status', { status, activity }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['status'] }),
  })

  return (
    <div className="max-w-md space-y-3 rounded-2xl border border-[var(--border)] p-5">
      <h2 className="font-semibold">Update Status</h2>
      <select
        value={status}
        onChange={(e) => setStatus(e.target.value as OnlineStatus['status'])}
        className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-2.5 text-sm outline-none focus:border-[var(--accent)]"
      >
        <option value="online">Online</option>
        <option value="coding">Coding</option>
        <option value="away">Away</option>
        <option value="offline">Offline</option>
      </select>
      <input
        value={activity}
        onChange={(e) => setActivity(e.target.value)}
        placeholder="What are you working on?"
        className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-2.5 text-sm outline-none focus:border-[var(--accent)]"
      />
      <MagneticButton
        onClick={() => updateStatus.mutate()}
        disabled={updateStatus.isPending}
        className="rounded-full bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-[var(--accent-fg)] disabled:opacity-60"
      >
        Update
      </MagneticButton>
    </div>
  )
}
