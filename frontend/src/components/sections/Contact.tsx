import { useState, type FormEvent } from 'react'
import { toneAt } from '@/lib/tones'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowUpRight, Check, CheckCircle2, Copy, Loader2, Send, XCircle } from 'lucide-react'
import { Reveal } from '@/components/animations/Reveal'
import { SOCIALS } from '@/components/layout/nav'
import { api } from '@/lib/api'
import { EASE_OUT } from '@/lib/motion'

const EMAIL = 'aakshayjoshi892@gmail.com'

type Status = 'idle' | 'submitting' | 'success' | 'error'

interface FormErrors {
  name?: string
  email?: string
  message?: string
}

function validate(name: string, email: string, message: string): FormErrors {
  const errors: FormErrors = {}
  if (name.trim().length < 2) errors.name = 'Please enter your name'
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) errors.email = 'Enter a valid email'
  if (message.trim().length < 10) errors.message = 'Message should be at least 10 characters'
  return errors
}

function Field({
  id,
  label,
  error,
  textarea,
  ...props
}: {
  id: string
  label: string
  error?: string
  textarea?: boolean
  value: string
  onChange: (v: string) => void
  type?: string
}) {
  const shared = {
    id,
    value: props.value,
    onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => props.onChange(e.target.value),
    'aria-invalid': !!error,
  }
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block font-mono text-xs text-[var(--fg-muted)]">
        {label}
      </label>
      {textarea ? (
        <textarea rows={5} {...shared} className="input resize-none" />
      ) : (
        <input type={props.type ?? 'text'} {...shared} className="input" />
      )}
      {error && <p className="mt-1.5 text-xs text-[var(--tone-clay)]">{error}</p>}
    </div>
  )
}

export function Contact() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [message, setMessage] = useState('')
  const [errors, setErrors] = useState<FormErrors>({})
  const [status, setStatus] = useState<Status>('idle')
  const [copied, setCopied] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const validationErrors = validate(name, email, message)
    setErrors(validationErrors)
    if (Object.keys(validationErrors).length > 0) return

    setStatus('submitting')
    try {
      await api.post('/contact', { name, email, message })
      setStatus('success')
      setName('')
      setEmail('')
      setMessage('')
      setTimeout(() => setStatus('idle'), 4000)
    } catch {
      setStatus('error')
      setTimeout(() => setStatus('idle'), 4000)
    }
  }

  async function copyEmail() {
    try {
      await navigator.clipboard.writeText(EMAIL)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      /* clipboard unavailable */
    }
  }

  return (
    <section id="contact" className="relative overflow-hidden px-4 py-20 sm:px-6 md:py-28">
      <div className="mx-auto max-w-7xl">
        <p className="eyebrow mb-2">// contact</p>

        <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          Let's build something{' '}
          <span className="serif-italic text-[var(--accent)]">great</span>
        </h2>
        <p className="mt-3 max-w-lg text-[var(--fg-muted)]">
          Have a role to fill, a project in mind, or just want to say hi? I usually reply within a day or two.
        </p>

        <div className="mt-10 grid gap-4 lg:grid-cols-12">
          <Reveal className="space-y-4 lg:col-span-5">
            <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6">
              <p className="mb-3 font-mono text-xs text-[var(--tone-amber)]">email</p>
              <div className="flex items-center justify-between gap-3">
                <a href={`mailto:${EMAIL}`} className="link-draw truncate font-mono text-[15px] sm:text-base">
                  {EMAIL}
                </a>
                <button
                  onClick={copyEmail}
                  aria-label="Copy email address"
                  className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-[var(--border)] text-[var(--fg-muted)] transition-colors hover:bg-[var(--bg-inset)] hover:text-[var(--fg)]"
                >
                  {copied ? <Check size={14} className="text-[var(--accent)]" /> : <Copy size={14} />}
                </button>
              </div>
            </div>

            <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6">
              <p className="mb-3 font-mono text-xs text-[var(--tone-teal)]">elsewhere</p>
              <ul className="divide-y divide-[var(--border)]">
                {SOCIALS.filter((s) => s.label !== 'Email').map((s, i) => (
                  <li key={s.label}>
                    <a
                      href={s.href}
                      target="_blank"
                      rel="noreferrer"
                      className="group flex items-center justify-between py-2.5 text-sm"
                    >
                      <span className="flex items-center gap-2.5">
                        <span className="h-2 w-2 rounded-full" style={{ background: toneAt(i) }} />
                        {s.label}
                      </span>
                      <span className="flex items-center gap-1 font-mono text-xs text-[var(--fg-subtle)] group-hover:text-[var(--fg)]">
                        {s.href.replace(/^https?:\/\/(www\.)?/, '').replace(/\/$/, '')}
                        <ArrowUpRight size={13} />
                      </span>
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </Reveal>

          <Reveal delay={0.06} className="lg:col-span-7">
            <form
              onSubmit={handleSubmit}
              noValidate
              className="space-y-5 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6"
            >
              <div className="grid gap-5 sm:grid-cols-2">
                <Field id="name" label="name" value={name} onChange={setName} error={errors.name} />
                <Field id="email" label="email" type="email" value={email} onChange={setEmail} error={errors.email} />
              </div>
              <Field id="message" label="message" textarea value={message} onChange={setMessage} error={errors.message} />
              <div className="flex justify-end">
                <button
                  type="submit"
                  disabled={status === 'submitting'}
                  className="inline-flex items-center gap-2 rounded-lg bg-[var(--accent)] px-5 py-2.5 text-sm font-medium text-[var(--accent-fg)] transition-opacity hover:opacity-90 disabled:opacity-60"
                >
                  {status === 'submitting' ? <Loader2 size={16} className="animate-spin" /> : <Send size={15} />}
                  Send message
                </button>
              </div>
            </form>
          </Reveal>
        </div>
      </div>

      <AnimatePresence>
        {(status === 'success' || status === 'error') && (
          <motion.div
            role="status"
            initial={{ opacity: 0, y: 30, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 30, scale: 0.9 }}
            transition={{ duration: 0.5, ease: EASE_OUT }}
            className="glass fixed bottom-6 left-1/2 z-[80] flex -translate-x-1/2 items-center gap-2 rounded-full border border-[var(--border)] px-5 py-3 text-sm font-medium shadow-2xl"
          >
            {status === 'success' ? (
              <CheckCircle2 size={16} className="text-[var(--tone-sage)]" />
            ) : (
              <XCircle size={16} className="text-[var(--tone-clay)]" />
            )}
            {status === 'success' ? 'Message sent — thank you!' : 'Something went wrong. Try again.'}
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  )
}
