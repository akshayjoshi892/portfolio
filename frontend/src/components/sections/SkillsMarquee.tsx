import { Marquee } from '@/components/animations/Marquee'
import { SKILL_GROUPS } from '@/data/sample'
import { SKILL_ICON_MAP } from '@/data/skillIcons'
import { toneFor } from '@/lib/tones'

const SKILLS = SKILL_GROUPS.flatMap((g) => g.skills.map((s) => s.name)).filter(
  (name, i, all) => all.indexOf(name) === i,
)

export function SkillsMarquee() {
  return (
    <div className="relative border-y border-[var(--border)] bg-[var(--bg-elevated)] py-3">
      <Marquee speed={1.6}>
        {SKILLS.map((name) => {
          const entry = SKILL_ICON_MAP[name]
          const Icon = entry?.icon
          return (
            <span key={name} className="mx-1.5 flex items-center gap-2 rounded-md px-3 py-1.5 font-mono text-sm" style={{ color: toneFor(name), background: `color-mix(in oklab, ${toneFor(name)} 10%, transparent)` }}>
              {Icon && <Icon size={15} />}
              {name}
            </span>
          )
        })}
      </Marquee>
    </div>
  )
}
