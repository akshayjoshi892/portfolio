export interface Project {
  id: string
  slug: string
  title: string
  description: string
  longDescription?: string
  image: string
  tags: string[]
  githubUrl?: string
  liveUrl?: string
  featured: boolean
}

export interface ExperienceEntry {
  id: string
  company: string
  role: string
  startDate: string
  endDate: string | null
  location: string
  achievements: string[]
  techStack: string[]
}

export interface SkillGroup {
  category: 'languages' | 'frameworks' | 'tools' | 'ai_ml'
  label: string
  skills: { name: string; icon: string }[]
}

export interface BlogPost {
  id: string
  slug: string
  title: string
  excerpt: string
  content: string
  coverImage?: string
  tags: string[]
  publishedAt: string
  readTimeMinutes: number
  published: boolean
}

export interface StudyPage {
  id: string
  slug: string
  title: string
  content: string
  parentId: string | null
  order: number
  icon?: string
  children?: StudyPage[]
}

export interface OnlineStatus {
  status: 'online' | 'away' | 'coding' | 'offline'
  activity: string
  updatedAt: string
  /** Last heartbeat from the owner while signed in. */
  lastSeenAt: string | null
  /** True when a heartbeat arrived in the last few minutes. */
  online: boolean
}

export interface VisitorStats {
  totalVisitors: number
  totalVisits: number
  todayVisitors: number
  activeNow: number
}
