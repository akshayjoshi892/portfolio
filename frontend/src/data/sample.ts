import type { ExperienceEntry, Project, SkillGroup } from '@/types'

export const SKILL_GROUPS: SkillGroup[] = [
  {
    category: 'languages',
    label: 'Languages',
    skills: [
      { name: 'TypeScript', icon: 'devicon-typescript-plain' },
      { name: 'JavaScript', icon: 'devicon-javascript-plain' },
      { name: 'Java', icon: 'devicon-java-plain' },
      { name: 'Python', icon: 'devicon-python-plain' },
      { name: 'SQL', icon: 'devicon-postgresql-plain' },
    ],
  },
  {
    category: 'frameworks',
    label: 'Frameworks',
    skills: [
      { name: 'React', icon: 'devicon-react-original' },
      { name: 'Node.js', icon: 'devicon-nodejs-plain' },
      { name: 'Express', icon: 'devicon-express-original' },
      { name: 'shadcn/ui', icon: 'simple-icons-shadcnui' },
      { name: 'Recharts', icon: 'devicon-react-original' },
    ],
  },
  {
    category: 'ai_ml',
    label: 'AI / ML',
    skills: [
      { name: 'Agentic Systems', icon: 'simple-icons-langchain' },
      { name: 'Google ADK', icon: 'devicon-googlecloud-plain' },
      { name: 'OpenAI API', icon: 'simple-icons-openai' },
    ],
  },
  {
    category: 'tools',
    label: 'Tools',
    skills: [
      { name: 'Docker', icon: 'devicon-docker-plain' },
      { name: 'MongoDB', icon: 'devicon-mongodb-plain' },
      { name: 'Redis', icon: 'devicon-redis-plain' },
      { name: 'GCP', icon: 'devicon-googlecloud-plain' },
      { name: 'Linux', icon: 'devicon-linux-plain' },
      { name: 'Git', icon: 'devicon-git-plain' },
    ],
  },
]

export const EXPERIENCE: ExperienceEntry[] = [
  {
    id: '1',
    company: 'Boeing India Pvt Ltd',
    role: 'Software Engineer',
    startDate: '2026-04',
    endDate: null,
    location: 'India',
    achievements: [
      'Built a reusable chatbot UI for agentic systems on top of a custom chat engine — plug in any agent by just swapping the API',
      'Rendered agent responses as interactive charts, tables and SQL queries, with saving, fetching and deleting of recent chats plus bookmarks',
      'Developed an Operations Cyber Risk Assessment web app where an AI system analyses machine flaws and suggests remediations',
      'Implemented database persistence, building-wise reports, previous-vs-current report comparison and user assignment of system assessments with due dates',
    ],
    techStack: ['React', 'TypeScript', 'SQL', 'shadcn/ui', 'Recharts'],
  },
  {
    id: '2',
    company: 'Zebra Technologies',
    role: 'Software Engineer',
    startDate: '2023-11',
    endDate: '2026-04',
    location: 'Bengaluru',
    achievements: [
      'Architected an enterprise commit tracking system for 300+ developers with microservices, JWT auth and RBAC',
      'Designed MongoDB schemas with compound indexes and aggregation pipelines for sub-200ms analytics over 1,000+ daily commits',
      'Built a TypeScript/React + Redux frontend with WebSocket real-time notifications and Chart.js dashboards',
      'Containerized the stack with Docker, set up CI/CD, and added a Redis cache that cut API latency by 40%',
      'Led kernel-level work for the A16 device (Linux Kernel 6.12) across Snapdragon chipsets, cutting boot time by 120 seconds',
      'Contributed changes to Google Gerrit improving test execution compatibility for CTS/GTS compliance',
    ],
    techStack: ['Node.js', 'Express', 'React', 'TypeScript', 'MongoDB', 'Redis', 'Docker', 'Linux Kernel'],
  },
]

export const PROJECTS: Project[] = [
  {
    id: '1',
    slug: 'agentic-chatbot-ui',
    title: 'Agentic Chatbot UI',
    description:
      'Pluggable chat engine and UI for agentic systems — renders charts, tables and SQL, with chat history and bookmarks. Swap the API to power any agent.',
    image: '/projects/agentic-chatbot-ui.png',
    tags: ['React', 'TypeScript', 'SQL', 'shadcn/ui', 'Recharts'],
    featured: true,
  },
  {
    id: '2',
    slug: 'cyber-risk-assessment',
    title: 'Operations Cyber Risk Assessment',
    description:
      'AI system that analyses machine flaws and suggests remediations, with building-wise reports, report comparison and due-date based assessment assignment.',
    image: '/projects/cyber-risk-assessment.png',
    tags: ['React', 'TypeScript', 'SQL', 'AI'],
    featured: true,
  },
  {
    id: '3',
    slug: 'android-log-debugging-platform',
    title: 'AI Android Log Debugging Platform',
    description:
      'Google ADK agents + OpenAI API automate log analysis — 82–85% crash categorization accuracy and 60% less debugging time.',
    image: '/projects/android-log-debugging.png',
    tags: ['Google ADK', 'OpenAI', 'React', 'Node.js', 'Java', 'Docker'],
    featured: false,
  },
  {
    id: '4',
    slug: 'distributed-task-scheduler',
    title: 'Distributed Task Scheduler',
    description: 'BullMQ + Redis job queue with priority scheduling, JWT/RBAC-secured REST APIs and 95% task execution efficiency.',
    image: '/projects/task-scheduler.png',
    tags: ['Node.js', 'Java', 'Redis', 'BullMQ', 'Docker'],
    githubUrl: 'https://github.com/akshayjoshi892/Task-queue-with-bullmq-and-expressjs',
    featured: false,
  },
]
