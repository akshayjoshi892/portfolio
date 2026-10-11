# Portfolio Monorepo

A modern, animated developer portfolio.

- `frontend/` — React (Vite) + TypeScript, Tailwind CSS v4, Framer Motion, React Router, TanStack Query
- `backend/` — Express + TypeScript, Prisma + PostgreSQL, JWT auth

## Getting started

### Backend

```bash
cd backend
cp .env.example .env      # fill in DATABASE_URL, JWT_SECRET, etc.
npm install
npm run prisma:migrate    # creates tables
npm run seed               # creates an admin user (SEED_ADMIN_EMAIL/PASSWORD env vars)
npm run dev                 # starts API on http://localhost:4000
```

### Frontend

```bash
cd frontend
cp .env.example .env       # VITE_API_URL=http://localhost:4000/api
npm install
npm run dev                 # starts app on http://localhost:5173
```

Admin panel: `/admin/login`, protected by JWT issued by the backend.

## Deployment

- Frontend: Vercel/Netlify (static build via `npm run build`)
- Backend: Render/Railway, with a managed PostgreSQL instance (Supabase works too)
