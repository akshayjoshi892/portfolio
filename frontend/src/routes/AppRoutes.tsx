import { Suspense, lazy } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { PageTransition } from '@/components/layout/PageTransition'
import { PageSkeleton } from '@/components/ui/PageSkeleton'
import { ProtectedRoute } from '@/components/ProtectedRoute'

const Home = lazy(() => import('@/pages/Home'))
const BlogList = lazy(() => import('@/pages/blog/BlogList'))
const BlogPost = lazy(() => import('@/pages/blog/BlogPost'))
const ProjectDetail = lazy(() => import('@/pages/projects/ProjectDetail'))
const StudyMaterials = lazy(() => import('@/pages/study/StudyMaterials'))
const AdminLogin = lazy(() => import('@/pages/admin/AdminLogin'))
const AdminDashboard = lazy(() => import('@/pages/admin/AdminDashboard'))
const NotFound = lazy(() => import('@/pages/NotFound'))

export function AppRoutes() {
  const location = useLocation()

  return (
    <AnimatePresence mode="wait" initial={false}>
      <Suspense fallback={<PageSkeleton />}>
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<PageTransition><Home /></PageTransition>} />
          <Route path="/blog" element={<PageTransition><BlogList /></PageTransition>} />
          <Route path="/blog/:slug" element={<PageTransition><BlogPost /></PageTransition>} />
          <Route path="/projects/:slug" element={<PageTransition><ProjectDetail /></PageTransition>} />
          <Route path="/study/*" element={<PageTransition><StudyMaterials /></PageTransition>} />
          <Route path="/admin/login" element={<PageTransition><AdminLogin /></PageTransition>} />
          <Route
            path="/admin/*"
            element={
              <ProtectedRoute>
                <PageTransition>
                  <AdminDashboard />
                </PageTransition>
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<PageTransition><NotFound /></PageTransition>} />
        </Routes>
      </Suspense>
    </AnimatePresence>
  )
}
