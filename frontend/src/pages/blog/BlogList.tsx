import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Helmet } from "react-helmet-async";
import {
  Reveal,
  staggerContainer,
  staggerItem,
} from "@/components/animations/Reveal";
import { CardSkeleton } from "@/components/ui/PageSkeleton";
import { motion } from "framer-motion";
import { api } from "@/lib/api";
import type { BlogPost } from "@/types";

async function fetchPosts(): Promise<BlogPost[]> {
  const { data } = await api.get<BlogPost[]>("/blog");
  return data;
}

export default function BlogList() {
  const {
    data: posts,
    isLoading,
    isError,
  } = useQuery({ queryKey: ["blog"], queryFn: fetchPosts });

  return (
    <div className="mx-auto max-w-3xl px-6 py-16">
      <Helmet>
        <title>Blog — Akshay Joshi</title>
        <meta
          name="description"
          content="Writing on full-stack engineering, AI, and building products."
        />
      </Helmet>

      <Reveal>
        <p className="eyebrow mb-4">Writing</p>
        <h1 className="text-[clamp(3rem,9vw,6.5rem)] font-medium leading-[0.92] tracking-[-0.05em]">
          Notes &amp;{" "}
          <span className="serif-italic font-normal text-[var(--accent)]">
            essays.
          </span>
        </h1>
        <p className="mt-6 max-w-md text-lg text-[var(--fg-muted)]">
          On engineering, AI, and building things people love.
        </p>
      </Reveal>

      {isLoading && (
        <div className="mt-10 space-y-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      )}

      {isError && (
        <p className="mt-10 text-sm text-[var(--fg-muted)]">
          Could not load posts. Please try again later.
        </p>
      )}

      <motion.div
        variants={staggerContainer}
        initial="hidden"
        animate={posts ? "show" : "hidden"}
        className="mt-10 space-y-4"
      >
        {posts?.map((post) => (
          <motion.div key={post.id} variants={staggerItem}>
            <Link
              to={`/blog/${post.slug}`}
              className="group block rounded-2xl border border-[var(--border)] p-6 transition-all duration-500 hover:-translate-y-1 hover:border-[var(--border-strong)] hover:bg-[var(--bg-elevated)]"
            >
              <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--fg-muted)]">
                <time dateTime={post.publishedAt}>
                  {new Date(post.publishedAt).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  })}
                </time>
                <span>·</span>
                <span>{post.readTimeMinutes} min read</span>
              </div>
              <h2 className="mt-2 text-2xl font-medium tracking-tight transition-colors group-hover:text-[var(--accent)]">
                {post.title}
              </h2>
              <p className="mt-1.5 text-sm text-[var(--fg-muted)]">
                {post.excerpt}
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {post.tags.map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full bg-[var(--bg-elevated)] px-2.5 py-1 text-xs font-medium"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </Link>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
