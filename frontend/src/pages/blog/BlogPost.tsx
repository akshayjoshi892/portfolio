import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Helmet } from "react-helmet-async";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import { ArrowLeft } from "lucide-react";
import { Reveal } from "@/components/animations/Reveal";
import { PageSkeleton } from "@/components/ui/PageSkeleton";
import { api } from "@/lib/api";
import type { BlogPost as BlogPostType } from "@/types";
import "highlight.js/styles/github-dark.css";

async function fetchPost(slug: string): Promise<BlogPostType> {
  const { data } = await api.get<BlogPostType>(`/blog/${slug}`);
  return data;
}

export default function BlogPost() {
  const { slug } = useParams<{ slug: string }>();
  const {
    data: post,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["blog", slug],
    queryFn: () => fetchPost(slug!),
    enabled: Boolean(slug),
  });

  if (isLoading) return <PageSkeleton />;
  if (isError || !post) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-24 text-center">
        <p className="text-[var(--fg-muted)]">Post not found.</p>
        <Link
          to="/blog"
          className="mt-4 inline-block text-sm font-medium text-[var(--accent)]"
        >
          Back to blog
        </Link>
      </div>
    );
  }

  return (
    <article className="mx-auto max-w-2xl px-6 py-16">
      <Helmet>
        <title>{post.title} — Akshay Joshi</title>
        <meta name="description" content={post.excerpt} />
        <meta property="og:title" content={post.title} />
        <meta property="og:description" content={post.excerpt} />
        <meta property="og:type" content="article" />
      </Helmet>

      <Reveal>
        <Link
          to="/blog"
          className="mb-6 inline-flex items-center gap-1.5 text-sm text-[var(--fg-muted)] hover:text-[var(--fg)]"
        >
          <ArrowLeft size={14} />
          Back to blog
        </Link>
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
          {post.title}
        </h1>
        <div className="mt-3 flex items-center gap-2 text-sm text-[var(--fg-muted)]">
          <time dateTime={post.publishedAt}>
            {new Date(post.publishedAt).toLocaleDateString("en-US", {
              month: "long",
              day: "numeric",
              year: "numeric",
            })}
          </time>
          <span>·</span>
          <span>{post.readTimeMinutes} min read</span>
        </div>
      </Reveal>

      <Reveal
        delay={0.1}
        className="prose prose-neutral mt-10 max-w-none dark:prose-invert prose-headings:scroll-mt-28 prose-a:text-[var(--accent)]"
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeSlug, rehypeHighlight]}
        >
          {post.content}
        </ReactMarkdown>
      </Reveal>
    </article>
  );
}
