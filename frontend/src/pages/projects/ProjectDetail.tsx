import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { Helmet } from "react-helmet-async";
import { ArrowLeft, ExternalLink } from "lucide-react";
import { GithubIcon } from "@/components/ui/GithubIcon";
import { Reveal } from "@/components/animations/Reveal";
import { ProjectArt } from "@/components/ui/ProjectArt";
import { SplitText } from "@/components/animations/SplitText";
import { PROJECTS } from "@/data/sample";

export default function ProjectDetail() {
  const { slug } = useParams<{ slug: string }>();
  const project = PROJECTS.find((p) => p.slug === slug);
  const [imgOk, setImgOk] = useState(false);

  if (!project) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-24 text-center">
        <p className="text-[var(--fg-muted)]">Project not found.</p>
        <Link
          to="/"
          className="mt-4 inline-block text-sm font-medium text-[var(--accent)]"
        >
          Back home
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-16">
      <Helmet>
        <title>{project.title} — Akshay Joshi</title>
        <meta name="description" content={project.description} />
      </Helmet>

      <Reveal>
        <Link
          to="/#projects"
          className="mb-6 inline-flex items-center gap-1.5 text-sm text-[var(--fg-muted)] hover:text-[var(--fg)]"
        >
          <ArrowLeft size={14} />
          Back to projects
        </Link>
        <p className="eyebrow mb-4">
          Case study — {String(PROJECTS.indexOf(project) + 1).padStart(2, "0")}
        </p>
        <h1 className="text-[clamp(2.75rem,8vw,6.5rem)] font-medium leading-[0.92] tracking-[-0.05em]">
          <SplitText text={project.title} stagger={0.02} />
        </h1>
        <p className="mt-6 max-w-xl text-lg text-[var(--fg-muted)]">
          {project.description}
        </p>

        <div className="mt-4 flex flex-wrap gap-2">
          {project.tags.map((tag) => (
            <span
              key={tag}
              className="rounded-full bg-[var(--bg-elevated)] px-3 py-1 text-xs font-medium"
            >
              {tag}
            </span>
          ))}
        </div>

        <div className="mt-6 flex gap-3">
          {project.githubUrl && (
            <a
              href={project.githubUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-full border border-[var(--border)] px-4 py-2 text-sm font-medium hover:bg-[var(--bg-elevated)]"
            >
              <GithubIcon size={14} /> Source
            </a>
          )}
          {project.liveUrl && (
            <a
              href={project.liveUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-full bg-[var(--accent)] px-4 py-2 text-sm font-medium text-[var(--accent-fg)]"
            >
              <ExternalLink size={14} /> Live Demo
            </a>
          )}
        </div>
      </Reveal>

      <Reveal
        delay={0.1}
        className="relative mt-12 aspect-[16/9] overflow-hidden rounded-3xl border border-[var(--border)]"
      >
        {!imgOk && (
          <ProjectArt
            index={PROJECTS.indexOf(project)}
            title={project.tags.join(" / ")}
          />
        )}
        <img
          src={project.image}
          alt={project.title}
          onLoad={() => setImgOk(true)}
          onError={(e) => (e.currentTarget.style.display = "none")}
          className={`absolute inset-0 h-full w-full object-cover ${imgOk ? "" : "opacity-0"}`}
        />
      </Reveal>

      {project.longDescription && (
        <Reveal
          delay={0.15}
          className="prose prose-neutral mt-10 max-w-none dark:prose-invert"
        >
          <p>{project.longDescription}</p>
        </Reveal>
      )}
    </div>
  );
}
