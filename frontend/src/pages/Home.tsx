import { Helmet } from "react-helmet-async";
import { Hero } from "@/components/sections/Hero";
import { SkillsMarquee } from "@/components/sections/SkillsMarquee";
import { About } from "@/components/sections/About";
import { Experience } from "@/components/sections/Experience";
import { Projects } from "@/components/sections/Projects";
import { Activity } from "@/components/sections/Activity";
import { Contact } from "@/components/sections/Contact";

export default function Home() {
  return (
    <>
      <Helmet>
        <title>Akshay Joshi — Full-Stack / AI Engineer</title>
        <meta
          name="description"
          content="Portfolio of Akshay Joshi, a full-stack and AI engineer building fast, polished, production-grade web applications."
        />
        <meta
          property="og:title"
          content="Akshay Joshi — Full-Stack / AI Engineer"
        />
        <meta
          property="og:description"
          content="Portfolio of Akshay Joshi, a full-stack and AI engineer building fast, polished, production-grade web applications."
        />
        <meta property="og:type" content="website" />
      </Helmet>

      <Hero />
      <SkillsMarquee />
      <About />
      <Experience />
      <Projects />
      <Activity />
      <Contact />
    </>
  );
}
