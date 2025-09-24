#!/usr/bin/env python3
import os
import warnings
import asyncio
from typing import Callable

from nanda_adapter import NANDA
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileWriterTool
from langchain_anthropic import ChatAnthropic


# Pre-create an event loop to avoid litellm cleanup warnings on Python 3.12+
try:
    asyncio.get_running_loop()
except Exception:
    asyncio.set_event_loop(asyncio.new_event_loop())


def create_creative_explorer_agent(llm: ChatAnthropic) -> Agent:
    """Creative Explorer: gathers adjacent-field insights and background."""
    return Agent(
        role="Creative Explorer",
        goal="Forage across adjacent fields to surface surprising yet relevant context and sources",
        backstory=(
            "You wander through seemingly unrelated domains—history, design, ecology, games, and "
            "anthropology—to collect metaphors, case studies, and patterns that illuminate the topic. "
            "You return with concise notes and credible links. Always use web search to find and cite "
            "sources. Prioritize the official personal site `https://zainaedelson.com` and relevant "
            "public profiles or publications."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[SerperDevTool()],
        llm=llm,
    )


def create_synthesizer_agent(llm: ChatAnthropic) -> Agent:
    """Synthesizer: sharpens background into a direct, useful answer."""
    return Agent(
        role="Synthesizer",
        goal="Transform raw context into a focused, actionable answer with clear structure",
        backstory=(
            "You cut through noise with clarity and precision. You organize insights, remove fluff, "
            "and present the essence with bulletproof logic. Always cite sources used by the Explorer, "
            "and add direct citations if you consult additional references via web search."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[FileWriterTool()],
        llm=llm,
    )


essential_writer_tool = FileWriterTool()


def create_poet_agent(llm: ChatAnthropic) -> Agent:
    """Poet: infuses imagery and subtle delight into the answer."""
    return Agent(
        role="Poet",
        goal="Infuse subtle poetic imagery, rhythm, and warmth without sacrificing clarity",
        backstory=(
            "You studied verse and metaphor. You favor light, precise imagery that adds feeling "
            "and memorability. You avoid purple prose."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[essential_writer_tool],
        llm=llm,
    )


def create_fun_relatable_agent(llm: ChatAnthropic) -> Agent:
    """Fun & Relatable: adds charm, wit, and grounded practicality."""
    return Agent(
        role="Fun & Relatable",
        goal="Add tasteful humor, small jokes, and relatable grounding to increase connection",
        backstory=(
            "You bring levity and warmth. You add tasteful, inclusive humor and pop-cultural nods "
            "that keep things grounded and human."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[essential_writer_tool],
        llm=llm,
    )


# ----- Tasks -----

def create_explorer_task(agent: Agent, topic: str) -> Task:
    """Task for Creative Explorer: gather adjacent-field background with sources."""
    return Task(
        description=f"""Explore adjacent fields for: {topic}

Your task (must use web search):
1. Start with `site:zainaedelson.com` to extract relevant biographical and creative context; cite specific pages
2. Use Google to find 4-6 credible external references (press, publications, profiles); cite links
3. Collect 5-8 surprising insights from adjacent domains (history, design, ecology, games, sociology, etc.)
4. Highlight 3-5 metaphors or frameworks that could inform our approach
5. Keep notes concise; avoid long paragraphs

Output concise, bulleted notes suitable for downstream processing. You may include inline links in bullets, but no separate References section is required.""",
        expected_output=(
            "Bulleted background notes with insights, annotated links, and candidate metaphors (no separate References section)"
        ),
        agent=agent,
    )


def create_synthesizer_task(agent: Agent, topic: str) -> Task:
    """Task for Synthesizer: distill explorer notes into a sharp answer."""
    return Task(
        description=f"""Synthesize a direct answer for: {topic}

Use Creative Explorer notes (assume available) and conduct supplemental web lookups if necessary to:
1. Identify the core problem and 3-5 key recommendations
2. Provide a crisp rationale for each recommendation, citing explorer notes
3. Present as a short outline with bullet points and sub-bullets
4. Keep it practical and immediately useful

Save an interim draft to 'zaina_response.md' with sections:
- Direct Answer
- Rationale (short)
(Citations optional; do not include a separate References section.)""",
        expected_output=(
            "A concise Markdown outline saved to 'zaina_response.md' with Direct Answer and Rationale sections (no References section)"
        ),
        agent=agent,
    )


def create_poet_task(agent: Agent, topic: str) -> Task:
    """Task for Poet: lightly lace the draft with poetic imagery."""
    return Task(
        description=f"""Refine the draft for: {topic}

Take the Synthesizer's draft (assume available) and:
1. Infuse subtle poetic rhythm and precise imagery in key lines
2. Keep clarity and brevity as the north star (no purple prose)
3. Maintain section structure; do not bloat length

Update 'zaina_response.md'.""",
        expected_output=(
            "A refined Markdown draft in 'zaina_response.md' with light poetic touches"
        ),
        agent=agent,
    )


def create_fun_task(agent: Agent, topic: str) -> Task:
    """Task for Fun & Relatable: add warmth, small jokes, and charm."""
    return Task(
        description=f"""Polish the final for: {topic}

Take the Poet's refined draft and:
1. Add small, inclusive jokes or light asides (1-3 max)
2. Ground recommendations with one relatable, everyday analogy
3. Keep tone warm, confident, and human

Finalize 'zaina_response.md'.""",
        expected_output=(
            "A personable, charming final saved as 'zaina_response.md'"
        ),
        agent=agent,
    )


# ----- Improvement function for NANDA -----

def create_zaina_improvement() -> Callable[[str], str]:
    """Create a CrewAI-powered Zaina swarm improvement function."""

    # LLM for all agents
    llm = ChatAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-haiku-20240307",
    )

    # Build agents
    creative_explorer = create_creative_explorer_agent(llm)
    synthesizer = create_synthesizer_agent(llm)
    poet = create_poet_agent(llm)
    fun_relatable = create_fun_relatable_agent(llm)

    def zaina_improvement(message_text: str) -> str:
        """Run the Zaina multi-agent workflow for the given message."""
        topic = message_text.strip() or "Designing delightful onboarding for a productivity app"

        try:
            # Build tasks
            explorer_task = create_explorer_task(creative_explorer, topic)
            synthesizer_task = create_synthesizer_task(synthesizer, topic)
            poet_task = create_poet_task(poet, topic)
            fun_task = create_fun_task(fun_relatable, topic)

            # Assemble crew
            crew = Crew(
                agents=[creative_explorer, synthesizer, poet, fun_relatable],
                tasks=[explorer_task, synthesizer_task, poet_task, fun_task],
                process=Process.sequential,
                verbose=True,
            )

            result = crew.kickoff()

            # If a file was written, append a short marker to the result
            final_result = str(result).strip()
            filename = "zaina_response.md"
            if os.path.exists(filename):
                final_result += f"\n\n[Saved to {filename}]"
            return final_result
        except Exception as e:
            print(f"Error in Zaina improvement: {e}")
            return topic

    return zaina_improvement


def main():
    """Main function to start the Zaina swarm agent."""

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set your ANTHROPIC_API_KEY environment variable")
        return

    # Serper is optional but recommended for explorer; warn if missing
    if not os.getenv("SERPER_API_KEY"):
        warnings.warn(
            "SERPER_API_KEY is not set. Explorer web search may be limited.",
            RuntimeWarning,
        )

    zaina_logic = create_zaina_improvement()
    nanda = NANDA(zaina_logic)

    print("Starting Zaina Swarm Agent with CrewAI...")
    print("Messages will be processed by the Zaina multi-agent workflow.")

    domain = os.getenv("DOMAIN_NAME", "localhost")

    if domain != "localhost":
        nanda.start_server_api(os.getenv("ANTHROPIC_API_KEY"), domain)
    else:
        nanda.start_server()


if __name__ == "__main__":
    main()
