"""departments.py — DeepAgent factory functions for LinguaFlow autonomous operations.

Creates six specialized department agents using create_deep_agent():
- Onboarding Agent: handles new student intake and profile setup
- Tutor Management Agent: manages tutor assignments and schedules
- Content Pipeline Agent: oversees lesson content creation and review
- QA Agent: runs quality checks across operations and flags anomalies
- Support Agent: triages and resolves student/tutor support requests
- Reporting Agent: generates dashboards and trend reports for leadership

Each agent loads SKILL.md files for domain knowledge, and the CompositeBackend
routes ephemeral working files to StateBackend and persistent records to
StoreBackend.

DeepAgents concepts demonstrated:
- create_deep_agent() with tools, skills, backend, and system_prompt
- CompositeBackend for hybrid ephemeral/persistent storage
- StateBackend (ephemeral) vs StoreBackend (persistent)
- Skills loading via SkillsMiddleware (automatic from skills= parameter)
- Dispatcher dict for dynamic agent instantiation by department name
"""

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.store import StoreBackend
from langchain_anthropic import ChatAnthropic
from langgraph.store.memory import InMemoryStore

from prompts import (
    CONTENT_PIPELINE_PROMPT,
    QA_PROMPT,
    REPORTING_PROMPT,
    STUDENT_ONBOARDING_PROMPT,
    SUPPORT_PROMPT,
    TUTOR_MANAGEMENT_PROMPT,
)
from tools import (
    CONTENT_PIPELINE_TOOLS,
    QA_TOOLS,
    REPORTING_TOOLS,
    STUDENT_ONBOARDING_TOOLS,
    SUPPORT_TOOLS,
    TUTOR_MANAGEMENT_TOOLS,
)

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

# Resolve the project root and skills directory so agents can load SKILL.md files
_PROJECT_DIR = Path(__file__).resolve().parent
SKILLS_DIR = _PROJECT_DIR / "skills"

# Always use the cheapest Anthropic model — this is a learning repo, not production
_MODEL = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)

# Shared InMemoryStore — persistent namespace across threads within a session
# (StoreBackend reads/writes here; StateBackend is per-thread ephemeral)
_store = InMemoryStore()

# LangSmith tags so every trace is filterable by project in the LangSmith UI
_TAGS = ["p8-autonomous-operations"]


# ---------------------------------------------------------------------------
# Composite backend factory
# ---------------------------------------------------------------------------

def create_composite_backend():
    """Create a CompositeBackend factory that routes file paths to the right storage.

    Routing rules:
    - /persistent/ → StoreBackend (survives across threads; cross-session records)
    - Default       → StateBackend (ephemeral; scoped to the current thread)

    DeepAgents concept: CompositeBackend lets a single agent transparently use
    multiple storage strategies. Ephemeral files disappear when a thread ends,
    while persistent files survive across sessions in _store.

    Returns:
        A factory callable ``(runtime) -> CompositeBackend`` that DeepAgents
        calls once per agent invocation to wire up the correct backends.
    """
    def factory(runtime):
        return CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/persistent/": StoreBackend(runtime),
            },
        )
    return factory


# ---------------------------------------------------------------------------
# Department agent factories
# ---------------------------------------------------------------------------

def create_onboarding_agent():
    """Create the Student Onboarding department agent.

    Responsible for processing new student applications, running eligibility
    checks, collecting placement-test scores, and provisioning accounts.
    Writes intake records to /persistent/students/ so they survive across
    invocations and are readable by downstream agents.
    """
    return create_deep_agent(
        name="student-onboarding",
        model=_MODEL,
        system_prompt=STUDENT_ONBOARDING_PROMPT,
        tools=STUDENT_ONBOARDING_TOOLS,
        skills=[str(SKILLS_DIR) + "/"],
        backend=create_composite_backend(),
        store=_store,
    )


def create_tutor_agent():
    """Create the Tutor Management department agent.

    Handles tutor availability queries, session scheduling, workload
    balancing, and performance reviews. Reads student profiles from
    /persistent/students/ to make informed assignment decisions.
    """
    return create_deep_agent(
        name="tutor-management",
        model=_MODEL,
        system_prompt=TUTOR_MANAGEMENT_PROMPT,
        tools=TUTOR_MANAGEMENT_TOOLS,
        skills=[str(SKILLS_DIR) + "/"],
        backend=create_composite_backend(),
        store=_store,
    )


def create_content_agent():
    """Create the Content Pipeline department agent.

    Oversees lesson creation, review, versioning, and publishing. Works with
    the curriculum catalogue stored in /persistent/content/ and flags content
    that requires human editorial review before going live.
    """
    return create_deep_agent(
        name="content-pipeline",
        model=_MODEL,
        system_prompt=CONTENT_PIPELINE_PROMPT,
        tools=CONTENT_PIPELINE_TOOLS,
        skills=[str(SKILLS_DIR) + "/"],
        backend=create_composite_backend(),
        store=_store,
    )


def create_qa_agent():
    """Create the Quality Assurance department agent.

    Runs automated checks across operational data: session completion rates,
    content accuracy, support ticket resolution times, and compliance with
    LinguaFlow's quality standards. Escalates anomalies to the orchestrator.
    """
    return create_deep_agent(
        name="quality-assurance",
        model=_MODEL,
        system_prompt=QA_PROMPT,
        tools=QA_TOOLS,
        skills=[str(SKILLS_DIR) + "/"],
        backend=create_composite_backend(),
        store=_store,
    )


def create_support_agent():
    """Create the Support department agent.

    Triages incoming support tickets, resolves common issues autonomously
    (password resets, scheduling conflicts, billing queries), and escalates
    complex cases to human staff with a full context summary.
    """
    return create_deep_agent(
        name="support",
        model=_MODEL,
        system_prompt=SUPPORT_PROMPT,
        tools=SUPPORT_TOOLS,
        skills=[str(SKILLS_DIR) + "/"],
        backend=create_composite_backend(),
        store=_store,
    )


def create_reporting_agent():
    """Create the Reporting department agent.

    Aggregates metrics from all other departments, produces weekly/monthly
    trend reports, and surfaces key KPIs (student progress, tutor utilisation,
    content engagement) as structured data for leadership dashboards.
    """
    return create_deep_agent(
        name="reporting",
        model=_MODEL,
        system_prompt=REPORTING_PROMPT,
        tools=REPORTING_TOOLS,
        skills=[str(SKILLS_DIR) + "/"],
        backend=create_composite_backend(),
        store=_store,
    )


# ---------------------------------------------------------------------------
# Dispatcher dict — maps department names to their factory functions
# ---------------------------------------------------------------------------

# The orchestrator (graph.py) uses this dict to instantiate the correct agent
# for a given department task without needing to import each factory directly.
DEPARTMENT_AGENTS = {
    "student_onboarding": create_onboarding_agent,
    "tutor_management": create_tutor_agent,
    "content_pipeline": create_content_agent,
    "quality_assurance": create_qa_agent,
    "support": create_support_agent,
    "reporting": create_reporting_agent,
}
