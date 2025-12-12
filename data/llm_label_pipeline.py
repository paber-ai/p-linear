"""Interactive CLI for LLM-based auto-labeling into the p-linear JSONL format.

This script is intended to run from the p-linear project root, e.g.:

    python -m data.llm_label_pipeline

It will interactively ask for:
- A dataset identifier (for your own tracking, e.g. an HF dataset ID).
- A local data file path (CSV / JSON / Parquet) to load with pandas.
- A comma-separated list of columns whose contents should be concatenated
  into the input `text` for p-linear.

For each row, it calls a LLM to analyze the text and predict the
p-linear heads:

    simple, complex, needs_tools, needs_memory, high_risk, code_like

It writes one JSON object per line to a JSONL file (default: data/data.jsonl
relative to the project root) with the canonical schema expected by the
training code:

    {"text": "...", "labels": {"simple": 0/1, ...}}

Note: this script uses the official Ollama Python SDK with custom structured
outputs, You are responsible for
downloading any HF/Kaggle datasets locally (e.g., as CSV/JSON/Parquet) before
running the script.
"""

from __future__ import annotations
from typing import List
from pydantic import BaseModel


SYSTEM_PROMPT = """
    # TASK
    You are a routing labeler for a meta-reasoning system.
    Your mission is a part of a platform called Paber AI, a platform that helps students to learn and understand the world better with AI and tools, and Paber AI models prefer to always use tools to search for the accurate real-time information from the web, put that in the `needs_tools`, and for most cases, its better to set `needs_tools=1`.

    Given a SINGLE user query in the field "text", you have to use the same language as the input (no translation), but if the field was in conversion messages JSON, you have to extract the user query from it, 
    and you may optionally look at other columns such as "instruction", "input", "prompt", "context", "category",
    and "output" only as extra hints. Sometimes the row contains full conversation messages in JSON; in that case,
    you must extract the main user query from the full conversation messages, not as JSON, you have to extract it as a string and treat that as "text".

    For example, you got this conv: [{ "text": "Hi", "type": "text" }] -> only take "Hi" as the user query.
    
    From this information you must assign SIX binary
    routing flags: "simple", "complex", "needs_tools", "needs_memory", "high_risk", "code_like".

    # GENERAL PRINCIPLES:
    - Always output a firm 0 or 1 for EVERY flag. Never omit a flag.
    - At least ONE of {"simple", "complex"} MUST be 1. They MAY both be 1, but they must not both be 0.
    - The user query in "text" is primary. Use other columns only to clarify intent, domain, and difficulty.
    - If you are unsure about "needs_memory", prefer 0 (conservative). For needs_tools, see the strong rules below.

    # DEFINITIONS:
    1- `simple`: if the query can likely be answered with a short, single-step response or a small number of obvious
        steps (e.g., a fact recall that does NOT require fresh/real-time data, a short definition, a small classification,
        rewriting one sentence, extracting a small piece of information from a short context). 0 if it clearly requires
        extended reasoning, multi-step planning, or careful reading of a long context.

    2- `complex`: if the query likely requires multi-step reasoning, planning, comparison, or processing long/structured
        information (e.g., summarization, multi-part instructions, multi-paragraph generation, long information-extraction,
        reasoning about math/logic with several steps). 0 for truly one-shot, straightforward questions.

    3- `needs_tools`: if solving the query is complex OR writing about any topic OR not simple OR the query is not clear that need more context OR the user query is needs deep info OR the user query is not about regular questions OR clearly involves coding, technical, scientific, mathematical,
        data-analysis, or real-world information that should be looked up or computed using external tools or web search.
        This includes: any request for real-time or up-to-date information ("current", "latest", "today", "specific recent
        years or dates"), anything that explicitly or implicitly needs web search or browsing, looking up statistics,
        rankings, prices, historical, comparison, populations, sports scores, stock/crypto/FX prices, or other external lists and databases;
        requests to generate detailed reports, research summaries, dashboards, or visualizations that normally rely on data;
        requests to call APIs, query databases, run code, use calculators, spreadsheets, plotting tools, or other external
        systems; and most programming, scripting, or engineering tasks. If the task is about coding or technical workflows,
        for most of the time you should set needs_tools=1. if you're gonna put complex=1, most of the time you should set needs_tools=1. Otherwise 0.
        Summary: if user query is not simple, or you have putted `complex=1` or `code_like=1`, put `needs_tools=1`, because mostly it will need to search web or calling tools, otherwise, put needs_tools=1.

    4- `needs_memory`: if the query clearly depends on long-term or large context (long documents, prior conversation,
        user history) OR on a long "context" column supplied with the row. For example, long articles to summarize, long
        legal or technical documents to analyze, multi-turn dialogues to reason about, or questions that explicitly say to
        "consider all of the above conversation" or similar. Also set needs_memory=1 when the answer must integrate
        information spread across many paragraphs or turns. Otherwise 0.

    5- `high_risk`: if the query touches safety-sensitive domains such as self-harm, suicide, serious medical advice,
        mental health crises, diagnosic or treatment choices, legal advice about serious matters, hate/abuse/harassment,
        extremist content, terrorism, serious financial scams or high-stakes investment decisions, instructions for weapons
        or serious crimes, or other clearly sensitive or harmful topics. 0 otherwise.

    6- `code_like`: if the query is primarily about source code, programming languages, libraries, APIs, error messages,
        configuration files, logs, stack traces, shell commands, SQL, or contains substantial code snippets or strongly
        implied coding tasks ("write a Python function...", "debug this error...", "explain this code snippet..."), even if
        the word "code" is not explicitly used. 0 otherwise.

    # INTERACTIONS AND EDGE CASES:
    - A query may be both simple=1 and complex=1 if it is short but still requires non-trivial reasoning or reading
        non-trivial context.

    - needs_tools=1 for ANY query that clearly benefits from web search, APIs, databases, real-time or up-to-date data,
        external computation, plotting, or specialized technical tools (including programming, data science, or scientific
        workflows). When in doubt for technical, scientific, or critical decision-making tasks, prefer needs_tools=1.

    - needs_memory=1 is appropriate when the answer clearly depends on information supplied in "context" or on earlier
        parts of a long story or conversation, or when the user explicitly refers back to many earlier turns.

    - high_risk=1 should be used liberally for self-harm, serious medical or legal advice, hate, harassment, extremist
        politics, or clearly harmful financial or criminal topics.

    - code_like=1 whenever the user is asking to write, debug, refactor, or analyze code, logs, traces, configuration,
        or other developer-centric artifacts.

    # EXAMPLES (do NOT copy text; they are only to illustrate labels):
    - "What is a polygon?" -> simple=1, complex=0, needs_tools=0, needs_memory=0, high_risk=0, code_like=0.
    - "Translate 'hello' to Spanish." -> simple=1, complex=0, needs_tools=0, needs_memory=0, high_risk=0, code_like=0.
    - "Summarize the following 3-paragraph article about key lime pie" with a long "context" -> simple=0, complex=1, needs_tools=0, needs_memory=1, high_risk=0, code_like=0.
    - "Give me the top 5 golf equipment company names." with no context -> simple=1, complex=0, needs_tools=1, needs_memory=0, high_risk=0, code_like=0 (requires up-to-date web search).
    - "What is the current price of Bitcoin in USD?" -> simple=1, complex=0, needs_tools=1, needs_memory=0, high_risk=0, code_like=0 (real-time external data).
    - "Write a Python function that sorts a list of users by age." -> simple=0, complex=1, needs_tools=1, needs_memory=0, high_risk=0, code_like=1 (coding + external execution environment).
    - [ {"text": "How to create a next.js v16 app with better-auth", "type": "text"} ] (Note: this is a full conversation, so, you need to return it "text" field as "How to create a next.js v16 app with better-auth") -> simple=0, complex=1, needs_tools=1, needs_memory=0, high_risk=0, code_like=1.
    - "Explain what this error means and how to fix it: TypeError: unsupported operand type(s) for +: int and str" -> simple=0, complex=1, needs_tools=1, needs_memory=0, high_risk=0, code_like=1.
    - "I want to hurt myself. What should I do?" -> simple=0, complex=1, needs_tools=0, needs_memory=0, high_risk=1, code_like=0.
    - "Compare the long essay above with this second essay and tell me which is more persuasive." with long contexts -> simple=0, complex=1, needs_tools=0, needs_memory=1, high_risk=0, code_like=0.
    - "Design an experiment to test whether a new drug is effective against a disease, using current clinical guidelines." -> simple=0, complex=1, needs_tools=1, needs_memory=0, high_risk=1, code_like=0.
    - [ {"text": "What is better? Supabase or Neon?", "type": "text"} ] (Note: this is a full conversation, so, you need to return it "text" field as "What is better? Supabase or Neon?") -> simple=0, complex=1, needs_tools=1, needs_memory=0, high_risk=0, code_like=1.

    # OUTPUT FORMAT (IMPORTANT):
    - You MUST return a single JSON object, not an array or list.
    - The JSON object MUST have exactly these top-level keys: simple, complex, needs_tools, needs_memory, high_risk, code_like.
    - Each of those keys MUST have value 0 or 1.
    - You MAY also include an optional `metadata` object with any additional fields you find useful (e.g., language, context, scores, extra info, etc.), but the six label keys must be present at the top level.
    - Do NOT include explanations, natural language commentary, or any other text outside of the JSON object.
"""


COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"


P_LINEAR_HEADS: List[str] = [
    "simple",
    "complex",
    "needs_tools",
    "needs_memory",
    "high_risk",
    "code_like",
]


class PLinearScores(BaseModel):
    """Confidence scores for each p-linear head (0.0-1.0)."""

    simple: float
    complex: float
    needs_tools: float
    needs_memory: float
    high_risk: float
    code_like: float


class PLinearMetadata(BaseModel):
    """Auxiliary metadata predicted alongside labels.

    - language: short language code or description (e.g., "en", "ar").
    - context: optional free-form notes about why the labels were chosen.
    - scores: per-head confidence scores.
    """

    language: str | None = None
    context: str | None = None
    scores: PLinearScores | None = None


class PLinearOutput(BaseModel):
    """Structured output model for p-linear heads and metadata."""

    simple: int
    complex: int
    needs_tools: int
    needs_memory: int
    high_risk: int
    code_like: int
    metadata: PLinearMetadata | None = None


def _prompt_for_missing(prompt: str, current: str | None = None) -> str:
    if current:
        return current

    value = input(prompt).strip()

    if not value:
        raise SystemExit("Aborted: required input was empty.")

    return value
