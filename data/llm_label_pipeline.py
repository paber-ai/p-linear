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
import ast
import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List
import pandas as pd
from ollama import chat
from pydantic import BaseModel
from tqdm.auto import tqdm


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

    4- `needs_memory`: if the query clearly depends on long-term or large context (long documents, prior conversation OR you were asked to summarize a text or an essay or a report,
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
    - [ {"text": "How to create a next.js v16 app with better-auth", "type": "text"} ] (Note: this is a full conversation, so, you need to return "text" field as "How to create a next.js v16 app with better-auth") -> simple=0, complex=1, needs_tools=1, needs_memory=0, high_risk=0, code_like=1.
    - "Explain what this error means and how to fix it: TypeError: unsupported operand type(s) for +: int and str" -> simple=0, complex=1, needs_tools=1, needs_memory=0, high_risk=0, code_like=1.
    - "I want to hurt myself. What should I do?" -> simple=0, complex=1, needs_tools=0, needs_memory=0, high_risk=1, code_like=0.
    - "Compare the long essay above with this second essay and tell me which is more persuasive." with long contexts -> simple=0, complex=1, needs_tools=0, needs_memory=1, high_risk=0, code_like=0.
    - "Design an experiment to test whether a new drug is effective against a disease, using current clinical guidelines." -> simple=0, complex=1, needs_tools=1, needs_memory=0, high_risk=1, code_like=0.
    - [ {"text": "What is better? Supabase or Neon?", "type": "text"} ] (Note: this is a full conversation, so, you need to return "text" field as "What is better? Supabase or Neon?") -> simple=0, complex=1, needs_tools=1, needs_memory=0, high_risk=0, code_like=1.

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


def load_table(path_str: str, columns: Iterable[str]) -> pd.DataFrame:
    """Load a tabular dataset with pandas based on file extension.

    The path may point to a local file or a remote URI supported by pandas/
    fsspec (e.g. hf://datasets/... for Hugging Face Hub datasets).
    """

    suffix = Path(path_str).suffix.lower()
    usecols = list(columns) if columns else None

    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"

        return pd.read_csv(path_str, usecols=usecols, sep=sep)

    if suffix in {".json", ".jsonl"}:
        # Assume JSON lines for .jsonl, normal JSON array for .json.
        lines = suffix == ".jsonl"

        return pd.read_json(path_str, lines=lines)

    if suffix in {".parquet"}:
        return pd.read_parquet(path_str, columns=usecols)

    raise SystemExit(
        f"Unsupported file extension '{suffix}'. Use CSV, TSV, JSON, JSONL, or Parquet."
    )


def build_text(row: pd.Series, columns: List[str]) -> str:
    """Extract the primary user query text from the first column.

    All additional columns are treated as auxiliary context and are not
    concatenated into the training `text`; they are instead passed to the LLM
    as structured context and stored under metadata.
    """

    if not columns:
        return ""

    first = columns[0]
    value = row.get(first, "")

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""

    except Exception:
        pass

    def _maybe_parse_json_like(raw: object) -> object | None:
        if isinstance(raw, (list, dict)):
            return raw

        if not isinstance(raw, str):
            return None

        stripped = raw.strip()

        if not stripped:
            return None

        if stripped[0] not in "[{":
            return None

        try:
            return json.loads(stripped)

        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(stripped)

        except (ValueError, SyntaxError):
            return None

    def _extract_text_from_conversation_payload(payload: object) -> str | None:
        messages: object = payload

        if isinstance(payload, dict):
            for key in ("messages", "conversation", "conversations"):
                if key in payload:
                    messages = payload.get(key)

                    break
            else:
                messages = [payload]

        if not isinstance(messages, list):
            return None

        candidates: list[tuple[str, str]] = []

        for item in messages:
            if isinstance(item, dict):
                role = (
                    item.get("role")
                    or item.get("speaker")
                    or item.get("from")
                    or item.get("type")
                )

                text_val = item.get("content")

                if text_val is None:
                    text_val = item.get("text")

                if text_val is None:
                    text_val = item.get("value")

                if isinstance(text_val, list):
                    parts: list[str] = []

                    for part in text_val:
                        if isinstance(part, dict):
                            part_text = part.get("text") or part.get("content")

                            if isinstance(part_text, str):
                                parts.append(part_text)

                        elif isinstance(part, str):
                            parts.append(part)

                    if parts:
                        text_val = "".join(parts)

                if isinstance(text_val, dict):
                    part_text = (
                        text_val.get("text")
                        if isinstance(text_val.get("text"), str)
                        else None
                    )

                    if part_text is not None:
                        text_val = part_text

                if not isinstance(text_val, str):
                    continue

                role_str = str(role).lower().strip() if role is not None else ""
                candidates.append((role_str, text_val))

            elif isinstance(item, str):
                candidates.append(("", item))

        if not candidates:
            return None

        user_like = [
            text
            for role, text in candidates
            if role in {"user", "human", "customer"} or role.startswith("user")
        ]

        if user_like:
            for text in reversed(user_like):
                if text.strip():
                    return text.strip()

            return user_like[-1].strip()

        for _, text in candidates:
            if text.strip():
                return text.strip()

        return candidates[0][1].strip()

    parsed = _maybe_parse_json_like(value)

    if parsed is not None:
        extracted = _extract_text_from_conversation_payload(parsed)

        if extracted:
            return extracted

    return str(value)


def call_ollama(
    text: str,
    extra_columns: Dict[str, object] | None = None,
) -> tuple[Dict[str, int], Dict[str, object] | None]:
    """Call LLM in JSON mode and extract p-linear labels + optional metadata.

    We use `format=PLinearOutput.model_json_schema()` so the model must return a
    single JSON object. Expected top-level keys:

        simple, complex, needs_tools, needs_memory, high_risk, code_like

    each as 0/1 integers. An optional `metadata` object may also be present and
    is passed through as-is for the caller to merge with dataset-specific info.
    """

    if extra_columns:
        extras_lines = [f"- {k}: {v}" for k, v in extra_columns.items()]
        extras_block = "\n".join(extras_lines)
        user_prompt = (
            "User query to label:\n\n"
            f"{text}\n\n"
            "Additional columns (for context only):\n"
            f"{extras_block}"
        )

    else:
        user_prompt = f"User query to label:\n\n{text}"

    resp = chat(
        model="gemma3:12b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format=PLinearOutput.model_json_schema(),
        options={"temperature": 0},
    )

    try:
        content = resp.message.content

    except Exception as exc:
        raise RuntimeError(f"Unexpected Ollama response shape: {resp!r}") from exc

    try:
        obj = json.loads(content)

    except json.JSONDecodeError:
        obj = None

    def _parse_from_json_obj(
        o: object,
    ) -> tuple[Dict[str, int], Dict[str, object] | None]:
        if not isinstance(o, dict):
            raise ValueError("Top-level JSON is not an object")

        missing_heads = [h for h in P_LINEAR_HEADS if h not in o]

        if missing_heads:
            raise ValueError(f"missing heads {missing_heads}")

        labels_dict: Dict[str, int] = {}

        for head in P_LINEAR_HEADS:
            value = o[head]

            try:
                labels_dict[head] = int(value)

            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"head {head!r} not int-like: {value!r} in {o!r}"
                ) from exc

        meta = o.get("metadata")

        if not isinstance(meta, dict):
            meta = None

        return labels_dict, meta

    def _flatten_to_text(o: object) -> str:
        if isinstance(o, dict):
            return json.dumps(o, ensure_ascii=False)

        if isinstance(o, list):
            parts: List[str] = []

            for item in o:
                parts.append(_flatten_to_text(item))

            return "\n".join(parts)

        return str(o)

    def _parse_from_text(text_value: str) -> Dict[str, int]:
        labels_dict: Dict[str, int] = {}

        for head in P_LINEAR_HEADS:
            # Look for patterns like simple=1 or "simple": 1
            patterns = [
                rf"{head}\s*[:=]\s*([01])",
                rf'"{head}"\s*:\s*([01])',
            ]

            value: int | None = None

            for pat in patterns:
                m = re.search(pat, text_value)

                if m:
                    value = int(m.group(1))

                    break

            if value is None:
                raise ValueError(f"could not find label for {head!r} in text")

            labels_dict[head] = value

        return labels_dict

    # Try strict JSON-object parsing first.
    if obj is not None:
        try:
            return _parse_from_json_obj(obj)

        except ValueError:
            # Fall back to text-based parsing below.
            pass

    # Fallback: attempt to recover labels from textual content when the model
    # returns a JSON array of reasoning strings or other unexpected structures.
    try:
        labels_only = _parse_from_text(
            content if obj is None else _flatten_to_text(obj)
        )

    except ValueError as exc:
        raise RuntimeError(
            f"Structured output could not be parsed for all heads: {exc}. Raw content: {content!r}"
        ) from exc

    return labels_only, None


def refine_labels(
    text: str,
    labels: Dict[str, int],
    extra_columns: Dict[str, object] | None = None,
) -> Dict[str, int]:
    """Apply simple rule-based refinements on top of the LLM labels.

    This acts like a tiny weak-supervision engine that snaps labels to
    consistent patterns based on the query text and known dataset metadata
    such as `category` and `context`.
    """

    refined: Dict[str, int] = dict(labels)

    category: str | None = None
    context: str | None = None

    if extra_columns:
        raw_cat = extra_columns.get("category")

        if raw_cat is not None:
            category = str(raw_cat).strip().lower()

        raw_ctx = extra_columns.get("context")

        if raw_ctx is not None:
            context = str(raw_ctx)

    text_lower = text.lower()
    ctx_len = len(context.split()) if context else 0
    text_len = len(text.split()) if text else 0

    # Category-based patterns (tuned for Dolly-style data).
    if category in {"open_qa", "closed_qa", "general_qa"}:
        if text_len <= 40 and ctx_len <= 80:
            refined["simple"] = 1
            refined["complex"] = 0

    if category in {"information_extraction"}:
        refined["complex"] = 1

        if ctx_len > 50:
            refined["needs_memory"] = 1

    if category in {"summarization"}:
        refined["complex"] = 1

        if ctx_len > 50:
            refined["needs_memory"] = 1

    if category in {"classification"}:
        refined["simple"] = 1
        refined["complex"] = 0

        if ctx_len < 100:
            refined["needs_memory"] = 0

    if category in {"brainstorming", "creative_writing"}:
        refined["complex"] = 1

    # Text-pattern rules for needs_tools.
    if any(
        kw in text_lower
        for kw in (
            "top 5",
            "top five",
            "top 10",
            "top ten",
            "top three",
            "top 3",
            "list of the best",
            "best companies",
            "current price",
            "current rate",
            "exchange rate",
            "latest version",
            "as of today",
            "right now",
        )
    ):
        refined["needs_tools"] = 1

    # Text-pattern rules for code_like.
    if any(
        kw in text_lower
        for kw in (
            "python",
            "javascript",
            "typescript",
            "rust",
            "golang",
            "java",
            "c++",
            "c#",
            "stack trace",
            "stacktrace",
            "exception",
            "traceback",
            "segmentation fault",
            "nullpointer",
            "def ",
            "class ",
            "function(",
            "async ",
            "await ",
        )
    ):
        refined["code_like"] = 1

    # Safety / high_risk keywords.
    if any(
        kw in text_lower
        for kw in (
            "suicide",
            "kill myself",
            "kill myself",
            "self-harm",
            "self harm",
            "overdose",
            "cut myself",
            "want to die",
            "depressed",
            "panic attack",
            "emergency room",
            "diagnose me",
            "prescription",
            "lawsuit",
            "sue ",
            "illegal",
            "crime",
            "terrorist",
            "extremist",
            "bomb",
            "racial slur",
            "racist joke",
            "hate speech",
        )
    ):
        refined["high_risk"] = 1

    # Ensure at least one of simple/complex is 1.
    simple_val = int(refined.get("simple", 0))
    complex_val = int(refined.get("complex", 0))

    if not (simple_val or complex_val):
        if text_len <= 40:
            refined["simple"] = 1

        else:
            refined["complex"] = 1

    # Clamp all heads to 0/1 integers.
    for head in P_LINEAR_HEADS:
        refined[head] = 1 if int(refined.get(head, 0)) != 0 else 0

    return refined


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive LLM-based labeling pipeline that converts a local data "
            "file into the canonical p-linear JSONL format using Ollama."
        ),
    )

    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Identifier for this dataset (e.g. HF ID); used only for logging.",
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to a local CSV/TSV/JSON/JSONL/Parquet file to label.",
    )

    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help=(
            "Comma-separated list of columns. The first column is treated as the "
            "user query (text); subsequent columns are passed as context and "
            "stored under metadata.source_columns. Example: question,context."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/data.jsonl",
        help=(
            "Path to write the labeled JSONL file. Defaults to data/data.jsonl "
            "relative to the project root."
        ),
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to process (for testing).",
    )

    args = parser.parse_args()

    dataset_id = _prompt_for_missing("Dataset ID: ", args.dataset_id)

    input_path_str = _prompt_for_missing(
        "Local data file path (CSV/JSON/Parquet): ", args.input_path
    )

    columns_str = _prompt_for_missing(
        "Columns to use (comma-separated, e.g. question,context): ", args.columns
    )

    columns = [c.strip() for c in columns_str.split(",") if c.strip()]

    if not columns:
        raise SystemExit("No columns specified.")

    input_path = Path(input_path_str)
    is_remote = input_path_str.startswith(("hf://", "s3://", "http://", "https://"))

    if not is_remote and not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    df = load_table(input_path_str, columns)

    if args.max_rows is not None:
        max_rows = min(args.max_rows, len(df))

        if max_rows <= 0:
            raise SystemExit("max-rows must be positive.")

        df = df.sample(n=max_rows, replace=False).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_rows = len(df)

    # Target fraction of time spent actively using the GPU (e.g., 0.8 = 80%).
    target_duty_cycle = 0.8
    # Number of rest blocks to spread over the dataset (up to 10).
    num_rest_blocks = 10 if num_rows >= 10 else 1
    rows_per_rest = max(1, num_rows // num_rest_blocks)

    # Moving average of per-row LLM call duration (seconds).
    avg_call_seconds = 0.0
    timed_rows = 0

    print(f"{COLOR_GREEN}Loaded {num_rows} rows from {input_path_str}.{COLOR_RESET}")
    print(f"{COLOR_CYAN}Labeling with 'gemma3:12b' model...{COLOR_RESET}")

    with output_path.open("a", encoding="utf-8") as out_f:
        processed = 0
        failed = 0

        progress = tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"{COLOR_CYAN}Labeling{COLOR_RESET}",
            unit="row",
        )

        for row_idx, row in progress:
            text = build_text(row, columns)

            if not text.strip():
                failed += 1
                progress.set_postfix(processed=processed, failed=failed)

                continue

            # Build a mapping of non-query columns to pass as context and store
            # under metadata.
            extra_columns: Dict[str, object] = {}

            for col in columns[1:]:
                if col in row and pd.notna(row[col]):
                    extra_columns[col] = row[col]

            start_ts = time.time()

            try:
                labels, metadata = call_ollama(
                    text,
                    extra_columns=extra_columns or None,
                )

                original_labels = dict(labels)
                labels = refine_labels(text, labels, extra_columns or None)
                processed += 1

                # Update moving average of per-row latency.
                row_duration = time.time() - start_ts
                timed_rows += 1

                if timed_rows == 1:
                    avg_call_seconds = row_duration

                else:
                    avg_call_seconds = (
                        avg_call_seconds * (timed_rows - 1) + row_duration
                    ) / timed_rows

            except Exception as exc:
                failed += 1

                print(
                    f"{COLOR_YELLOW}[WARN]{COLOR_RESET} LLM call failed at row {row_idx}: {exc}"
                )

                progress.set_postfix(processed=processed, failed=failed)

                continue

            obj: Dict[str, object] = {"text": text, "labels": labels}

            combined_meta: Dict[str, object] = {}

            if metadata:
                combined_meta.update(metadata)

            if extra_columns:
                combined_meta.setdefault("source_columns", {}).update(
                    {k: str(v) for k, v in extra_columns.items()}
                )

            if dataset_id:
                combined_meta.setdefault("dataset_id", dataset_id)

            # Record rule-based refinement info to make debugging easier.
            if processed > 0:
                combined_meta.setdefault("rule_engine", {})
                rule_info = combined_meta["rule_engine"]

                if isinstance(rule_info, dict):
                    rule_info.setdefault("applied", True)
                    rule_info.setdefault("original_labels", original_labels)

            if combined_meta:
                obj["metadata"] = combined_meta

            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # GPU rest schedule: after each block of rows_per_rest processed rows,
            # sleep long enough to maintain the target duty cycle.
            if processed % rows_per_rest == 0 and avg_call_seconds > 0:
                # For a block of rows_per_rest rows, active time is approximately:
                #   T_active_block = rows_per_rest * avg_call_seconds
                # To achieve a duty cycle D, required rest time per block is:
                #   T_rest_block = T_active_block * (1 - D) / D
                active_block = rows_per_rest * avg_call_seconds

                rest_block = (
                    active_block * (1.0 - target_duty_cycle) / target_duty_cycle
                )

                if rest_block > 0:
                    print(
                        f"{COLOR_YELLOW}Resting GPU for {rest_block:.1f}s after {processed} rows...{COLOR_RESET}"
                    )

                    time.sleep(rest_block)

            progress.set_postfix(processed=processed, failed=failed)

    print(
        f"{COLOR_GREEN}Done.{COLOR_RESET} Wrote labeled data to {output_path}. "
        f"Processed={processed}, failed={failed}."
    )


if __name__ == "__main__":
    main()
