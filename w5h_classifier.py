"""
QUEST — Module 1: W5H Dimensional Classifier
=============================================
Classifies natural language database queries along six semantic dimensions:
WHO, WHAT, WHERE, WHEN, WHY, HOW.

Each dimension carries:
  - active       : bool   — whether the dimension is present in the query
  - evidence     : str    — the phrase(s) from the query that triggered it
  - confidence   : float  — model confidence in [0.0, 1.0]
  - warning      : str    — populated for WHY / HOW frontier dimensions

Backed by a local Ollama instance (OpenAI-compatible endpoint).
Recommended models (4-bit, fit in 8–16 GB VRAM):
  - qwen2.5:7b-instruct-q4_K_M   (8 GB)
  - qwen2.5:14b-instruct-q4_K_M  (16 GB)
  - llama3.1:8b-instruct-q4_K_M  (8 GB, alternative)
"""

from __future__ import annotations

import json
import logging
import textwrap
import warnings
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("quest.w5h")

# ---------------------------------------------------------------------------
# Pydantic schema — one DimensionResult per W5H dimension
# ---------------------------------------------------------------------------

class DimensionResult(BaseModel):
    """Classification result for a single W5H dimension."""

    active: bool = Field(
        description="True if this dimension is present in the query."
    )
    evidence: str = Field(
        description=(
            "The phrase or clause from the query that triggered this dimension. "
            "Empty string if active is False."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score in [0.0, 1.0].",
    )
    warning: Optional[str] = Field(
        default=None,
        description=(
            "Populated only for WHY and HOW when active. "
            "Explains why this is a frontier dimension."
        ),
    )

    @field_validator("evidence")
    @classmethod
    def evidence_empty_when_inactive(cls, v: str, info) -> str:
        # Pydantic v2: info.data contains already-validated fields
        active = info.data.get("active", True)
        if not active and v:
            # Silently clear evidence when dimension is inactive
            return ""
        return v


class W5HProfile(BaseModel):
    """Full W5H dimensional profile for a single natural language query."""

    query: str = Field(description="The original natural language query.")
    who: DimensionResult
    what: DimensionResult
    where: DimensionResult
    when: DimensionResult
    why: DimensionResult
    how: DimensionResult

    @model_validator(mode="after")
    def inject_frontier_warnings(self) -> "W5HProfile":
        """
        Post-validation: ensure WHY and HOW carry explicit warnings
        whenever they are active, regardless of what the model returned.
        These are 'frontier' dimensions that often require reasoning
        beyond standard SQL semantics (see QUEST paper §2.1).
        """
        if self.why.active and not self.why.warning:
            self.why.warning = (
                "WHY is a frontier dimension. Causal/explanatory constraints "
                "often require inference that SQL semantics do not natively support "
                "(e.g. foreign-key chains or cross-schema linkages). "
                "Verify that this constraint can be grounded in a recorded field."
            )
        if self.how.active and not self.how.warning:
            subtype = "mechanistic" if "mechanism" in (self.how.evidence or "").lower() else "quantitative"
            if subtype == "mechanistic":
                self.how.warning = (
                    "HOW (mechanistic) is a frontier dimension. Queries asking "
                    "'by what process/pathway' an outcome occurs are not expressible "
                    "in standard SQL. This may require capabilities beyond current "
                    "query systems."
                )
            else:
                self.how.warning = (
                    "HOW detected. If this reduces to HOW MANY (COUNT/AVG/etc.) "
                    "it is expressible in standard SQL. If it asks for a mechanism "
                    "or procedural pathway, it is a frontier dimension."
                )
        return self

    def active_dimensions(self) -> list[str]:
        """Return names of all active dimensions."""
        return [
            dim for dim in ("who", "what", "where", "when", "why", "how")
            if getattr(self, dim).active
        ]

    def frontier_warnings(self) -> dict[str, str]:
        """Return {dimension: warning} for all active frontier dimensions."""
        out = {}
        for dim in ("why", "how"):
            result: DimensionResult = getattr(self, dim)
            if result.active and result.warning:
                out[dim.upper()] = result.warning
        return out

    def summary(self) -> str:
        """Human-readable one-line summary."""
        active = self.active_dimensions()
        dims = ", ".join(d.upper() for d in active) if active else "none"
        return f"Active dimensions: [{dims}]"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are a semantic classifier for natural language database queries.
Your task is to classify a query along six dimensions (WHO, WHAT, WHERE, WHEN, WHY, HOW)
as defined by the QUEST framework.

Dimension definitions:
  WHO   — Primary entities represented by records (people, orgs, objects).
           Manifests as primary-key or foreign-key references.
           Examples: "customers who", "patients aged over 65", "employees in".

  WHAT  — Record properties, values, states, or calculated fields.
           Manifests as within-record field values or derived attributes.
           Examples: "with revenue over $1M", "rated 5 stars", "colored blue".

  WHERE — Location, scope, physical or categorical boundaries.
           Manifests as geographic or categorical fields.
           Examples: "in California", "within department", "in the ICU".

  WHEN  — Temporal constraints: time points, periods, sequences, durations.
           Manifests as timestamps, date ranges, or temporal relationships.
           NOTE: if the temporal value is derived from another entity
           (e.g. "before the last departure of flight AA100"), mark it active
           and include "WHO-anchored" in the evidence string.
           Examples: "last quarter", "before 2020", "within 30 days of admission".

  WHY   — Causal or explanatory relationships between entities.
           Manifests as foreign-key chains or cross-schema linkages.
           Examples: "readmitted due to surgical complications",
                     "canceled because of stockouts".

  HOW   — Methods, processes, mechanisms, or procedural pathways.
           Also active when the query asks HOW MANY (COUNT/AVG/etc.).
           Examples: "admitted through the emergency department",
                     "by what route", "readmission rate by treatment arm".

Output format — you MUST return ONLY a valid JSON object, no markdown, no explanation:
{
  "who":   {"active": bool, "evidence": "...", "confidence": float},
  "what":  {"active": bool, "evidence": "...", "confidence": float},
  "where": {"active": bool, "evidence": "...", "confidence": float},
  "when":  {"active": bool, "evidence": "...", "confidence": float},
  "why":   {"active": bool, "evidence": "...", "confidence": float},
  "how":   {"active": bool, "evidence": "...", "confidence": float}
}

Rules:
- confidence is a float in [0.0, 1.0] reflecting how certain you are.
- evidence is the exact phrase from the query that triggered the dimension.
  Use an empty string "" when active is false.
- Do not include any text outside the JSON object.
""").strip()


def _build_user_prompt(query: str) -> str:
    return f'Classify this query:\n\n"{query}"'


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class W5HClassifier:
    """
    Classifies natural language database queries using the W5H framework.

    Parameters
    ----------
    model : str
        Ollama model tag, e.g. "qwen2.5:7b-instruct-q4_K_M".
    base_url : str
        Ollama's OpenAI-compatible endpoint (default: http://localhost:11434/v1).
    temperature : float
        Sampling temperature. Lower = more deterministic (recommended: 0.0–0.2).
    max_retries : int
        How many times to retry on JSON parse failure before raising.
    timeout : float
        Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct-q4_K_M",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # Ollama's OpenAI-compatible endpoint does not require a real API key
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=timeout,
        )
        logger.info("W5HClassifier initialised — model: %s  endpoint: %s", model, base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> W5HProfile:
        """
        Classify a single natural language query.

        Parameters
        ----------
        query : str
            The natural language database query to classify.

        Returns
        -------
        W5HProfile
            Validated dimensional profile with confidence scores and
            frontier warnings where applicable.

        Raises
        ------
        ValueError
            If the model returns unparseable JSON after all retries.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        raw_json = self._call_with_retry(query)
        profile = self._parse_and_validate(query, raw_json)
        self._emit_warnings(profile)
        return profile

    def classify_batch(self, queries: list[str]) -> list[W5HProfile]:
        """
        Classify a list of queries sequentially (synchronous).

        Parameters
        ----------
        queries : list[str]
            Natural language queries to classify.

        Returns
        -------
        list[W5HProfile]
            One profile per query, in the same order.
        """
        results = []
        for i, query in enumerate(queries, 1):
            logger.info("Classifying query %d/%d", i, len(queries))
            results.append(self.classify(query))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, query: str) -> dict:
        """Call the model and return parsed JSON, retrying on failure."""
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": _build_user_prompt(query)},
                    ],
                    temperature=self.temperature,
                    # Ask Ollama to constrain sampling to valid JSON tokens.
                    # This is the single most effective reliability improvement
                    # for small local models on structured-output tasks.
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content.strip()
                logger.debug("Raw model output (attempt %d):\n%s", attempt, content)
                return json.loads(content)

            except json.JSONDecodeError as e:
                logger.warning(
                    "JSON parse failure on attempt %d/%d: %s",
                    attempt, self.max_retries, e,
                )
                last_error = e

            except Exception as e:
                logger.error("Ollama request failed on attempt %d: %s", attempt, e)
                raise

        raise ValueError(
            f"W5H classification failed after {self.max_retries} attempts. "
            f"Last JSON error: {last_error}"
        )

    def _parse_and_validate(self, query: str, raw: dict) -> W5HProfile:
        """
        Construct and validate a W5HProfile from the raw model output.
        Pydantic handles type coercion and range validation.
        """
        try:
            return W5HProfile(
                query=query,
                who=DimensionResult(**raw["who"]),
                what=DimensionResult(**raw["what"]),
                where=DimensionResult(**raw["where"]),
                when=DimensionResult(**raw["when"]),
                why=DimensionResult(**raw["why"]),
                how=DimensionResult(**raw["how"]),
            )
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Model output missing expected dimension keys. Raw output: {raw}"
            ) from e

    def _emit_warnings(self, profile: W5HProfile) -> None:
        """Emit Python warnings for active frontier dimensions."""
        for dim, message in profile.frontier_warnings().items():
            warnings.warn(
                f"[QUEST] Frontier dimension {dim} detected — {message}",
                UserWarning,
                stacklevel=3,
            )
            logger.warning("Frontier dimension %s active: %s", dim, message)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_profile(profile: W5HProfile) -> None:
    """Print a W5HProfile in a readable table format."""
    width = 72
    print("\n" + "=" * width)
    print(f"  Query : {profile.query}")
    print(f"  Active: {profile.summary()}")
    print("=" * width)
    print(f"  {'Dimension':<10} {'Active':<8} {'Conf':>6}  Evidence")
    print("-" * width)

    for dim_name in ("who", "what", "where", "when", "why", "how"):
        dim: DimensionResult = getattr(profile, dim_name)
        active_str  = "✓" if dim.active else "–"
        conf_str    = f"{dim.confidence:.2f}"
        evidence    = dim.evidence if dim.evidence else "—"
        # Wrap long evidence strings
        evidence_wrapped = textwrap.shorten(evidence, width=42, placeholder="…")
        print(f"  {dim_name.upper():<10} {active_str:<8} {conf_str:>6}  {evidence_wrapped}")

    warnings_dict = profile.frontier_warnings()
    if warnings_dict:
        print("-" * width)
        print("  ⚠  Frontier dimension warnings:")
        for dim, msg in warnings_dict.items():
            wrapped = textwrap.fill(msg, width=width - 6, subsequent_indent="        ")
            print(f"    {dim}: {wrapped}")

    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Quick smoke-test (runs only when executed directly, not when imported)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings as _warnings
    _warnings.simplefilter("always", UserWarning)

    # These are the three case-study queries from the QUEST paper (§2.3)
    test_queries = [
        # Simple — WHERE + WHEN only
        "Find all morning flights from Boston to New York.",

        # Semantic ambiguity — WHO-anchored WHEN requires subquery
        "Find flights from Boston arriving before the last departure of flight AA100.",

        # Multi-failure — constraint operator + correlated WHEN
        (
            "Find the cheapest morning flight from Boston to New York "
            "that arrives at least 45 minutes before the next departing flight to London."
        ),

        # Healthcare — strong WHO + WHEN + WHY (frontier)
        "Which patients over 65 were readmitted within 30 days due to surgical complications?",

        # HOW (mechanistic, frontier)
        "By what route were patients admitted to the ICU last quarter?",
    ]

    classifier = W5HClassifier(
        model="qwen2.5:7b-instruct-q4_K_M",
        temperature=0.1
    )

    for query in test_queries:
        profile = classifier.classify(query)
        print_profile(profile)
