# AI Workshop – Evaluator–Optimizer Workflow Pattern

## Google Gemini API Edition (Terminal-only, Runnable)

---

## Purpose

In this workshop, we’ll build an **agentic AI workflow** using **Python** and **Google Gemini**, runnable entirely from the **terminal**.

As we go, We’ll build the foundation first (LLM client), then the roles (agents), then the capabilities (tools), then the planner, and finally wire everything together.

Everything runs via:

```bash
uv run python main.py
```

---

## What You Are Building

We will build a travel itinerary planner app that:

1. Prompts the user in the terminal for:
   - city
   - start date
   - end date
   - pace (easy / intense)
2. Uses a **Planner Agent** to decide which tools to call and in what order i.e. geocode_city then get_weather tools, and produce context for the user prompt.
3. Uses an **Optimizer Agent** to generate (and revise) an itinerary based on the user prompt (and revise itinerary based on feedbacks from evaluator agent).
4. Uses an **Evaluator Agent** to judge the itinerary provided by optimizer agent and return structured feedback.
5. Iterates in an **Evaluator–Optimizer loop** until itinerary is approved (or we hit a maximum number of tries).

---

## Step 0: Get a Free Gemini API Key

We need an API key so our local code can call the Gemini API.

1. Go to [https://aistudio.google.com](https://aistudio.google.com)
2. Sign in with a Google account
3. Click **Get API key**
4. Create a new API key
5. Copy the key

---

## Step 1: Install Required Software

### macOS

We’ll install Python and uv so we can create a reproducible project.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python uv
```

Verify:

```bash
python3 --version
uv --version
```

### Windows

Install Python first (tick **Add to PATH**) and then install uv.

1. Python: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. uv:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```powershell
python --version
uv --version
```

---

## Step 2: Create the Project (uv)

Now we create a reproducible project. uv will generate `pyproject.toml`, resolve dependencies, and lock them in `uv.lock`.

```bash
mkdir ai-evaluator-optimizer-workshop
cd ai-evaluator-optimizer-workshop

uv init
uv add google-genai requests python-dotenv
uv sync
```

At this point you will have:

- `pyproject.toml`
- `uv.lock`
- `.venv/`

---

## Step 3: Create Project Files

We split responsibilities into separate files so each component is easy to explain, test, and reuse.

Create these files:

- `llm_client.py`
- `agents.py`
- `planner.py`
- `tools.py`
- `utils.py`
- `prompts.py`
- `main.py`
- `.env`

Run this command: `touch llm_client.py agents.py planner.py tools.py utils.py prompts.py main.py .env`

---

## Step 4: Store the API Key

We store secrets in `.env` so they never appear in code.

```txt
GOOGLE_API_KEY=YOUR_API_KEY_HERE
```

---

## Step 5: Build the LLM Client (Foundation Layer)

Every agent will call the LLM. So we build one shared client that handles:

- model selection
- rate limits (429)
- overload (503)

```python
"""LLM client wrapper.

This module is the single integration point with Gemini.
All agents call `GeminiLLMClient.generate(...)` to interact with the LLM model.

Why:
- centralize retry logic
- centralize model selection
- keep agent code small and readable
"""

import os
import time
from dataclasses import dataclass
from typing import List, Sequence

from google import genai


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM calls.

    model_candidates:
        Ordered list of models to try.
    per_model_attempts:
        How many attempts per model before trying the next.
    rate_limit_wait_seconds:
        Sleep duration on 429/quota.
    overload_wait_seconds:
        Sleep duration on 503/overload.
    """

    model_candidates: Sequence[str]
    per_model_attempts: int = 2
    rate_limit_wait_seconds: float = 8.0
    overload_wait_seconds: float = 2.0


class GeminiLLMClient:
    """Retry-aware wrapper around google-genai.

    Call pattern:
        text = client.generate([system_prompt, user_prompt])

    We keep `contents` as a list of strings to make the workshop simpler.
    """

    def __init__(self, config: LLMConfig):
        # Fail fast if env is missing; this avoids confusing downstream errors.
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not found in .env")

        self._client = genai.Client()
        self._config = config

    def generate(self, contents: List[str]) -> str:
        """Generate a response from Gemini.

        Returns plain text (empty string if the SDK returns None).
        Raises on non-transient errors.
        """

        last_err: Exception | None = None

        for model in self._config.model_candidates:
            for _ in range(self._config.per_model_attempts):
                try:
                    resp = self._client.models.generate_content(
                        model=model,
                        contents=contents,
                    )
                    return resp.text or ""
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()

                    # 429 / quota exhaustion (often temporary per-minute). We wait and retry.
                    if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                        print(f"[System] Rate limited on {model}. Waiting {self._config.rate_limit_wait_seconds}s...")
                        time.sleep(self._config.rate_limit_wait_seconds)
                        continue

                    # 503 / overload. We wait briefly and retry.
                    if "503" in msg or "overloaded" in msg or "unavailable" in msg:
                        print(f"[System] Model {model} overloaded, waiting {self._config.overload_wait_seconds}s...")
                        time.sleep(self._config.overload_wait_seconds)
                        continue

                    # Non-transient: fail fast.
                    raise

        raise last_err if last_err else RuntimeError("Unknown LLM failure")
```

---

## Step 6: Define Prompts (Policy Layer)

Prompts define the boundaries of each role. They are the policy layer that keeps the system predictable:

- Planner: tool steps only (JSON)
- Optimizer: itinerary text only
- Evaluator: JSON decision only

```python
"""System prompts for all agents.

Keep prompts in one place so:
- role boundaries are visible
- changes are easy to review
- other files stay clean
"""

PLANNER_SYSTEM_PROMPT = """
You are a planning agent responsible for preparing context for a travel itinerary.

Responsibilities:
- Decide which tools must be called to plan the trip
- Determine the correct order of tool usage
- Request only the minimum information required

Available tools:
1. geocode_city(city) → latitude, longitude
2. get_weather(latitude, longitude, start_date, end_date) → day-wise weather summary

Rules:
- Respond ONLY in valid JSON
- Do NOT generate an itinerary
- Do NOT explain reasoning

Output format:
{
  "steps": [
    {"tool": "geocode_city", "args": {"city": "<city>"}},
    {"tool": "get_weather", "args": {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}
  ],
  "prompt_context_template": "Weather summary for the trip: {weather}."
}
"""


OPTIMIZER_SYSTEM_PROMPT = """
You are a travel itinerary optimizer.

Responsibilities:
- Generate or revise itineraries
- Follow all constraints strictly
- Adapt plans based on provided weather summaries

Output format (strict):
Day 1:
Morning: ... (Tags)
Afternoon: ... (Tags)
Evening: ... (Tags)

Rules:
- Plain text only
- No JSON
- No markdown
- No code fences
- No explanations
"""


EVALUATOR_SYSTEM_PROMPT = """
You are an itinerary evaluation agent.

Responsibilities:
- Evaluate the itinerary against constraints
- Verify pacing matches the requested intensity
- Assess weather suitability

Output MUST be valid JSON:
{
  "approved": true | false,
  "issues": ["string"]
}

Rules:
- If Iteration number is 1: approved MUST be false and at least one itinerary-related issue must be listed
- On Iteration number >= 2: approve only if issues are resolved
- Return ONLY the JSON object
"""
```

---

## Step 7: Add Utilities (Shared Helpers)

Utilities are cross-cutting concerns:

- Startup jitter helps reduce synchronized requests in a workshop.
- JSON extraction makes the evaluator robust when the model wraps JSON.

**File:**``

````python
"""Shared utilities.

These helpers are used in multiple modules to keep the rest of the code focused.
"""

import json
import random
import re
import time
from typing import Any, Dict, Optional


def apply_startup_jitter(min_seconds: float = 0.0, max_seconds: float = 1.0) -> None:
    """Sleep a small random amount to reduce simultaneous API spikes."""
    random.seed()
    time.sleep(random.uniform(min_seconds, max_seconds))


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model response.

    Handles common cases where the model wraps JSON in markdown code fences.
    Returns None if parsing fails.
    """
    if not text or not text.strip():
        return None

    t = text.strip()

    # Strip markdown fences if present
    if t.startswith("```"):
        t = re.sub(r"^```.*?\n", "", t, flags=re.DOTALL)
        t = re.sub(r"\n?```$", "", t).strip()

    match = re.search(r"\{[\s\S]*\}", t)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
````

---

## Step 8: Build the Tools (Deterministic Capabilities)

Tools are *not* LLM calls. They are deterministic functions that ground the system in real data.

We also include a forecast date-range helper so we can validate user input and avoid out-of-range errors.

```python
"""External tools (deterministic).

These functions are intentionally simple and testable.
They do not call the LLM.
"""

import requests


def geocode_city(city: str):
    """Convert city name → (latitude, longitude) using Open-Meteo geocoding."""
    print(f"[Tool] Geocoding city - get lat/lon for {city}")

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}

    data = requests.get(url, params=params).json()
    location = data["results"][0]

    return location["latitude"], location["longitude"]


def get_forecast_date_range():
    """Return (min_date, max_date) supported by Open-Meteo forecast today."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": 0, "longitude": 0, "daily": "weathercode", "timezone": "UTC"}

    data = requests.get(url, params=params).json()
    dates = data.get("daily", {}).get("time", [])

    return (dates[0], dates[-1]) if dates else ("", "")


def get_weather(lat, lon, start_date, end_date):
    """Fetch daily weather codes and return a day-by-day human summary."""
    print("[Tool] Fetching weather forecast")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "weathercode",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }

    data = requests.get(url, params=params).json()

    if "daily" not in data or "weathercode" not in data["daily"]:
        return "Weather data unavailable"

    codes = data["daily"]["weathercode"]

    def interpret(code: int) -> str:
        """Map Open-Meteo weather codes to human labels."""
        if code == 0:
            return "clear sky"
        if 1 <= code <= 3:
            return "partly cloudy"
        if 51 <= code <= 55:
            return "drizzle"
        if 61 <= code <= 65:
            return "rain"
        if 71 <= code <= 75:
            return "snow"
        if code >= 95:
            return "thunderstorms"
        return "mixed conditions"

    return ", ".join(f"Day {i + 1}: {interpret(c)}" for i, c in enumerate(codes))
```

---

## Step 9: Build the Agents (Optimizer + Evaluator)

Agents are small role wrappers around:

- a system prompt
- the shared LLM client

This keeps the loop logic in `main.py` readable.

```python
"""Agent role wrappers.

Agents are intentionally thin:
- Optimizer: generates/revises
- Evaluator: judges and returns JSON

All LLM calling is delegated to GeminiLLMClient.
"""

from dataclasses import dataclass
from typing import Any, Dict

from llm_client import GeminiLLMClient
from utils import extract_json_object


@dataclass
class ItineraryOptimizerAgent:
    """Generates or revises an itinerary."""

    llm: GeminiLLMClient
    system_prompt: str

    def generate(self, prompt: str) -> str:
        print("[Agent] Optimizer generating itinerary")
        return self.llm.generate([self.system_prompt, prompt])


@dataclass
class ItineraryEvaluatorAgent:
    """Evaluates an itinerary and returns a JSON decision."""

    llm: GeminiLLMClient
    system_prompt: str

    def evaluate(self, itinerary: str, iteration_number: int) -> Dict[str, Any]:
        print("[Agent] Evaluator reviewing itinerary")

        txt = self.llm.generate([
            self.system_prompt,
            f"Iteration number: {iteration_number}\n\n{itinerary}",
        ])

        parsed = extract_json_object(txt)
        if parsed is not None:
            return parsed

        # One retry with stricter instruction
        print("[System] Evaluator returned invalid JSON; retrying once...")
        txt2 = self.llm.generate([
            self.system_prompt,
            (
                f"Iteration number: {iteration_number}\n\n"
                "Return ONLY valid JSON. No extra text. No markdown.\n\n"
                f"{itinerary}"
            ),
        ])

        parsed2 = extract_json_object(txt2)
        if parsed2 is not None:
            return parsed2

        # Hard fallback keeps the loop moving.
        return {"approved": False, "issues": ["Pacing may be too aggressive on at least one day."]}
```

---

## Step 10: Build the Planner Agent (Tool Planning)

The Planner requests tool steps in JSON. Our code executes those tool calls deterministically.

```python
"""Planner agent.

The Planner does not generate itineraries.
It decides what context to gather and in what order.

In this workshop implementation:
- the Planner returns JSON tool steps
- our Python code executes those tools
- the Planner output is converted into prompt context for the Optimizer
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from llm_client import GeminiLLMClient
from utils import extract_json_object
from tools import geocode_city, get_weather
from prompts import PLANNER_SYSTEM_PROMPT


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "geocode_city",
        "description": "Convert a city name into latitude/longitude.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        "output_schema": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}}, "required": ["lat", "lon"]},
    },
    {
        "name": "get_weather",
        "description": "Fetch a human-readable daily weather summary for the date range.",
        "input_schema": {"type": "object", "properties": {"start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["start_date", "end_date"]},
        "output_schema": {"type": "string"},
    },
]


@dataclass
class PlannerAgent:
    """Requests tool steps and produces optimizer-ready context."""

    llm: GeminiLLMClient

    def plan_context(self, city: str, start_date: str, end_date: str, intensity: str) -> Dict[str, str]:
        state: Dict[str, Any] = {
            "city": city,
            "start_date": start_date,
            "end_date": end_date,
            "intensity": intensity,
            "lat": None,
            "lon": None,
            "weather": None,
        }

        # Ask the planner to return tool steps.
        prompt1 = {
            "inputs": {"city": city, "start_date": start_date, "end_date": end_date, "intensity": intensity},
            "available_tools": TOOL_SCHEMAS,
            "instruction": "Return valid JSON with tool steps (geocode_city then get_weather).",
        }

        txt = self.llm.generate([PLANNER_SYSTEM_PROMPT, f"{prompt1}"])
        msg = extract_json_object(txt)
        if not msg:
            return self._fallback(state)

        steps = msg.get("steps") or msg.get("calls")
        if not isinstance(steps, list):
            return self._fallback(state)

        # Execute steps deterministically.
        for call in steps:
            tool = call.get("tool")
            args = call.get("args", {})

            if tool == "geocode_city":
                lat, lon = geocode_city(args.get("city", city))
                state["lat"], state["lon"] = lat, lon

            elif tool == "get_weather":
                if state["lat"] is None or state["lon"] is None:
                    lat, lon = geocode_city(city)
                    state["lat"], state["lon"] = lat, lon

                state["weather"] = get_weather(
                    state["lat"],
                    state["lon"],
                    args.get("start_date", start_date),
                    args.get("end_date", end_date),
                )

        weather = state.get("weather")
        return {
            "tool_results_summary": f"Weather: {weather}",
            "prompt_context": (
                f"Weather summary for the trip: {weather}. "
                "If rain/drizzle is expected, prefer indoor museums/churches and shorten outdoor ruins/parks."
            ),
        }

    def _fallback(self, state: Dict[str, Any]) -> Dict[str, str]:
        weather = state.get("weather")
        return {
            "tool_results_summary": f"City={state.get('city')}, Weather={weather}",
            "prompt_context": (
                f"Weather summary for the trip: {weather}. "
                "If rain/drizzle is expected, prefer indoor museums/churches and shorten outdoor ruins/parks."
            ),
        }
```

---

## Step 11: Orchestration (main.py)

Now we wire everything together. This is the only place where we:

- load environment variables
- prompt for user inputs
- run planner → tools → optimizer
- run evaluator–optimizer loop (up to 10 tries)

```python
"""Entry point.

This file is intentionally simple:
- gather inputs
- assemble components
- run the workflow

All complexity is pushed into modules we already built.
"""

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from tools import get_forecast_date_range
from utils import apply_startup_jitter
from llm_client import LLMConfig, GeminiLLMClient
from agents import ItineraryOptimizerAgent, ItineraryEvaluatorAgent
from planner import PlannerAgent
from prompts import OPTIMIZER_SYSTEM_PROMPT, EVALUATOR_SYSTEM_PROMPT


MODEL_CANDIDATES = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]

# Increased to 10 so the workshop reliably converges even when the model is noisy.
MAX_TRIES = 10


def trip_length(a: str, b: str) -> int:
    """Compute trip length in days (inclusive)."""
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).days + 1


def prompt_user_inputs():
    """Prompt the user for inputs and validate dates against forecast range."""

    city = input("City to travel: ").strip()

    # Prevent common workshop failure: weather forecast out-of-range.
    min_date, max_date = get_forecast_date_range()
    if min_date and max_date:
        print(f"Note: Weather forecasts are available only between {min_date} and {max_date} (inclusive).")

    def read_date(label: str) -> str:
        while True:
            s = input(f"{label} (YYYY-MM-DD): ").strip()
            try:
                datetime.fromisoformat(s)
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD.")
                continue

            if min_date and max_date and not (min_date <= s <= max_date):
                print(f"Date out of forecast range. Enter a date between {min_date} and {max_date}.")
                continue

            return s

    start_date = read_date("Start date")
    end_date = read_date("End date")

    while end_date < start_date:
        print("End date must be the same as or after start date.")
        end_date = read_date("End date")

    intensity = input("Pace (easy/intense): ").strip().lower()
    if intensity not in ("easy", "intense"):
        intensity = "easy"

    return city, start_date, end_date, intensity


def main():
    """Run the full workflow from the terminal."""

    # Load environment variables from .env
    load_dotenv(Path(__file__).parent / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not found in .env")

    # Small random delay to reduce API spikes in a room full of participants.
    apply_startup_jitter()

    # 1) Gather user inputs
    city, start_date, end_date, intensity = prompt_user_inputs()
    days = trip_length(start_date, end_date)

    # 2) Construct the shared LLM client
    llm = GeminiLLMClient(
        LLMConfig(
            model_candidates=MODEL_CANDIDATES,
            per_model_attempts=2,
            rate_limit_wait_seconds=8.0,
            overload_wait_seconds=2.0,
        )
    )

    # 3) Construct agents
    planner = PlannerAgent(llm=llm)
    optimizer = ItineraryOptimizerAgent(llm=llm, system_prompt=OPTIMIZER_SYSTEM_PROMPT)
    evaluator = ItineraryEvaluatorAgent(llm=llm, system_prompt=EVALUATOR_SYSTEM_PROMPT)

    print("[System] Starting agentic workflow")

    # 4) Planner prepares context using tools
    plan = planner.plan_context(city=city, start_date=start_date, end_date=end_date, intensity=intensity)
    print("[System] Planner summary:", plan["tool_results_summary"])

    # 5) Optimizer generates the initial itinerary
    user_prompt = f"""
Create a {days}-day travel itinerary for {city}.

Desired pace: {intensity}

Constraints:
- 35% History & heritage
- 25% Arts & architecture
- 20% Local life & food
- 15% Nature & outdoor beauty
- 5% Modern / contemporary city

Additional context (Planner Agent):
{plan["prompt_context"]}

Adjust pacing strictly based on the desired pace.
Provide activities suitable for the weather conditions.
""".strip()

    current = optimizer.generate(user_prompt)

    print("\n[System] Initial itinerary generated\n")
    print(current)
    print("\n" + "-" * 60 + "\n")

    # 6) Evaluator–Optimizer loop
    for i in range(MAX_TRIES):
        iteration = i + 1
        print(f"[System] Iteration {iteration}")

        evaluation = evaluator.evaluate(current, iteration)

        if evaluation.get("approved"):
            print("\n[System] Itinerary approved\n")
            print(current)
            break

        issues = evaluation.get("issues", [])
        print("[System] Issues found:")
        for issue in issues:
            print(f"- {issue}")

        revision_prompt = (
            "Revise the itinerary addressing the following issues:\n"
            + "\n".join(issues)
            + "\n\n"
            + current
        )

        current = optimizer.generate(revision_prompt)


if __name__ == "__main__":
    main()
```

---

## Step 12: Run the Application

From the project root:

```bash
uv run python main.py
```

---

## Key Takeaway

If you remember only one thing, remember this: the **Evaluator–Optimizer loop** is the decision-making core.

The Planner prepares context. The Optimizer proposes. The Evaluator decides.

That’s what makes the workflow agentic, controllable, and debuggable.

