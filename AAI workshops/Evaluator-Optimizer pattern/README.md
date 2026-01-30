# AI Workshop – Evaluator–Optimizer Workflow Pattern

## Google Gemini API Edition (Terminal-only, Runnable)

---

## Purpose

In this workshop, we’ll build an **agentic AI workflow** using **Python** and **Google Gemini**, runnable entirely from the **terminal**.

As we go, we’ll build the foundation first (LLM client), then the roles (agents), then the capabilities (tools), then the planner, and finally wire everything together.

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
2. Uses a **Planner Agent** to decide which tools to call and in what order (geocode\_city then get\_weather), and produce context for the Optimizer.
3. Uses an **Optimizer Agent** to generate (and revise) an itinerary based on the user prompt.
4. Uses an **Evaluator Agent** to judge the itinerary provided by the Optimizer and return structured feedback.
5. Iterates in an **Evaluator–Optimizer loop** until the itinerary is approved (or we hit a maximum number of tries).

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

macOS:

```bash
touch llm_client.py agents.py planner.py tools.py utils.py prompts.py main.py .env
```

Windows (PowerShell):

```powershell
New-Item llm_client.py -ItemType File
New-Item agents.py -ItemType File
New-Item planner.py -ItemType File
New-Item tools.py -ItemType File
New-Item utils.py -ItemType File
New-Item prompts.py -ItemType File
New-Item main.py -ItemType File
New-Item .env -ItemType File
```

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
    """Configuration for LLM calls."""

    model_candidates: Sequence[str]
    per_model_attempts: int = 2
    rate_limit_wait_seconds: float = 8.0
    overload_wait_seconds: float = 2.0


class GeminiLLMClient:
    """Retry-aware wrapper around google-genai."""

    def __init__(self, config: LLMConfig):
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not found in .env")

        self._client = genai.Client()
        self._config = config

    def generate(self, contents: List[str]) -> str:
        """Generate a response from Gemini."""

        last_err: Exception | None = None

        for model in self._config.model_candidates:
            for _ in range(self._config.per_model_attempts):
                try:
                    resp = self._client.models.generate_content(model=model, contents=contents)
                    return resp.text or ""
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()

                    if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                        print(f"[System] Rate limited on {model}. Waiting {self._config.rate_limit_wait_seconds}s...")
                        time.sleep(self._config.rate_limit_wait_seconds)
                        continue

                    if "503" in msg or "overloaded" in msg or "unavailable" in msg:
                        print(f"[System] Model {model} overloaded, waiting {self._config.overload_wait_seconds}s...")
                        time.sleep(self._config.overload_wait_seconds)
                        continue

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
"""System prompts for all agents."""

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

You will be given a Context block that contains the constraints and weather summary.
You MUST judge the itinerary against that Context.

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

````python
"""Shared utilities."""

import json
import random
import re
import time
from typing import Any, Dict, Optional


def apply_startup_jitter(min_seconds: float = 0.0, max_seconds: float = 1.0) -> None:
    """Small random delay to reduce synchronized API spikes."""
    random.seed()
    time.sleep(random.uniform(min_seconds, max_seconds))


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model response."""
    if not text or not text.strip():
        return None

    t = text.strip()

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

We also include:

- city validation error (`CityNotFoundError`)
- forecast date-range helper (`get_forecast_date_range`) to avoid out-of-range errors

```python
"""External tools (deterministic)."""

import requests


class CityNotFoundError(ValueError):
    """Raised when a city name cannot be resolved to coordinates."""


def geocode_city(city: str):
    """Convert city name → (latitude, longitude). Raises CityNotFoundError if not found."""

    if not validationCall:
        print(f"[Tool] Geocoding city - get lat/lon for {city}")

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}

    data = requests.get(url, params=params).json()
    results = data.get("results")

    if not results:
        raise CityNotFoundError(
            f"City could not be resolved by the geocoding tool: '{city}'. "
            "Please check spelling and try again."
        )

    location = results[0]
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

The evaluator now receives the shared Context block.

```python
"""Agent role wrappers."""

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
    """Evaluates an itinerary against a provided Context block."""

    llm: GeminiLLMClient
    system_prompt: str

    def evaluate(self, itinerary: str, iteration_number: int, context: str) -> Dict[str, Any]:
        """Evaluate itinerary against explicit context (constraints + weather)."""

        print("[Agent] Evaluator reviewing itinerary")

        txt = self.llm.generate([
            self.system_prompt,
            f"Iteration number: {iteration_number}\n\nContext:\n{context}\n\nItinerary:\n{itinerary}",
        ])

        parsed = extract_json_object(txt)
        if parsed is not None:
            return parsed

        print("[System] Evaluator returned invalid JSON; retrying once...")
        txt2 = self.llm.generate([
            self.system_prompt,
            (
                f"Iteration number: {iteration_number}\n\n"
                "Return ONLY valid JSON. No extra text. No markdown.\n\n"
                f"Context:\n{context}\n\nItinerary:\n{itinerary}"
            ),
        ])

        parsed2 = extract_json_object(txt2)
        if parsed2 is not None:
            return parsed2

        return {"approved": False, "issues": ["Evaluator returned invalid JSON output."]}
```

---

## Step 10: Build the Planner Agent (Tool Planning)

Planner is responsible for choosing tool calls and producing planner context.

```python
"""Planner agent."""

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
    llm: GeminiLLMClient

    def plan_context(self, city: str, start_date: str, end_date: str, intensity: str) -> Dict[str, str]:
        state: Dict[str, Any] = {"city": city, "start_date": start_date, "end_date": end_date, "intensity": intensity, "lat": None, "lon": None, "weather": None}

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

This step wires everything together and also contains input validation.

Input validation messages are labeled explicitly as **[Input Validation]** so they are not mistaken for LLM output.

```python
"""Entry point."""

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from tools import get_forecast_date_range, geocode_city, CityNotFoundError
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

MAX_TRIES = 10


def trip_length(a: str, b: str) -> int:
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).days + 1


def prompt_user_inputs():
    """Prompt for city/dates/intensity and validate them."""

    min_date, max_date = get_forecast_date_range()
    if min_date and max_date:
        print(f"[Input Validation] Weather forecasts are available only between {min_date} and {max_date} (inclusive).")

    # City validation loop
    while True:
        city = input("City to travel: ").strip()
        try:
            geocode_city(city, True)
            break
        except CityNotFoundError as e:
            print(f"[Input Validation] {e}")
            print("[Input Validation] Please re-enter the city name.\n")

    def read_date(label: str) -> str:
        while True:
            s = input(f"{label} (YYYY-MM-DD): ").strip()
            try:
                datetime.fromisoformat(s)
            except ValueError:
                print("[Input Validation] Invalid date format. Please use YYYY-MM-DD.")
                continue
            if min_date and max_date and not (min_date <= s <= max_date):
                print(f"[Input Validation] Date out of forecast range. Enter a date between {min_date} and {max_date}.")
                continue
            return s

    start_date = read_date("Start date")
    end_date = read_date("End date")

    while end_date < start_date:
        print("[Input Validation] End date must be the same as or after start date.")
        end_date = read_date("End date")

    # Intensity validation loop
    while True:
        intensity = input("Pace (easy/intense): ").strip().lower()
        if intensity in ("easy", "intense"):
            break
        print("[Input Validation] Invalid input. Please type exactly: easy or intense.")

    return city, start_date, end_date, intensity


def build_context(*, days: int, intensity: str, planner_context: str) -> str:
    """Common context shared across Optimizer and Evaluator."""

    return f"""
Trip constraints:
- Duration: {days} days
- Desired pace: {intensity}
- Category balance:
  * 35% History & heritage
  * 25% Arts & architecture
  * 20% Local life & food
  * 15% Nature & outdoor beauty
  * 5% Modern / contemporary city

Weather context:
{planner_context}
""".strip()


def main():
    load_dotenv(Path(__file__).parent / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not found in .env")

    apply_startup_jitter()

    city, start_date, end_date, intensity = prompt_user_inputs()
    days = trip_length(start_date, end_date)

    llm = GeminiLLMClient(
        LLMConfig(
            model_candidates=MODEL_CANDIDATES,
            per_model_attempts=2,
            rate_limit_wait_seconds=8.0,
            overload_wait_seconds=2.0,
        )
    )

    planner = PlannerAgent(llm=llm)
    optimizer = ItineraryOptimizerAgent(llm=llm, system_prompt=OPTIMIZER_SYSTEM_PROMPT)
    evaluator = ItineraryEvaluatorAgent(llm=llm, system_prompt=EVALUATOR_SYSTEM_PROMPT)

    print("[System] Starting agentic workflow")

    plan = planner.plan_context(city=city, start_date=start_date, end_date=end_date, intensity=intensity)
    print("[System] Planner summary:", plan["tool_results_summary"])

    # Shared Context
    context = build_context(days=days, intensity=intensity, planner_context=plan["prompt_context"])

    # Optimizer uses context
    user_prompt = f"""
Create a {days}-day travel itinerary for {city}.

{context}

Rules:
- Adjust pacing strictly based on the desired pace.
- Provide activities suitable for the weather conditions.
""".strip()

    current = optimizer.generate(user_prompt)

    print("\n[System] Initial itinerary generated\n")
    print(current)
    print("\n" + "-" * 60 + "\n")

    for i in range(MAX_TRIES):
        iteration = i + 1
        print(f"[System] Iteration {iteration}")

        evaluation = evaluator.evaluate(current, iteration, context)

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
