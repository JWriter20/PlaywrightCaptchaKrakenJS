"""
CaptchaKraken — OpenCV grid detection + a fine-tuned Qwen3.5-9B vision LoRA
served on vLLM.

Usage:
    from captchakraken import CaptchaSolver
    solver = CaptchaSolver()   # auto-starts / connects to a local vLLM server
    actions = solver.solve("captcha.png")

Model/endpoint defaults live in `captchakraken.config` and are fully
env-overridable (VLLM_BASE_URL, CAPTCHA_LORA_ADAPTER, …); the solver itself is
model-agnostic.
"""

from pathlib import Path

try:  # pragma: no cover
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
except Exception:
    pass

from .action_types import (
    CaptchaAction,
    ClickAction,
    DragAction,
    TypeAction,
    WaitAction,
)
from .image_processor import ImageProcessor
from .overlay import add_overlays_to_image

# The hosted API's refusal type. Outside the try below on purpose: `errors`
# imports nothing but `typing`, and docs/hosted-api.md tells every caller to
# branch on its `.code` — a recipe that has to work in both ports, and in a
# minimal install that has no serving stack.
from .errors import CaptchaKrakenAPIError

# How gestures are PERFORMED — mouse, touch, or nothing. Deliberately outside
# the try below: `humanize` imports only `trajectory`, so it has no dependency
# floor at all, and a caller writing a custom Humanizer must be able to import
# the base class without the serving stack installed.
from .humanize import (
    AppiumTouchBackend,
    CdpTouchBackend,
    Humanizer,
    MobileHumanizer,
    MouseHumanizer,
    NullHumanizer,
    TouchBackend,
    TouchscreenTouchBackend,
)

# The planner (requests) and solver (torch/vllm/transformers) pull in the heavy
# serving stack. Keep them optional so leaf modules — e.g. tool_calls.find_grid,
# which needs only cv2 + numpy + pillow — can be imported in a minimal env (CI's
# hermetic grid-detection test) without the full GPU dependency set installed.
try:  # pragma: no cover - exercised only when the serving stack is installed
    from .planner import ActionPlanner
    from .solver import CaptchaSolver, solve_captcha

    # The page driver sits behind the same guard: it imports `solver`, so it has
    # the same dependency floor. It needs NO browser package of its own — the
    # caller supplies the Playwright-compatible page (see page_solver's module
    # docstring on why we duck-type rather than import one).
    from .page_solver import PageSolver, SolveResult, solve_captcha_on_page
    from .watcher import CaptchaWatcher
except ModuleNotFoundError:
    ActionPlanner = None  # type: ignore[assignment,misc]
    CaptchaSolver = None  # type: ignore[assignment,misc]
    solve_captcha = None  # type: ignore[assignment]
    PageSolver = None  # type: ignore[assignment,misc]
    CaptchaWatcher = None  # type: ignore[assignment,misc]
    SolveResult = None  # type: ignore[assignment,misc]
    solve_captcha_on_page = None  # type: ignore[assignment]

__all__ = [
    "CaptchaSolver",
    "solve_captcha",
    "PageSolver",
    "CaptchaWatcher",
    "SolveResult",
    "solve_captcha_on_page",
    "ActionPlanner",
    "ImageProcessor",
    "CaptchaAction",
    "ClickAction",
    "DragAction",
    "TypeAction",
    "WaitAction",
    "add_overlays_to_image",
    "CaptchaKrakenAPIError",
    "Humanizer",
    "MouseHumanizer",
    "MobileHumanizer",
    "NullHumanizer",
    "TouchBackend",
    "CdpTouchBackend",
    "AppiumTouchBackend",
    "TouchscreenTouchBackend",
]

# KEEP IN STEP with python/pyproject.toml and js/package.json — the two ports
# ship together (rule 1c) and this is the only one importable code can read.
# It said 2.6.0 while the wheel on PyPI said 2.6.1, so anyone gating on the
# runtime attribute saw a version that had not been current for weeks.
# tests/test_public_contract.py compares all three.
__version__ = "2.10.0"
