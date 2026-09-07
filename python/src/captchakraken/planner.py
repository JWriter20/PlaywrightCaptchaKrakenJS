"""
ActionPlanner (v2) — talks to local vLLM serving the `captcha` LoRA over the
OpenAI-compatible /v1/chat/completions endpoint.

v1 supported transformers/vllm-local/gemini/openrouter and a whole tool-using
planner with detect/segment/drag-refine; that code is preserved on the
`v1-old-architecture` branch.
"""

import base64
import io
import json
import math
import os
import re
import sys
from mimetypes import guess_type
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from . import config, errors, prompts
from .server_manager import ensure_server
from .timing import timed

DEBUG = os.getenv("CAPTCHA_DEBUG", "0") == "1"

# The pixel floor the adapters are TRAINED at, and therefore the floor an image
# has to clear before it is worth sending.
#
# Training exports `MIN_PIXELS=200704` (448², in the finetune repo's
# scripts/train_unified.sh), so anything smaller is enlarged before the vision
# encoder sees it. Serving did none of that: vLLM runs with no
# `--mm-processor-kwargs` and this client used to re-encode the file unchanged,
# so a small captcha arrived at a geometry the model was never tuned on. It
# fails as plausible-but-wrong coordinates, never as an error — measured on real
# geetest_v3_slide captures (277x285), predictions landed 80-105 px from the
# hand label at native size and 1-4 px away once upscaled.
#
# A FLOOR, NOT A RESIZE. Images already above it are passed through byte-for-byte;
# re-encoding every screenshot would spend time and fidelity on the types that
# were never affected.
#
# Unconditional as a DEFAULT, because the deployed v1.1 adapter improves under
# it too (mean error ~40 px native vs ~4 px upscaled). But it is no longer a
# constant: `models.json` may declare a `pixel_budget` per model, because the
# band an adapter TRAINED under is a property of that adapter, exactly like its
# prompt generation. See prompts.pixel_budget.
MIN_PIXELS = 448 * 448


def _encode_image(path: str,
                  budget: "Optional[prompts.PixelBudget]" = None) -> tuple:
    """`(mime, base64)` for one image, fitted into `budget`'s pixel band.

    Area is clamped, never dimensions: aspect ratio is the geometry of a grid
    puzzle, and squashing it moves every tile centre — fatal when the answer is
    a coordinate on a normalized 0-1000 scale.

    Anything Pillow cannot open is passed through untouched — this is a
    preprocessing nicety, and it must never be the reason a solve fails.
    """
    budget = budget or prompts.pixel_budget(None)
    with open(path, "rb") as f:
        raw = f.read()
    mime = guess_type(path)[0] or "image/png"
    try:
        im = Image.open(io.BytesIO(raw))
        width, height = im.size
    except Exception:  # noqa: BLE001 — unreadable/animated: send it as-is
        return mime, base64.b64encode(raw).decode()
    cap = budget.maximum
    if cap and width * height > cap:
        # floor, not ceil: rounding up here can land back OVER the cap, which
        # is the mirror of the bug the floor branch below guards against.
        scale = math.sqrt(cap / (width * height))
        im = im.convert("RGB").resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return "image/png", base64.b64encode(buf.getvalue()).decode()
    if width * height >= budget.minimum:
        return mime, base64.b64encode(raw).decode()
    # ceil, not round: rounding both sides down lands just under the floor
    # (442x454 = 200,668 for a 277x285 capture) and defeats the whole point.
    scale = math.sqrt(budget.minimum / (width * height))
    im = im.convert("RGB").resize(
        (math.ceil(width * scale), math.ceil(height * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "image/png", base64.b64encode(buf.getvalue()).decode()

# Header the fleet's haproxy front (on the reverse-proxy EC2) routes on. When a
# caller sets CAPTCHA_REQUEST_PRIORITY to a positive int, every request carries
# `X-JH-Priority: <n>`; haproxy sends anything above its threshold (5) straight
# to the backup GPUs so it never competes with production traffic on the main
# 5090. The tier-2 CI gate sets this — its traffic is throwaway and must stay
# off the primary.
#
# Deliberately a HEADER, not vLLM's request-body `priority` field: that field is
# lower-is-higher and already means captcha=0 / apply=100 / label=200 on the
# priority-scheduled primary, so a value like 10 would MISORDER there (it would
# outrank apply/label). Routing is a fleet concern, kept out of the model's
# scheduler.
_PRIORITY_HEADER = "X-JH-Priority"
_PRIORITY_ENV = "CAPTCHA_REQUEST_PRIORITY"

# Hosted-API metadata. Both are OPTIONAL and absent unless deliberately set, so
# self-hosted users never send them and are unaffected.
#
# X-CK-Client  — which integration issued this solve (e.g. "camoufox/0.4.11").
#   camoufox sets CAPTCHA_KRAKEN_CLIENT in the env it spawns the solver with, so
#   the gateway can account camoufox-attributed usage separately. Attribution
#   ONLY: it is caller-supplied and therefore never priced on (the gateway
#   derives billable puzzle class from the request body instead).
# X-CK-Session — groups the 1..N inference rounds of ONE captcha into a single
#   billable attempt. The JS driver mints a UUID per solve() and reuses it for
#   every CLI invocation in that solve, which is what lets the gateway cap an
#   attempt's billable rounds instead of charging per round without limit.
# X-CK-Site    — the HOSTNAME the solve is happening on, and nothing else: no
#   path, no query, no credentials. `page_solver._site_of` derives it from
#   `page.url` per solve. It is what lets the hosted API answer "which sites is
#   this failing on", which is the question that turns a solve rate into work —
#   one vendor rolling out a new variant shows up as one host's rate falling
#   while the aggregate does not move. Attribution only, like X-CK-Client:
#   caller-supplied, never priced on, and `CAPTCHA_REPORT_SITE=0` turns it off.
_CLIENT_HEADER = "X-CK-Client"
_CLIENT_ENV = "CAPTCHA_KRAKEN_CLIENT"
_SESSION_HEADER = "X-CK-Session"
_SESSION_ENV = "CAPTCHA_KRAKEN_SESSION"
_SITE_HEADER = "X-CK-Site"
_SITE_ENV = "CAPTCHA_KRAKEN_SITE"

# Send the page's hostname with each solve. "0" turns it off.
#
# SEPARATE FROM THE OUTCOME SWITCH, because they are separate disclosures. The
# outcome is a fact about OUR model — did it get this one right — and it is what
# makes a failure improvable at all. The site is a fact about the CUSTOMER'S
# business, and an account can reasonably want the first sent and not the
# second. One switch for both would price that choice at "tell us nothing", and
# the failure corpus would be the thing that lost.
#
# The server-side half is `captureOptOut` on the account, which turns off the
# whole store; either one alone is enough to keep nothing.
_REPORT_SITE_ENV = "CAPTCHA_REPORT_SITE"

# Report whether the widget accepted, once per solve. "0" turns it off.
#
# ON by default because the report is what makes a failed solve improvable, and
# because it is the only way a hosted account's failures can be found at all.
# Opting out here stops the CLIENT sending it; `captureOptOut` on the account is
# the server-side half, and either one alone is enough to keep nothing.
_REPORT_OUTCOME_ENV = "CAPTCHA_REPORT_OUTCOME"

# Arbitrary extra headers, as "Name: value" pairs separated by newlines or
# commas. Empty and absent by default.
#
# WHY THIS EXISTS. Some endpoints sit behind a gate that is not the API key —
# our own dev gateway (api.dev.captchakraken.com) wants X-CK-Dev-Auth, and a
# corporate egress proxy may want its own token. Without this the published
# client simply cannot reach such an endpoint, which meant the dev environment
# could not exercise the real client path at all: the thing dev exists to
# rehearse was the one thing it could not do.
#
# IT CANNOT OVERWRITE Authorization, Content-Type, OR THE X-CK-* HEADERS. Those
# carry the credential and the billing attribution. Allowing an env var to
# rewrite Authorization would turn a stray export into "your key is not the key
# being charged", and allowing it to rewrite X-CK-Session would let a caller
# escape the per-attempt billing cap by pinning one session id forever.
_EXTRA_HEADERS_ENV = "CAPTCHA_KRAKEN_EXTRA_HEADERS"
_PROTECTED_HEADERS = frozenset(
    {"authorization", "content-type", _CLIENT_HEADER.lower(),
     _SESSION_HEADER.lower(), _SITE_HEADER.lower()}
)

# These values reach the wire verbatim from the environment, so they are
# sanitized rather than trusted: a CR/LF would otherwise splice arbitrary
# headers into the upstream request.
_HEADER_VALUE_MAX = 128


# RFC 7230 token characters. Anything else is not a header name.
_VALID_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def _clean_header_value(raw: str) -> str:
    """Printable-ASCII, length-capped header value ("" when nothing survives)."""
    return "".join(c for c in raw.strip() if 0x20 <= ord(c) < 0x7F)[:_HEADER_VALUE_MAX]


def _extra_headers(raw: str) -> Dict[str, str]:
    """Parse CAPTCHA_KRAKEN_EXTRA_HEADERS into a header dict.

    Accepts "Name: value" pairs separated by newlines or commas:

        CAPTCHA_KRAKEN_EXTRA_HEADERS='X-CK-Dev-Auth: hunter2'
        CAPTCHA_KRAKEN_EXTRA_HEADERS=$'X-One: a\nX-Two: b'

    Malformed entries are DROPPED, never guessed at. A pair with no colon is
    not a header, and inventing one from it would put an attacker-shaped string
    on the wire. Names are restricted to the RFC 7230 token characters for the
    same reason the values are sanitized: a CR/LF in either would splice extra
    headers into the request.

    Protected headers are refused, so this cannot be used to replace the
    credential or the billing attribution.
    """
    out: Dict[str, str] = {}
    if not raw.strip():
        return out

    for entry in raw.replace(",", "\n").splitlines():
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        name, _, value = entry.partition(":")
        name = name.strip()
        value = _clean_header_value(value)
        if not name or not value:
            continue
        if not _VALID_HEADER_NAME.match(name):
            continue
        if name.lower() in _PROTECTED_HEADERS:
            continue
        out[name] = value
    return out


def routing_headers(env=None) -> Dict[str, str]:
    """Fleet-routing + hosted-API headers derived from the environment.

    Empty by default. Split out and env-injectable so it can be unit-tested
    without a live server. A missing or non-integer CAPTCHA_REQUEST_PRIORITY
    yields no header at all — an unset/garbage value must never silently tag
    traffic for the backups.

    Each header is derived independently: a malformed priority must not suppress
    the attribution headers, or one typo would silently make a camoufox solve
    look like direct traffic and understate the partner's revenue share.
    """
    env = os.environ if env is None else env
    headers: Dict[str, str] = {}

    raw = (env.get(_PRIORITY_ENV) or "").strip()
    if raw:
        try:
            headers[_PRIORITY_HEADER] = str(int(raw))
        except ValueError:
            pass

    for header, var in ((_CLIENT_HEADER, _CLIENT_ENV), (_SESSION_HEADER, _SESSION_ENV),
                        (_SITE_HEADER, _SITE_ENV)):
        # The site is the one of the three a caller may switch off on its own.
        # Checked inside the loop rather than around it so a disabled site
        # cannot take the other two headers with it — that is the same
        # independence the priority parse above is written for, and one typo
        # there would make a camoufox solve look like direct traffic.
        if header == _SITE_HEADER and env.get(_REPORT_SITE_ENV, "1") == "0":
            continue
        value = _clean_header_value(env.get(var) or "")
        if value:
            headers[header] = value

    headers.update(_extra_headers(env.get(_EXTRA_HEADERS_ENV) or ""))

    return headers


# ── Prompt constants ────────────────────────────────────────────────────────
#
# These are ALIASES for the newest generation's built-ins, kept so that code and
# tests written against the old module-level constants keep working. They are
# NOT what a solve sends. The prompt a model gets is resolved per model
# (`prompts.resolve`), because a model answers in whatever schema its prompt
# asks for — sending a version-1 model the version-2 prompt does not error, it
# silently degrades every puzzle. See prompts.py and models.json.
#
# The text itself lives in prompts.py and nowhere else. Two copies of a prompt
# is the drift this module already shipped once (2026-07-18).
_LATEST = prompts.builtin(prompts.LATEST_PROMPT_VERSION)

SELECT_GRID_PROMPT = _LATEST.grid_template


# Non-grid click/drag puzzles, newest generation. Alias — see the note above.
PIXEL_ACTION_PROMPT = _LATEST.action_prompt


# Animated puzzles: the challenge is recorded, sliced into keyframes
# (`keyframes.py`) and sent as an ORDINARY MULTI-IMAGE request — one image per
# keyframe, all in one context so the model can compare them. The answer carries
# a `frame` naming which keyframe it acted on, and the page driver waits for the
# live widget to look like that frame before pressing the mouse down.
#
# Alias for the newest generation — see the note above. Generation 1 has NO
# animated prompt (the family did not exist), which is why resolution is per
# model rather than per client.
VIDEO_ACTION_PROMPT_TEMPLATE = _LATEST.video_template


def video_action_prompt(n_keyframes: int) -> str:
    """The prompt for a challenge served as `n_keyframes` stills.

    The count is in the text because the model has no other way to know how many
    images arrived — frame identities live in the prompt, not in the image payload.
    Mirrors `instructions.video_instruction` in the finetune repo exactly.
    """
    n = int(n_keyframes)
    if n < 1:
        raise ValueError(f"a keyframe request needs at least one frame, got {n_keyframes}")
    listing = ", ".join(f"frame {i}" for i in range(1, n + 1))
    return VIDEO_ACTION_PROMPT_TEMPLATE.format(n=n, listing=listing)


class ActionPlanner:
    """Thin client for the vLLM `captcha` LoRA."""

    def __init__(
        self,
        model: Optional[str] = None,
        debug_callback: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        expert: Optional[str] = None,
        **_: Any,
    ):
        self.debug_callback = debug_callback
        self.token_usage: List[Dict[str, Any]] = []

        # All model/endpoint defaults come from `config` (env-overridable), so
        # the planner itself is model-agnostic — swap models via env, not code.
        self.model = model or config.lora_name()
        self.base_url = base_url or config.base_url()
        self.api_key = api_key or config.api_key()
        # Prompts follow the MODEL, not this client's release. Resolve by the
        # served name first (a hosted endpoint serves `captcha`, not a repo id,
        # and models.json maps the alias back), falling back to the adapter repo
        # id for a self-hosted deploy that serves under its own name.
        #
        # Doing this ONCE per planner rather than per request: resolution can
        # touch the Hub, and a per-request lookup would put a network call in
        # front of every round of every solve.
        _prompt_key = (self.model if prompts.canonical_model_id(self.model)
                       else config.lora_adapter())
        self.prompts = prompts.resolve(_prompt_key)
        # Resolution follows the MODEL for the same reason prompts do, and off
        # the same key: an adapter reads a puzzle at the pixel band it trained
        # under. Resolved once per planner — see the note above.
        self.pixel_budget = prompts.pixel_budget(_prompt_key)
        # WHICH ADAPTER answers each prompt family. Empty for every model that
        # is not routed, which is every model published so far — `_model_for`
        # then returns `self.model` for every request and the wire is unchanged.
        #
        # Resolved once, like the two above, and for a stronger reason: the pin
        # is validated here, so `--expert grid` against a single-adapter model
        # fails when the solver is BUILT rather than on the first grid it meets.
        self.expert = (expert if expert is not None
                       else prompts.expert_pin())
        self.experts = prompts.experts(_prompt_key)
        if self.expert:
            # Raises on an unknown family, and on a pin against an unrouted
            # model. Both are configuration errors, and a benchmark that
            # quietly measured the generalist while reporting an expert is the
            # exact class of silent mispairing models.json exists to prevent.
            prompts.route(_prompt_key, None, pin=self.expert)
        # Auto-start a local vLLM server on the first request if one isn't up
        # (no-op for a healthy or remote endpoint). Guarded so we only try once.
        self._server_ensured = False
        # ONE connection, reused for every round of every solve this planner
        # serves. `requests.post` opens a fresh TCP connection each call — and
        # to a hosted endpoint a TLS handshake with it. Measured against the
        # gate's own endpoint: 258ms per request connectionless vs 144ms
        # pooled, so every inference was paying ~110ms to re-dial a server it
        # had just finished talking to. A multi-round grid pays it per round.
        #
        # The planner is per-PageSolver and a PageSolver is reusable across
        # solves, so a caller who keeps one keeps the connection warm too.
        self._http = requests.Session()
        # Latched off by the first 404 — see `report_outcome`.
        self._outcome_supported = True

    def _model_for(self, family: Optional[str]) -> str:
        """The `model` string for a request in `family`.

        `self.model` whenever the model is not routed, so a caller may pass a
        family unconditionally. A routed model with no expert for this family
        degrades to the generalist — never to another expert, never to an
        error (docs/MOE_LORA_DESIGN.md §11).
        """
        if not self.experts:
            return self.model
        return prompts.route(self.model, family, pin=self.expert) or self.model

    # ── the solve's own verdict, sent back ─────────────────────────────────
    #
    # THE ONE THING THE SERVER CANNOT SEE. A wrong answer reaches the gateway as
    # a 200 with well-formed JSON in it; only this driver watched the widget.
    # Without this report the hosted API cannot tell a solved captcha from a
    # failed one, and the failures — the exact boards the model is worst at, on
    # real sites — are the one dataset that cannot be collected any other way.
    #
    # BEST-EFFORT, ALWAYS. It runs after the solve is over and its result is
    # already decided, so nothing it does can change the answer the caller gets.
    # Every failure is swallowed: a self-hosted vLLM has no such route, a
    # network blip is not the caller's problem, and an exception here would turn
    # a successful solve into a raised one.
    #
    # ONE 404 IS ENOUGH. A self-hosted endpoint answers 404 forever, so the
    # first one latches this off for the life of the planner rather than paying
    # a round trip per solve to be told the same thing again.
    _OUTCOME_PATH = "/solve-outcome"
    _OUTCOME_TIMEOUT_S = 3.0

    def report_outcome(self, session_id: Optional[str], solved: bool) -> bool:
        """Tell the API whether the widget accepted. True if it was delivered.

        Never raises. The return value is for tests and for `--debug`; no caller
        should branch on it.
        """
        if not session_id or not self._outcome_supported:
            return False
        if os.getenv(_REPORT_OUTCOME_ENV, "1") == "0":
            return False
        url = f"{self.base_url}{self._OUTCOME_PATH}"
        try:
            resp = self._http.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"session": session_id, "solved": bool(solved)},
                timeout=self._OUTCOME_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 — see the note above
            self._log(f"outcome report failed: {exc}")
            return False
        if resp.status_code == 404:
            # Not an error: this endpoint does not serve the route. Stop asking.
            self._outcome_supported = False
            self._log("outcome reporting: endpoint has no /solve-outcome; disabled")
            return False
        ok = 200 <= resp.status_code < 300
        self._log(f"outcome report {session_id} solved={solved} -> {resp.status_code}")
        return ok

    def _log(self, message: str) -> None:
        if DEBUG:
            print(f"[Planner] {message}", file=sys.stderr)
        if self.debug_callback:
            self.debug_callback(f"[Planner] {message}")

    def _chat_with_image(self, prompt: str, image_path: str, max_tokens: int = 512,
                         family: Optional[str] = None) -> str:
        return self._chat_with_images(prompt, [image_path], max_tokens=max_tokens,
                                      family=family)

    def _chat_with_images(
        self, prompt: str, image_paths: List[str], max_tokens: int = 512,
        family: Optional[str] = None,
    ) -> str:
        """One request carrying N images followed by the prompt.

        The images go in ONE message, before the text, in the order given. That
        ordering is what makes an animated challenge answerable: the model is told
        "you are given N keyframes, in order: frame 1, frame 2, …" and matches those
        names to the images positionally. Splitting them across requests, or putting
        the text first, would break the correspondence between the numbers in the
        prompt and the pictures — and the frame number is the part the page driver
        acts on.
        """
        if not image_paths:
            raise ValueError("no images to send")

        parts: List[Dict[str, Any]] = []
        for p in image_paths:
            mime, b64 = _encode_image(p, self.pixel_budget)
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{mime};base64,{b64}"}})

        messages = [
            {
                "role": "system",
                "content": "You are an expert captcha solver. Respond ONLY with the JSON action.",
            },
            {
                "role": "user",
                "content": [*parts, {"type": "text", "text": prompt}],
            },
        ]

        model = self._model_for(family)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            # Qwen3.5's reasoning otherwise eats the token budget. `/no_think`
            # in the prompt alone is unreliable; disabling at the chat-template
            # level is the documented way.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Fleet routing: absent unless CAPTCHA_REQUEST_PRIORITY is set (see
            # routing_headers). Low-priority batch traffic is steered to the
            # backup GPUs by the haproxy front on this header.
            **routing_headers(),
        }

        # Hands-off server: before the very first request, make sure something
        # is listening (auto-start a local vLLM if needed; no-op if it's already
        # up or the endpoint is remote). Best-effort — a failure here surfaces as
        # a normal connection error on the request below.
        if not self._server_ensured:
            try:
                ensure_server(self.base_url)
            except Exception as e:  # noqa: BLE001 — don't mask the real request error
                self._log(f"ensure_server: {e}")
            self._server_ensured = True

        url = f"{self.base_url}/chat/completions"
        self._log(f"POST {url} model={model} max_tokens={max_tokens} "
                  f"images={len(parts)}")

        with timed("planner.chat"):
            resp = self._http.post(url, headers=headers, json=payload, timeout=120)

        # Surface auth / billing / server errors as something the reader can act
        # on, instead of letting resp.json() blow up with a cryptic "Expecting
        # value: line 1 column 1" on a non-JSON response.
        #
        # `errors.from_response` decides which of the two worlds we are in by
        # looking for the gateway's `error.code`: present means the hosted API
        # refused us and there is a real explanation to give (out of credits,
        # rate limited, attempt abandoned); absent means a local vLLM or a proxy,
        # and the old message — including the bearer-token hint on 401/403 — is
        # reproduced unchanged.
        if not resp.ok:
            raise errors.from_response(resp, url)

        try:
            data = resp.json()
        except ValueError:
            body = (resp.text or "")[:300]
            raise RuntimeError(
                f"vLLM returned a non-JSON body from {url} (is the server up and "
                f"is VLLM_BASE_URL correct?). Body: {body}"
            )

        if data.get("usage"):
            self.token_usage.append(data["usage"])

        content = data["choices"][0]["message"].get("content") or ""
        self._log(f"Raw content: {content[:300]}")
        return content

    @staticmethod
    def _parse_json(text: str) -> Any:
        text = (text or "").strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        start_obj = text.find("{")
        start_list = text.find("[")
        if start_list != -1 and (start_obj == -1 or start_list < start_obj):
            start = start_list
            end = text.rfind("]") + 1
        elif start_obj != -1:
            start = start_obj
            end = text.rfind("}") + 1
        else:
            return None

        # strict=False tolerates unescaped control chars INSIDE strings — the
        # model routinely emits coordinates as pretty-printed strings with a
        # literal newline, e.g. "click": ["277,\n  728", ...], which strict JSON
        # rejects. Without this the whole response fails to parse and a solvable
        # click puzzle is dropped as "unsupported".
        try:
            return json.loads(text[start:end], strict=False)
        except json.JSONDecodeError:
            # The model sometimes truncates (hit max_tokens) or over-nests its
            # pretty-printed JSON, leaving brackets unclosed. Repair by balancing
            # from the first opener to the end of the raw text.
            repaired = ActionPlanner._balance_json(text[start:])
            if repaired is not None:
                try:
                    return json.loads(repaired, strict=False)
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _balance_json(text: str) -> Optional[str]:
        """Close any brackets/braces the model left open (string-aware), so a
        truncated ``{"action": {"points": [[1,2],`` still parses. Returns None if
        there was nothing to balance."""
        stack: List[str] = []
        in_str = False
        esc = False
        for ch in text:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
        if not stack and not in_str:
            return None
        out = text
        if in_str:
            out += '"'
        # Drop a dangling comma / partial token before the closers.
        out = out.rstrip().rstrip(",")
        for opener in reversed(stack):
            out += "}" if opener == "{" else "]"
        return out

    def get_grid_selection(
        self,
        image_path: str,
        rows: int,
        cols: int,
        retry_mode: Optional[str] = None,
    ) -> List[int]:
        """Return the list of 1-indexed cells the model wants to click.

        retry_mode == "missed-tiles": the previous submission was rejected
        by the captcha vendor with an under-selection error. Append an
        explicit recovery instruction that tells the model the FULL grid
        contains at least one matching tile it didn't pick last time. This
        nudges it off the "I already covered everything" attractor.
        """
        total = rows * cols
        if rows == 4 and cols == 4:
            grid_hint = "Hint: Single large image split into tiles. Select ALL parts."
        else:
            grid_hint = "Hint: Separate images. Select only clear matches."

        prompt = self.prompts.grid_prompt(
            rows=rows, cols=cols, grid_hint=grid_hint
        )
        if retry_mode == "missed-tiles":
            prompt = (
                prompt
                + "\n\nIMPORTANT: A previous submission was rejected because not all "
                  "matching tiles were selected. Re-examine EVERY cell in the grid "
                  "carefully. There is at least one more matching tile you missed. "
                  "Return the complete list of cell numbers that match the description, "
                  "including any matches you may have overlooked."
            )
        raw = self._chat_with_image(prompt, image_path, max_tokens=128,
                                    family="grid")
        out = self._normalize_grid(self._parse_json(raw), total)
        self._log(f"grid selection -> {out}")
        return out

    @staticmethod
    def _normalize_grid(data: Any, total: int) -> List[int]:
        """Map the model's grid JSON to a list of 1-indexed cell numbers.

        Accepts the trained shape (a bare JSON array) plus the wrapped forms the
        model drifts into: {"target_ids": [...]} and {"action": {"target_ids":
        [...]}}. Out-of-range and non-numeric entries are dropped rather than
        raising — a single junk element must not cost the whole selection.

        Split out of get_grid_selection so the parse is testable without a
        model, mirroring _normalize_pixel.
        """
        if isinstance(data, list):
            ids = data
        elif isinstance(data, dict):
            # `action` is a dict on the grid path but a bare string ("drag") on
            # the pixel path — guard so a mis-routed response returns [] rather
            # than raising AttributeError.
            nested = data.get("action")
            nested_ids = nested.get("target_ids") if isinstance(nested, dict) else None
            ids = data.get("target_ids") or nested_ids or []
        else:
            ids = []

        out: List[int] = []
        for v in ids:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= iv <= total:
                out.append(iv)
        return out

    def get_pixel_actions(self, image_path: str, text_mode: bool = False) -> List[Dict[str, Any]]:
        """Solve a non-grid click/drag/slide/text puzzle.

        Sends the trained action prompt + image, parses the model's JSON, and
        returns a list of normalized actions with all coordinates on a 0–1
        scale:

          {"kind": "click", "points": [(x, y), ...]}
          {"kind": "drag",  "src": (x, y), "dst": (x, y)}
          {"kind": "slide", "dst": (x, y)}          — puzzle-piece slider
          {"kind": "type",  "text": "<the code>"}   — distorted text

        `text_mode` swaps in the distorted-text prompt. It is chosen by the
        DRIVER, from the widget's DOM (a visible text box in the challenge),
        because nothing about the picture reliably says "this one is typed" —
        BotDetect's warped letters and hCaptcha's "click the matching letter"
        look alike to a pixel classifier, and the two want opposite answers.

        Returns [] if the model produced nothing usable. The solver turns these
        into ClickAction / DragAction / TypeAction.
        """
        prompt = self.prompts.text_prompt() if text_mode else self.prompts.action_prompt
        raw = self._chat_with_image(prompt, image_path, max_tokens=512,
                                    family="text" if text_mode else "pixel")
        data = self._parse_json(raw)
        actions = self._normalize_pixel(data)
        self._log(f"pixel actions -> {actions}")
        return actions

    def get_keyframe_actions(self, keyframe_paths: List[str]) -> List[Dict[str, Any]]:
        """Solve an ANIMATED challenge from its keyframes.

        Same normalized actions as `get_pixel_actions`, each additionally carrying
        `"frame"`: the 1-based keyframe the model chose to act on, or None if it did
        not name a usable one.

        The frame is not decoration. On these puzzles the target is only there part
        of the time, so the coordinates are only correct while the widget looks like
        the frame they were read off. The page driver holds the mouse until it does.
        An action with `frame=None` is still returned — the caller decides whether to
        click blind or give up, and dropping it here would look identical to "the
        model had nothing to say".
        """
        if not keyframe_paths:
            return []
        # Per model, not per client: generation 1 has no animated prompt at all,
        # so this raises a clear error there instead of sending a v1 model a
        # keyframe request it was never trained to read.
        prompt = self.prompts.video_prompt(len(keyframe_paths))
        raw = self._chat_with_images(prompt, list(keyframe_paths), max_tokens=512,
                                     family="video")
        data = self._parse_json(raw)
        actions = self._normalize_pixel(data)
        frame = self._normalize_frame(data, len(keyframe_paths))
        for a in actions:
            a["frame"] = frame
        self._log(f"keyframe actions (frame={frame}) -> {actions}")
        return actions

    @staticmethod
    def _normalize_frame(data: Any, n_keyframes: int) -> Optional[int]:
        """The keyframe number the model chose, or None.

        Out-of-range is treated as "did not choose" rather than clamped. Clamping a
        7 to 6 would invent an intent the model never had, and the driver would then
        wait for a picture the answer was not about — a slow failure that looks like
        a flaky page. Tolerates the nested shape the model drifts into
        ({"action": {"frame": n}}), the same way the click/drag parsing does.
        """
        if not isinstance(data, dict):
            return None
        raw = data.get("frame")
        if raw is None and isinstance(data.get("action"), dict):
            raw = data["action"].get("frame")
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return None
        return n if 1 <= n <= n_keyframes else None

    @staticmethod
    def _normalize_pixel(data: Any) -> List[Dict[str, Any]]:
        """Map the model's 0–1000 click/drag JSON to 0–1 normalized actions.

        Tolerant of the two trained shapes plus a few near-misses:
          click: {"action": {"action": "click", "points": [[x, y], ...]}}
                 or top-level {"points": [...]} / {"action": {"points": [...]}}
          drag:  {"output": [{"Action": "simulate_drag",
                              "SourcePosition": {x, y},
                              "EstimatedPosition": {x, y}}, ...]}
        """
        def norm_xy(x: Any, y: Any) -> Optional[tuple]:
            try:
                fx, fy = float(x) / 1000.0, float(y) / 1000.0
            except (TypeError, ValueError):
                return None
            if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
                # Some outputs use 0–1 already; accept those too.
                if 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0:
                    fx, fy = float(x), float(y)
                else:
                    fx, fy = min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)
            return (fx, fy)

        def flat_numbers(v: Any) -> List[float]:
            """Every number anywhere inside v, in order. Handles nested lists,
            {x,y} dicts, bare numbers, and numeric strings like "277, 728" (the
            model sometimes emits coordinates as strings, occasionally even split
            across separate array elements)."""
            nums: List[float] = []
            if isinstance(v, bool):
                return nums
            if isinstance(v, (list, tuple)):
                for e in v:
                    nums.extend(flat_numbers(e))
            elif isinstance(v, dict):
                for k in ("x", "y", "X", "Y"):
                    if k in v:
                        nums.extend(flat_numbers(v[k]))
            elif isinstance(v, (int, float)):
                nums.append(float(v))
            elif isinstance(v, str):
                nums.extend(float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", v))
            return nums

        def coordish(v: Any) -> bool:
            """True when v is a non-empty list of COORDINATES (numbers, [x,y]
            pairs, {x,y} dicts, or numeric strings) — not text labels like
            "dog". Used to decide whether a "click"/"coordinates" array carries
            points we should salvage."""
            if not isinstance(v, (list, tuple)) or not v:
                return False
            for e in v:
                if isinstance(e, bool):
                    return False
                if isinstance(e, (int, float, list, tuple, dict)):
                    continue
                if isinstance(e, str) and re.search(r"\d", e):
                    continue
                return False
            return True

        out: List[Dict[str, Any]] = []
        if not isinstance(data, dict):
            return out

        # ---- type (CANONICAL — the schema TEXT_INSTRUCTION asks for on the
        #      distorted-text captchas: {"action": "type", "text": "<the code>"}).
        # First, because it is the one answer family with NO coordinate in it:
        # every branch below keys off a number, so a typed answer that fell
        # through reached the end and normalized to nothing.
        # The code is passed through VERBATIM — case and spacing are part of what
        # the model read off the image, and "tidying" them submits a different
        # answer than the one it gave.
        act = data.get("action")
        if isinstance(act, dict) and act.get("action") == "type":
            act = act.get("action")  # unwrap {"action": {"action": "type", ...}}
        if act == "type" or (isinstance(data.get("text"), str) and "drags" not in data):
            container = data if isinstance(data.get("text"), str) else data.get("action")
            text = container.get("text") if isinstance(container, dict) else None
            if isinstance(text, str) and text:
                return [{"kind": "type", "text": text}]

        # ---- drag (CANONICAL — the schema PIXEL_ACTION_PROMPT asks for and the
        #      LoRA is trained on, see instructions.py::ACTION_INSTRUCTION):
        #      {"action":"drag","drags":[{"source","from":[x,y],"destination","to":[x,y]}]}
        # Note "action" here is the STRING "drag", not a dict, so the click path
        # below never sees it. The legacy/simulate_drag branches that follow stay
        # as fallbacks for older adapters.
        content_drags = data.get("drags")
        if content_drags is None and isinstance(data.get("action"), dict):
            content_drags = data["action"].get("drags")
        if isinstance(content_drags, dict):
            content_drags = [content_drags]
        if isinstance(content_drags, list) and content_drags:
            for d in content_drags:
                if not isinstance(d, dict):
                    continue
                snums = flat_numbers(d.get("from"))
                dnums = flat_numbers(d.get("to"))
                if len(dnums) < 2:
                    continue  # a piece with nowhere to go is not actionable
                dst = norm_xy(dnums[0], dnums[1])
                if not dst:
                    continue
                if len(snums) >= 2:
                    src = norm_xy(snums[0], snums[1])
                    if src:
                        out.append({"kind": "drag", "src": src, "dst": dst})
                else:
                    # SOURCELESS — a puzzle-piece slider. The prompt's "FOR
                    # PUZZLE PIECE SLIDER PUZZLES" clause tells the model to
                    # leave the source empty precisely because the piece is not
                    # what you pick up: the handle is, somewhere else entirely,
                    # and how far it has to travel is not knowable from the
                    # picture. Requiring both ends here dropped the one answer
                    # shape the prompt asks for on every slide puzzle.
                    out.append({"kind": "slide", "dst": dst})
            if out:
                return out

        # ---- drag: {"output": [ {simulate_drag ...}, ... ]} ----
        drags = data.get("output")
        if isinstance(drags, list) and drags:
            for d in drags:
                if not isinstance(d, dict):
                    continue
                sp = d.get("SourcePosition") or {}
                ep = d.get("EstimatedPosition") or d.get("DestinationPosition") or {}
                src = norm_xy(sp.get("x"), sp.get("y")) if isinstance(sp, dict) else None
                dst = norm_xy(ep.get("x"), ep.get("y")) if isinstance(ep, dict) else None
                if src and dst:
                    out.append({"kind": "drag", "src": src, "dst": dst})
            if out:
                return out

        # ---- drag: {"action": {"simulate_drag": [{source_position,
        #                                           destination_position}]}} ----
        # The full-puzzle LoRA actually emits drags in this snake_case shape
        # (not the prompt's {"output":[{"Action":"simulate_drag",...}]}), and
        # packs each coordinate as {"x": [x, y]} — flat_numbers() pulls the pair
        # out of that. Without this branch a correctly-solved drag (e.g. hCaptcha
        # "drag ONE character to the matching character") is dropped as
        # "unsupported".
        sd = data.get("simulate_drag")
        if sd is None and isinstance(action := data.get("action"), dict):
            sd = action.get("simulate_drag")
        # The model emits simulate_drag as EITHER a list of drags OR a single
        # drag object; normalize the single-object form to a one-element list so
        # both parse (the object form was silently dropped as "unsupported").
        if isinstance(sd, dict):
            sd = [sd]
        if isinstance(sd, list) and sd:
            def _drag_coords(obj: Any, *roles: str) -> List[float]:
                # Pull the source/destination coordinate pair regardless of key
                # casing or separators — the LoRA freely varies between
                # source_position / sourcePosition / SourcePosition and
                # destination_position / destinationPosition / EstimatedPosition.
                # The coord value may be {"x": [x, y]}, [x, y], {x, y}, or a
                # string; flat_numbers() handles all of those.
                if not isinstance(obj, dict):
                    return []
                for k, v in obj.items():
                    kn = str(k).lower().replace("_", "").replace("-", "")
                    if "pos" not in kn:
                        continue
                    if any(r in kn for r in roles):
                        nums = flat_numbers(v)
                        if len(nums) >= 2:
                            return nums
                return []
            for d in sd:
                if not isinstance(d, dict):
                    continue
                snums = _drag_coords(d, "source", "src")
                dnums = _drag_coords(d, "destination", "dest", "estimated", "target")
                if len(snums) >= 2 and len(dnums) >= 2:
                    src = norm_xy(snums[0], snums[1])
                    dst = norm_xy(dnums[0], dnums[1])
                    if src and dst:
                        out.append({"kind": "drag", "src": src, "dst": dst})
            if out:
                return out

        # ---- click: {"action": {"action":"click","points":[...]}} ----
        action = data.get("action")
        # The model sometimes over-wraps: {"action": {"action": {"action":
        # "click", "points":[...]}}}. Peel nested "action" dicts until we reach
        # the one carrying points/coords (bounded so we never loop forever).
        for _ in range(4):
            if (
                isinstance(action, dict)
                and action.get("points") is None
                and action.get("source") is None
                and isinstance(action.get("action"), dict)
            ):
                action = action["action"]
            else:
                break
        points = None
        if isinstance(action, dict):
            points = action.get("points")
            # single drag emitted under "action"
            if points is None and action.get("action") == "drag":
                src = norm_xy(*(action.get("source") or (None, None)))
                dst = norm_xy(*(action.get("target") or (None, None)))
                if src and dst:
                    return [{"kind": "drag", "src": src, "dst": dst}]
        if points is None:
            points = data.get("points")
        # Salvage: some responses carry the click coordinates under "click" (or
        # "coordinates") instead of "points", as [x,y] pairs, {x,y} dicts, or
        # "x, y" strings (sometimes split across elements), with no "points" key
        # at all. Pull every number out of a coordinate-like value and pair them
        # into points so a well-intentioned answer isn't dropped as "unsupported".
        # Skipped when the value is text labels ("dog", "duck") rather than coords.
        if not (isinstance(points, list) and points):
            for container in (action if isinstance(action, dict) else None, data):
                if not isinstance(container, dict):
                    continue
                cand = container.get("click")
                if cand is None:
                    cand = container.get("coordinates")
                if coordish(cand):
                    nums = flat_numbers(cand)
                    if len(nums) >= 2:
                        points = [
                            [nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)
                        ]
                        break
        if isinstance(points, list) and points:
            pts = []
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    xy = norm_xy(p[0], p[1])
                elif isinstance(p, dict):
                    xy = norm_xy(p.get("x"), p.get("y"))
                else:
                    xy = None
                if xy:
                    pts.append(xy)
            if pts:
                out.append({"kind": "click", "points": pts})
        return out
