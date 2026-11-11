"""The Babamul assistant: a chat panel in Babamul that answers with these tools.

The browser posts the conversation to `{prefix}/chat` with the person's Babamul
token. This server runs the tool loop against an OpenAI-compatible model and
streams what happens back as server-sent events. The model key never leaves the
server, and every tool call runs with the person's own token. It follows
SkyPortal's assistant (services/assistant/app.py, skyportal/utils/assistant.py).

Tools run in this process, not over loopback HTTP to /mcp: the token reaches
them through `app.caller`, set inside the task answering one question. That
needs no knowledge of the address this server is reached at, no Host checks
and no MCP session per call, and a tool cannot fall back to the server's token.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
import traceback
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field

import anyio
import httpx
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import keystore
from . import providers as p
from . import summarize as s
from .app import INSTRUCTIONS, caller, mcp, settings
from .babamul import USER_AGENT, Babamul, base_url_from_env
from .keystore import SHARED, KeyStore

log = logging.getLogger("boom_mcp.assistant")

SYSTEM_PROMPT = """You are an assistant inside Babamul, the web app of the BOOM \
alert broker. You are talking with an astronomer in a chat panel of the app.

Use the tools to look things up rather than guessing; if the tools do not answer \
the question, say so plainly. Be brief: this is a chat message, not a report. \
Write object ids in full, exactly as the tools give them, so the app can link them. \
Use plain punctuation: commas, colons or parentheses, never em dashes."""

NO_VISION = """Images the tools return, such as cutouts and light-curve plots, \
are shown to the person in the chat as soon as the tool finishes. You cannot see \
them yourself, so never describe what they show: say they are in the chat, and \
what to look for in them. Mention images only when a tool result says it showed \
one."""

VISION = """Images the tools return, such as cutouts and light-curve plots, are \
shown to the person in the chat as soon as the tool finishes, and you are given \
them too. Describe only what you can actually see in them, and mention images \
only when a tool result says it showed one."""

ANSWER_NOW = (
    "You cannot call tools any more. Answer now, in plain text, from what you have, "
    "and say what is still unknown."
)

# What the person reads when something goes wrong. Each says whether asking
# again can help, because the person cannot tell otherwise.
UNREACHABLE = "The model is not answering right now. Try again in a minute."
STALLED = "The model stopped answering partway. Nothing is wrong with what you asked; try again."
MISCONFIGURED = "The assistant is not set up correctly. Tell the Babamul team."
REFUSED = (
    "The model could not take this conversation, maybe because it is too long. "
    "Start a new chat and try again."
)
BROKEN = "Something went wrong while looking that up. Try again."
OUT_OF_TIME = "I ran out of time working that out. Please ask again, or narrow the question."
NO_ANSWER = "I could not find an answer to that."
CUT_SHORT = "\n\n(I ran out of time here, so this answer is cut short.)"
TOO_LONG = "\n\n(This answer reached its length limit, so it is cut short.)"
LINE_FULL = "The assistant is busy with other people's questions. Try again in a few minutes."
WAITED_TOO_LONG = "The assistant is still busy with other people's questions. Try again in a few minutes."
REPLACED = "Stopped because you asked a new question, here or in another tab."
# The status line's "Answering with ..." when the key store cannot be read.
STORE_DOWN_LABEL = "the shared model: saved keys cannot be reached right now"

# The part of the answer timeout the tool loop may use. The rest is kept for the
# final answer, so running out of time still ends in an answer.
TOOL_SHARE = 0.75
# How much of a tool result the trace under the answer shows.
SUMMARY_BUDGET = 300
# An old tool result shrinks to this when the conversation outgrows the model.
SHRUNK_RESULT = 2000
# A status event after this many quiet seconds, so proxies keep the stream open.
HEARTBEAT = 15.0
TOKEN_TTL = 300.0
BAD_TOKEN_TTL = 60.0
# Tokens remembered at once. Past it the oldest go, so trying many tokens cannot grow it.
TOKEN_CACHE = 4096
# A Bearer token as RFC 6750 spells it. Babamul's API and login tokens both fit; anything
# else Babamul would refuse, and a byte outside ASCII would crash the request to it.
TOKEN = re.compile(r"[A-Za-z0-9._~+/-]{1,4096}=*")
# A login token of the Babamul web app, a JWT, as opposed to a bbml_ API token. Only
# these may manage or spend a person's own model key: API tokens end up in notebooks
# and repositories, and Babamul never expires them.
LOGIN_TOKEN = re.compile(r"eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
# A model API key: printable ASCII with no spaces. Anything else in an Authorization
# header makes h11 raise an error that quotes the whole header, key and all, and
# that error would reach the logs.
API_KEY = re.compile(r"[!-~]{1,4096}")
# Extra headers an endpoint sends: a header name, and a value of printable ASCII.
HEADER_NAME = re.compile(r"[A-Za-z0-9!#$%&'*+.^_`|~-]{1,64}")
HEADER_VALUE = re.compile(r"[!-~](?:[ -~]{0,1022}[!-~])?")
# Headers stream() sets itself, which an endpoint's extra headers may not replace.
OWN_HEADERS = frozenset({"accept", "authorization", "content-length", "content-type", "host"})
# How an endpoint wants its requests, see chat_payload().
KINDS = ("llamacpp", "openai")
MAX_BODY = 1_000_000
MAX_MESSAGE_CHARS = 20_000
# How long a person may have their key kept, in days, and the usual choice.
KEY_DAYS = (1, 7, 30, 90)
DEFAULT_KEY_DAYS = 30
# Keys a person may save in an hour, each checked with the provider, so that this
# server cannot be used to try out keys found elsewhere.
SAVES_PER_HOUR = 10
# Seconds between sweeps of expired keys.
SWEEP_EVERY = 3600.0
MAX_KEY_BODY = 16_000


def env_flag(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def env_number(env, name, default):
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return type(default)(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, not {raw!r}") from None


def is_login(token: str) -> bool:
    return not token.startswith("bbml_") and bool(LOGIN_TOKEN.fullmatch(token))


def check_key(key: str, what: str) -> str:
    """key, if it can go in an Authorization header. The error names what the key
    is for and never the key itself."""
    if key and not API_KEY.fullmatch(key):
        raise ValueError(f"{what} has spaces, line breaks or characters outside ASCII, so it cannot be sent")
    return key


def shown_url(url: str) -> str:
    """url for a log line: without a user name, password or query, which can hold a key."""
    try:
        return str(httpx.URL(url).copy_with(userinfo=b"", query=None, fragment=None))
    except (httpx.InvalidURL, TypeError, ValueError):
        return "(a URL that does not parse)"


@dataclass(frozen=True)
class Endpoint:
    """An OpenAI-compatible server: base_url ends in /v1.

    kind is what the request body may hold (see chat_payload): "llamacpp", the
    operator's own server, or "openai", a hosted provider. headers are extra
    (name, value) pairs for every request, such as a provider's workspace id,
    and body extra fields for an "openai" one, such as OpenRouter's routing.
    """

    base_url: str
    model: str = ""
    api_key: str = field(default="", repr=False)
    kind: str = "llamacpp"
    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    body: dict = field(default_factory=dict, repr=False, hash=False, compare=False)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"an endpoint's kind is one of {', '.join(KINDS)}, not {self.kind!r}")
        check_key(self.api_key, f"the API key for {shown_url(self.base_url)}")
        for name, value in self.headers:
            # Checked here so that a bad one never reaches h11, whose error would quote it.
            if not HEADER_NAME.fullmatch(name) or name.lower() in OWN_HEADERS or not HEADER_VALUE.fullmatch(value):
                raise ValueError(f"the extra header {name!r} for {shown_url(self.base_url)} cannot be sent")


@dataclass
class Config:
    llm: Endpoint
    fallback: Endpoint | None = None
    vision: bool = False
    thinking: bool = False
    max_tool_calls: int = 10
    answer_timeout: float = 240.0
    stall_timeout: float = 60.0
    result_budget: int = 20_000
    context_budget: int = 60_000
    max_messages: int = 30
    max_tokens: int = 1024
    # Questions answered at once. Match it to the model server's slots (llama.cpp -np):
    # more would only queue inside the model, where nobody can tell how long the wait is.
    concurrency: int = 2
    queue_timeout: float = 300.0
    max_queue: int = 20
    rate_limit: int = 20
    rate_window: float = 600.0
    fallback_period: float = 300.0
    # People's own keys, when the operator has set up a store for them.
    own_keys: "OwnKeys | None" = None

    @classmethod
    def from_env(cls, env=None) -> "Config | None":
        """The configuration in ASSISTANT_* variables, or None when ASSISTANT_LLM_URL is unset."""
        env = os.environ if env is None else env
        url = env.get("ASSISTANT_LLM_URL", "").strip()
        if not url:
            return None

        def number(name, default):
            return env_number(env, name, default)

        def text(name):
            # A trailing newline from an env file would otherwise reach the model's
            # name or the Authorization header.
            return env.get(name, "").strip()

        def key(name):
            # Refused at startup, by name, rather than failing on every question.
            return check_key(text(name), name)

        fallback = None
        if fallback_url := text("ASSISTANT_FALLBACK_URL"):
            fallback = Endpoint(fallback_url, text("ASSISTANT_FALLBACK_MODEL"), key("ASSISTANT_FALLBACK_API_KEY"))
        return cls(
            llm=Endpoint(url, text("ASSISTANT_LLM_MODEL"), key("ASSISTANT_LLM_API_KEY")),
            fallback=fallback,
            vision=env_flag(env.get("ASSISTANT_LLM_VISION")),
            thinking=env_flag(env.get("ASSISTANT_THINKING")),
            max_tool_calls=number("ASSISTANT_MAX_TOOL_CALLS", cls.max_tool_calls),
            answer_timeout=number("ASSISTANT_ANSWER_TIMEOUT", cls.answer_timeout),
            stall_timeout=number("ASSISTANT_STALL_TIMEOUT", cls.stall_timeout),
            result_budget=number("ASSISTANT_RESULT_BUDGET", cls.result_budget),
            context_budget=number("ASSISTANT_CONTEXT_BUDGET", cls.context_budget),
            max_messages=number("ASSISTANT_MAX_MESSAGES", cls.max_messages),
            max_tokens=number("ASSISTANT_MAX_TOKENS", cls.max_tokens),
            concurrency=number("ASSISTANT_CONCURRENCY", cls.concurrency),
            queue_timeout=number("ASSISTANT_QUEUE_TIMEOUT", cls.queue_timeout),
            max_queue=number("ASSISTANT_MAX_QUEUE", cls.max_queue),
            rate_limit=number("ASSISTANT_RATE_LIMIT", cls.rate_limit),
            rate_window=number("ASSISTANT_RATE_WINDOW", cls.rate_window),
            own_keys=OwnKeys.from_env(env),
        )


@dataclass
class OwnKeys:
    """People's own model keys: where they are kept, which providers take them, and
    the limits on the questions they answer.

    Those questions have a line and a rate limit of their own. They do not wait
    for the operator's model, whose slots are few, and the provider and the
    person's spending limit their pace; these limits keep this server and
    Babamul, where their tools run, from being swamped all the same.
    """

    store: KeyStore
    providers: dict[str, p.Provider]
    max_days: int = 90
    concurrency: int = 8
    max_queue: int = 16
    rate_limit: int = 40
    rate_window: float = 600.0
    max_tokens: int = 8192
    max_images: int = 6

    @classmethod
    def from_env(cls, env) -> "OwnKeys | None":
        """The settings in ASSISTANT_KEYS_* and ASSISTANT_BYOK_*, or None when no store is set up."""
        store = KeyStore.from_env(env)
        if store is None:
            return None
        own = cls(
            store=store,
            providers=p.from_env(env),
            max_days=env_number(env, "ASSISTANT_KEYS_MAX_DAYS", cls.max_days),
            concurrency=env_number(env, "ASSISTANT_BYOK_CONCURRENCY", cls.concurrency),
            max_queue=env_number(env, "ASSISTANT_BYOK_MAX_QUEUE", cls.max_queue),
            rate_limit=env_number(env, "ASSISTANT_BYOK_RATE_LIMIT", cls.rate_limit),
            rate_window=env_number(env, "ASSISTANT_BYOK_RATE_WINDOW", cls.rate_window),
            max_tokens=env_number(env, "ASSISTANT_BYOK_MAX_TOKENS", cls.max_tokens),
            max_images=env_number(env, "ASSISTANT_BYOK_MAX_IMAGES", cls.max_images),
        )
        if own.max_days < KEY_DAYS[0]:
            raise ValueError(f"ASSISTANT_KEYS_MAX_DAYS must be at least {KEY_DAYS[0]}")
        return own

    @property
    def days(self) -> list[int]:
        return [d for d in KEY_DAYS if d <= self.max_days]

    @property
    def default_days(self) -> int:
        return DEFAULT_KEY_DAYS if DEFAULT_KEY_DAYS in self.days else self.days[-1]

    def public(self) -> dict:
        return {
            "providers": [provider.public() for provider in self.providers.values()],
            "days": self.days,
            "default_days": self.default_days,
        }


@dataclass(frozen=True)
class Model:
    """What answers one question, and how it is asked.

    endpoints are tried in turn; none means the operator's model, with its
    fallback. The rest is what the tool loop needs to know about the model:
    whether it can see images, and how much it may read and write.
    """

    endpoints: tuple[Endpoint, ...] = ()
    vision: bool = False
    thinking: bool = False
    max_tokens: int = 1024
    context_budget: int = 60_000
    # Images the model is shown at most, the newest; 0 for no limit.
    max_images: int = 0
    # For a person's own key: the provider's id and name, and what the status line calls it.
    provider: str = ""
    name: str = ""
    label: str = ""


# The prompt.

SURVEYS = ("ZTF", "LSST")
OBJECT_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
PAGE_PATH = re.compile(r"/[A-Za-z0-9_./:%-]{0,200}")


def now_line() -> str:
    """The current time, in the two forms the tools take.

    A model guessing the date from memory reaches for one in its training data,
    and then searches a night with no alerts.
    """
    jd = s.now_jd()
    nights = []
    for survey in s.NIGHTS_UTC:
        w = s.night_windows(survey)
        last = w["last_night"]
        text = f"for {survey} {last['start']} to {last['end']}"
        if "tonight_so_far" in w:
            text += f" (tonight's {survey} night began at {w['tonight_so_far']['start']})"
        nights.append(text)
    return (
        f"The time is {s.jd_to_iso(jd)[:16].replace('T', ' ')} UTC, which is JD {jd:.4f}. "
        "Any window you pass to a tool should be near it. "
        f"\"Last night\" means the last night that has ended: {'; '.join(nights)} (UTC)."
    )


def describe_context(context: dict | None) -> str | None:
    """What the person is looking at, from the page's own description of itself.

    Only shapes an id or a path can have get through: this is text from the
    browser going into the system prompt.
    """
    if not isinstance(context, dict):
        return None
    survey = context.get("survey")
    survey = survey.upper() if isinstance(survey, str) and survey.upper() in SURVEYS else None
    object_id = context.get("object_id")
    if isinstance(object_id, str) and OBJECT_ID.fullmatch(object_id.strip()):
        return f"the {survey} object {object_id.strip()}" if survey else f"the object {object_id.strip()}"
    path = context.get("path")
    if isinstance(path, str) and PAGE_PATH.fullmatch(path.strip()):
        return f"the page {path.strip()} of the app"
    return None


def system_prompt(context: dict | None = None, vision: bool = False) -> str:
    lines = [SYSTEM_PROMPT, VISION if vision else NO_VISION, INSTRUCTIONS, now_line()]
    if where := describe_context(context):
        lines.append(f"They are looking at {where}.")
    return "\n\n".join(lines)


# Ported from SkyPortal: fitting results into the context, and streamed replies.

# Below this a per-item share carries nothing worth reading.
MIN_ITEM_SHARE = 300


def _reparsed(text):
    """A condensed result as JSON where it still parses, else as its own text."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def condense(text: str, budget: int = 6000) -> str:
    """Fit a tool result into a context budget without handing back broken JSON."""
    if len(text) <= budget:
        return text

    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return text[:budget] + f"\n... truncated, {len(text)} characters in total"

    if isinstance(payload, list):
        kept = []
        for item in payload:
            kept.append(item)
            if len(json.dumps(kept)) > budget:
                kept.pop()
                break
        # Two items kept whole answer less than every item cut down, so take
        # whichever form names more of them.
        room = max(1, min(len(payload), budget // MIN_ITEM_SHARE))
        if len(kept) >= room:
            note = f"{len(payload) - len(kept)} more of {len(payload)} not shown"
            return json.dumps({"items": kept, "note": note})
        shown = payload[:room]
        share = budget // len(shown)
        items = [_reparsed(condense(json.dumps(item, default=str), share)) for item in shown]
        left = len(payload) - len(shown)
        note = "each item cut down to fit; asking for more per page will not add to it"
        if left:
            note = f"{len(shown)} of {len(payload)} shown, " + note
        return json.dumps({"items": items, "note": note})

    if not isinstance(payload, dict):
        return text[:budget]

    for key, value in payload.items():
        if _numbers(value):
            payload[key] = {"n": len(value), "min": min(value), "max": max(value)}
    # Unlike SkyPortal's, small fields stay whole and the big ones share what is
    # left. Cut to the whole budget each, one long list (the rows of a search)
    # still overflows next to the counts, and is then dropped outright.
    if (cap := _share(payload, budget)) is not None:
        for key, value in payload.items():
            if len(json.dumps(value, default=str)) > cap:
                smaller = _shrunk(value, max(cap - NOTE_ROOM, MIN_ITEM_SHARE // 2))
                if smaller is not None:
                    payload[key] = smaller

    dropped = []
    by_size = sorted(payload, key=lambda k: -len(json.dumps(payload[k], default=str)))
    for key in by_size:
        current = json.dumps(payload, default=str)
        if len(current) <= budget:
            return current
        payload.pop(key)
        dropped.append(key)
        payload["_dropped"] = f"fields too large to show: {', '.join(dropped)}"
    return json.dumps(payload, default=str)


# What condensing a list adds around the items it keeps: {"items": ..., "note": ...}.
NOTE_ROOM = 120


def _numbers(value) -> bool:
    """A long list of plain numbers, which an answer can use the range of, never the values."""
    return (
        isinstance(value, list)
        and len(value) >= 20
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def _share(payload: dict, budget: int) -> int | None:
    """The most any one field may take so that all of them fit, or None if they already do."""
    sizes = sorted(len(json.dumps(value, default=str)) for value in payload.values())
    room = budget - sum(len(json.dumps(key)) + 2 for key in payload) - 2
    for i, size in enumerate(sizes):
        if size * (len(sizes) - i) > room:
            return room // (len(sizes) - i)
        room -= size
    return None


def _shrunk(value, budget):
    """A smaller stand-in for an oversized value, or None if there isn't one.

    A field is worth shrinking before it is worth dropping: a light curve cut to
    its first rows still answers a question, and a deleted one answers nothing.
    """
    if _numbers(value):
        return {"n": len(value), "min": min(value), "max": max(value)}
    if not isinstance(value, (list, dict)) or not value:
        return None
    if len(json.dumps(value, default=str)) <= budget:
        return None
    return _reparsed(condense(json.dumps(value, default=str), budget))


def condense_table(text: str, budget: int) -> str | None:
    """A CSV table cut to its header and its newest rows, or None if text is not one.

    Tables come oldest row first (a light curve sorted by time), and the newest
    rows are the ones a question is usually about.
    """
    lines = text.rstrip("\n").split("\n")
    header, rows = lines[0], lines[1:]
    commas = header.count(",")
    if len(rows) < 2 or not commas or any(row.count(",") != commas for row in rows[:5]):
        return None
    kept, size = [], len(header) + 60
    for row in reversed(rows):
        if size + len(row) + 1 > budget:
            break
        kept.append(row)
        size += len(row) + 1
    return "\n".join([f"(the {len(rows) - len(kept)} oldest rows are not shown)", header, *reversed(kept)])


def condense_blocks(texts: list[str], budget: int) -> str:
    """Fit the text blocks of one tool result into budget, each cut down on its own.

    get_light_curve returns a JSON summary and then a CSV table. Cut as one
    text, the table would keep its oldest rows and lose its newest. As in
    condense(), small blocks stay whole and the big ones share what is left.
    """
    joined = "\n".join(texts)
    if len(joined) <= budget:
        return joined
    room = budget - len(texts) + 1
    cap = room
    for i, size in enumerate(sorted(len(t) for t in texts)):
        if size * (len(texts) - i) > room:
            cap = room // (len(texts) - i)
            break
        room -= size
    return "\n".join(t if len(t) <= cap else condense_table(t, cap) or condense(t, cap) for t in texts)


# A tool call written out as text. llama.cpp parses these only when tools are
# offered, so one in the final round would reach the person as markup.
TOOL_MARKUP = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.S)


def strip_tool_markup(text: str | None) -> str:
    return TOOL_MARKUP.sub("", text or "").strip()


def chat_payload(
    model, messages, tools, thinking=False, max_tokens=0, kind="llamacpp", extra=None, tool_choice=None
) -> dict:
    """The body of a streamed chat-completions request, for an endpoint of this kind.

    `tools` is omitted when empty rather than sent as []: the last round asks
    for an answer with no tools left to call, and a server that rejects an
    empty array turns that into a 400.

    max_tokens is what stops a reply that repeats itself: greedy decoding can
    loop, and the stall timeout never fires while it keeps talking. There is no
    repetition penalty, because it would also stop the model from writing an
    object id exactly as a tool gave it.

    A hosted provider ("openai") gets only standard fields: chat_template_kwargs
    is llama.cpp's own, and Anthropic refuses a temperature below 1 on its newer
    models, so the provider's sampling defaults stay. tool_choice is never a
    forced one, which Anthropic refuses; "none" lets the last round keep its
    tools, since Anthropic wants tools defined while the conversation holds
    calls to them. stream_options asks for the tokens used in the last chunk,
    since on a provider someone pays for them. extra fields come last and
    replace none of these.
    """
    if kind == "openai":
        payload = {"model": model, "messages": messages}
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            if tool_choice == "none":
                payload["tool_choice"] = "none"
        payload["stream_options"] = {"include_usage": True}
        for name, value in (extra or {}).items():
            payload.setdefault(name, value)
        return payload
    payload = {"model": model, "messages": messages, "temperature": 0}
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
    if not thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


# The fields of a streamed tool call that merge_deltas puts together itself.
CALL_FIELDS = frozenset({"index", "id", "type", "function"})
# The streamed reasoning fragments that join the block before them, and the field that grows.
REASONING_TEXT = {"reasoning.text": "text", "reasoning.summary": "summary"}


def merge_deltas(deltas) -> dict:
    """The message the deltas of a streamed chat completion add up to.

    Tool calls are put together by index. A part with no index is a call of
    its own: Gemini sends each call whole and without one, and adding it to
    call 0 would make one broken call of several.

    Some fields go back to the provider on the next round, as it sent them:
    any other field of a tool call (Gemini's extra_content, which holds its
    thought signature) and the message's reasoning_details (OpenRouter's
    reasoning blocks). A model that loses them may refuse the next round.
    Other message fields are dropped, llama.cpp's reasoning_content among
    them: some servers refuse it in a request.
    """
    content, indexed, loose, reasoning = [], {}, [], []
    for delta in deltas:
        content.append(delta.get("content") or "")
        for part in delta.get("tool_calls") or []:
            if part.get("index") is None:
                call = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                loose.append(call)
            else:
                call = indexed.setdefault(
                    part["index"],
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
            function = part.get("function") or {}
            call["id"] = part.get("id") or call["id"]
            call["function"]["name"] = function.get("name") or call["function"]["name"]
            call["function"]["arguments"] += function.get("arguments") or ""
            for name, value in part.items():
                if name not in CALL_FIELDS and value is not None:
                    call[name] = merged(call.get(name), value)
        for detail in delta.get("reasoning_details") or []:
            add_reasoning(reasoning, detail)
    message = {"role": "assistant", "content": "".join(content) or None}
    if calls := [indexed[index] for index in sorted(indexed)] + loose:
        message["tool_calls"] = calls
    if reasoning:
        message["reasoning_details"] = reasoning
    return message


def merged(old, new):
    """A field sent again in a later delta: objects are merged, anything else replaced."""
    if isinstance(old, dict) and isinstance(new, dict):
        return {**old, **{name: merged(old.get(name), value) for name, value in new.items()}}
    return new


def add_reasoning(details: list[dict], part) -> None:
    """Add one streamed reasoning_details fragment, the way OpenRouter's own client does.

    A block arrives in many small fragments, and the provider wants back the
    blocks it made, in order and unchanged. A text or summary fragment joins
    the block before it if that is of the same type, until a signature closes a
    text block. Encrypted blocks, and types not known here, are kept as they came.
    """
    if not isinstance(part, dict):
        return
    grows = REASONING_TEXT.get(part.get("type"))
    last = details[-1] if details else None
    if grows and last is not None and last.get("type") == part["type"] and not last.get("signature"):
        last[grows] = (last.get(grows) or "") + (part.get(grows) or "")
        for name, value in part.items():
            if name != grows and value is not None and last.get(name) is None:
                last[name] = value
        return
    details.append(dict(part))


IMAGES_LEFT_OUT = "The images the tools above returned were shown to the person. They are left out here, to save room."


def keep_newest_images(messages: list[dict], keep: int):
    """Leave out all but the newest keep images.

    Every image is read again in every later round, and on a person's own key
    each costs them, so a long chat with cutouts would cost more each round.
    """
    seen = 0
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        kept, dropped = [], 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                if seen >= keep:
                    dropped += 1
                    continue
                seen += 1
            kept.append(part)
        if dropped:
            images = [q for q in kept if isinstance(q, dict) and q.get("type") == "image_url"]
            message["content"] = kept if images else IMAGES_LEFT_OUT


def fit(messages: list[dict], budget: int, max_images: int = 0) -> list[dict]:
    """Shrink the oldest tool results until the conversation fits in budget characters,
    and keep only the newest max_images images when it is above 0.

    Ten results of 20,000 characters are more than a 32k-token model holds, and
    the server then refuses the whole request. The newest results matter most.
    """
    if max_images > 0:
        keep_newest_images(messages, max_images)

    def size(message):
        content = message.get("content")
        return len(content) if isinstance(content, str) else 0

    total = sum(size(m) for m in messages)
    for message in messages:
        if total <= budget:
            break
        if message.get("role") == "tool" and size(message) > SHRUNK_RESULT:
            shorter = condense(message["content"], SHRUNK_RESULT)
            total -= size(message) - len(shorter)
            message["content"] = shorter
    return messages


def openai_tool(tool) -> dict:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.input_schema},
    }


def split_content(result) -> tuple[list[str], list[dict]]:
    """A tool result as the texts the model reads and the images the person sees."""
    texts, images = [], []
    for block in result.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "image":
            images.append({"mime_type": block.mime_type, "data": block.data})
    return texts, images


LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "notice": logging.INFO, "warning": logging.WARNING}


class LocalContext(Context):
    """The Context a tool gets when the assistant calls it in this process.

    No MCP client is listening, so the plain Context refuses ctx.info() and
    ctx.report_progress() and the tool crashes. Here they go to the log instead,
    and a plugin works the same in the assistant as over /mcp.
    """

    async def log(self, level, data, *, logger_name=None) -> None:
        log.log(LOG_LEVELS.get(level, logging.ERROR), "tool log%s: %s", f" {logger_name}" if logger_name else "", data)

    async def report_progress(self, progress, total=None, message=None) -> None:
        log.debug("tool progress: %s of %s %s", progress, total, message or "")

    @property
    def headers(self):
        # As over stdio: the tool runs with no HTTP request of its own.
        return None


def crash_cause(exc: BaseException) -> BaseException:
    """What a tool really raised, under the SDK's wrapper and any task group."""
    while True:
        if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
            exc = exc.exceptions[0]
        elif isinstance(exc, UnexpectedToolError) and exc.__cause__ is not None:
            exc = exc.__cause__
        else:
            return exc


def image_message(images: list[dict]) -> dict:
    """The images of one round, for a model that can see."""
    parts = [{"type": "text", "text": "The images the tools above returned, which the person also sees."}]
    parts += [{"type": "image_url", "image_url": {"url": f"data:{i['mime_type']};base64,{i['data']}"}} for i in images]
    return {"role": "user", "content": parts}


def sse(event: dict) -> bytes:
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()


def announcing(emit, message: str):
    """emit, saying message once the model has taken the question.

    Said then and not before, so that a key the provider refuses outright
    still gets a plain error instead of a stream that starts and then fails.
    """
    said = False

    def wrapped(event):
        nonlocal said
        emit(event)
        if not said and isinstance(event, dict) and event.get("type") == "_started":
            said = True
            emit({"type": "status", "message": message})

    return wrapped


def add_usage(spent: dict, usage) -> None:
    """Add a reply's usage, in OpenAI's names or Anthropic's, to the question's."""
    if not isinstance(usage, dict):
        return
    for ours, theirs in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
        value = usage.get(theirs, usage.get(ours))
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            spent[ours] = spent.get(ours, 0) + value


# Errors.


class ModelError(Exception):
    """The model failed. `message` is for the person; `switchable` means another
    endpoint (or another try) might do better. A failure of a person's own key
    has a code, for the web app to offer a fix, and its own status."""

    def __init__(self, message: str, switchable: bool = False, detail: str = "", code: str = "", status: int = 503):
        super().__init__(detail or message)
        self.message = message
        self.switchable = switchable
        self.code = code
        self.status = status


class OutOfTime(Exception):
    """The answer's time ran out partway through a round; `said` is what it got out."""

    def __init__(self, said: str = ""):
        super().__init__("out of time")
        self.said = said


class Stalled(Exception):
    """The model went quiet partway through its reply."""


class BabamulDown(Exception):
    pass


class BadRequest(Exception):
    """A request refused before anyone is asked. code tells the web app what to offer."""

    def __init__(self, message: str, status: int = 400, code: str = "", headers: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.headers = headers


@dataclass(frozen=True)
class Who:
    """Who asks: person is what their limits are kept by, account their Babamul
    user id. Only an account, never a token's stand-in, may hold keys."""

    person: str
    account: str | None


@dataclass
class Failure:
    """The end of a question that failed. Before the stream starts it becomes a
    plain JSON error with this status; after, an error event."""

    status: int
    message: str
    code: str = ""


# An error type or code worth logging: a short word, never a sentence.
ERROR_LABEL = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


def error_fields(body) -> str:
    """The type and code of a model server's error, from its body, for the logs.

    Never its message or anything else in it: a provider may quote the request
    back, the key it refused included. body is the raw bytes or text of an
    error, or a parsed one; a body that is not JSON gives nothing.
    """
    if isinstance(body, (bytes, str)):
        try:
            body = json.loads(body)
        except (ValueError, RecursionError):
            body = None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    if not isinstance(body, dict):
        return "no error type"
    fields = []
    for name in ("type", "code"):
        value = body.get(name)
        if (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, str) and ERROR_LABEL.fullmatch(value)
        ):
            fields.append(f"{name} {value}")
    return ", ".join(fields) or "no error type"


def refused(status: int, detail: str) -> ModelError:
    if status == 429 or status >= 500:
        return ModelError(UNREACHABLE, switchable=True, detail=detail)
    if status in (401, 403, 404):
        return ModelError(MISCONFIGURED, detail=detail)
    return ModelError(REFUSED, detail=detail)


# What the owner of a key reads when it fails. Theirs to fix, not the Babamul team's,
# and each says the shared model is still there, since nothing moves to it by itself.
OWN_MESSAGES = {
    p.KEY_MISSING: (
        "You chose to answer with your own {name} key, but it is missing, expired or can no longer be read. "
        "Save it again on your Profile, or ask the shared model."
    ),
    p.KEY_REFUSED: "{name} did not accept your key. Check it on your Profile, or ask the shared model.",
    p.NO_CREDIT: (
        "Your {name} account is out of credit or reached its spending limit. "
        "Add credit or raise the limit, or ask the shared model."
    ),
    p.BUSY: "{name} is limiting how fast your key can ask. Try again in a minute, or ask the shared model.",
    p.MODEL_UNAVAILABLE: (
        "Your {name} key cannot use {model}. Pick another model on your Profile, or ask the shared model."
    ),
    p.DOWN: "{name} is not answering right now. Try again in a minute, or ask the shared model.",
}


def own_failure(code: str, name: str, model: str = "", detail: str = "", switchable: bool = False) -> ModelError:
    """A failure of a person's own key. Never a 401, which the web app takes for a
    Babamul login gone stale: a 503 when the provider is down, else a 422."""
    message = OWN_MESSAGES[code].format(name=name, model=model or "that model")
    return ModelError(
        message, switchable=switchable, detail=detail or code, code=code, status=503 if code == p.DOWN else 422
    )


def own_error(exc: ModelError, model: "Model") -> ModelError:
    """A failure on a person's own key, with a code: their provider not answering, or
    stopping partway, is theirs to retry or to ask the shared model instead."""
    if exc.code or exc.message not in (UNREACHABLE, STALLED):
        return exc
    message = exc.message if exc.message == STALLED else OWN_MESSAGES[p.DOWN].format(name=model.name)
    return ModelError(message, detail=str(exc), code=p.DOWN, status=503)


def own_refused(model: "Model", body, status: int | None, detail: str, said: bool) -> ModelError:
    """A provider's error for a question on a person's key. Asked again at once only
    when the provider itself failed (a 5xx or 529) before any text, never after a 429."""
    code = p.classify(model.provider, body, status)
    if code is None:
        return ModelError(REFUSED, detail=detail)
    status = p.status_of(p.error_of(body), status)
    again = code == p.DOWN and status is not None and status >= 500 and not said
    return own_failure(code, model.name, model.endpoints[0].model, f"{detail}: {code}", again)


# Limits.


class RateLimiter:
    """At most `limit` questions per `window` seconds for each person."""

    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self.hits: dict[str, deque] = {}

    def retry_after(self, key: str, now: float | None = None) -> float:
        """Seconds until key may ask again, or 0 when it may ask now (which counts)."""
        now = time.monotonic() if now is None else now
        hits = self.hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return hits[0] + self.window - now
        hits.append(now)
        return 0.0

    def refund(self, key: str):
        if hits := self.hits.get(key):
            hits.pop()


class Place:
    """One question's place: answering, or waiting in line."""

    def __init__(self):
        self.admitted = False
        self.moved = asyncio.Event()


class Line:
    """The questions being answered, at most `capacity`, and the line of those
    waiting to start, first come first served.

    A question keeps its place for its whole answer, tool calls included. Taking
    turns for each model round instead turned a burst into round-robin: twenty
    questions each got one round in turn, and all of them ran out of time
    together. The cost is a model slot left idle while a question's tools run:
    usually a few seconds, though a slow alert search can take a minute. In
    exchange every question let in can finish, and llama.cpp keeps its prompt
    cached in the slot, so the next round does not read the whole prompt again.
    """

    def __init__(self, capacity: int, max_waiting: int):
        self.capacity = max(1, capacity)
        self.max_waiting = max(0, max_waiting)
        self.answering: set[Place] = set()
        self.waiting: list[Place] = []

    def join(self) -> Place | None:
        """A place for a new question, or None when the line is full."""
        place = Place()
        if len(self.answering) < self.capacity and not self.waiting:
            place.admitted = True
            self.answering.add(place)
        elif len(self.waiting) < self.max_waiting:
            self.waiting.append(place)
        else:
            return None
        return place

    def ahead(self, place: Place) -> int:
        return self.waiting.index(place)

    def leave(self, place: Place):
        """Give up a place, answering or waiting, and let the next in line start."""
        self.answering.discard(place)
        if place in self.waiting:
            self.waiting.remove(place)
        moved = list(self.waiting)
        while self.waiting and len(self.answering) < self.capacity:
            first = self.waiting.pop(0)
            first.admitted = True
            self.answering.add(first)
        for other in moved:
            other.moved.set()


def waiting_message(ahead: int) -> str:
    if not ahead:
        return "Waiting for the model, you are next"
    return f"Waiting for the model, {ahead} question{'s' * (ahead > 1)} ahead of you"


class EventStream(StreamingResponse):
    """A stream of events that stops the question as soon as the client leaves.

    Starlette only watches for a disconnect on ASGI before 2.4, and otherwise
    notices at the next write. A stopped question would then keep its model slot
    and its place in line until it next said something, so this always watches.
    """

    def __init__(self, content, on_close):
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self.on_close = on_close

    async def __call__(self, scope, receive, send):
        try:
            async with anyio.create_task_group() as tg:

                async def stream():
                    try:
                        await self.stream_response(send)
                    except OSError:
                        pass  # the client left mid-write
                    tg.cancel_scope.cancel()

                tg.start_soon(stream)
                await self.listen_for_disconnect(receive)
                tg.cancel_scope.cancel()
        finally:
            self.on_close()


class Assistant:
    # How long a new question waits for the one it replaces to wind down.
    stop_grace = 3.0

    def __init__(self, config: Config, *, http: httpx.AsyncClient | None = None, babamul_url: str | None = None):
        self.config = config
        self.http = http
        self.babamul_url = (babamul_url or base_url_from_env()).rstrip("/")
        self.line = Line(config.concurrency, config.max_queue)
        self.limiter = RateLimiter(config.rate_limit, config.rate_window)
        self.own = config.own_keys
        self.saves = RateLimiter(SAVES_PER_HOUR, 3600.0)
        if self.own is not None:
            # Questions on people's own keys never wait for the operator's model.
            self.own_line = Line(self.own.concurrency, self.own.max_queue)
            self.own_limiter = RateLimiter(self.own.rate_limit, self.own.rate_window)
        # Person -> (ticket, stop) of their running question; stop ends it.
        self.active: dict[str, tuple[object, Callable[[], Awaitable[None]]]] = {}
        # Person -> the newest of their questions still taking over from the last.
        self.claims: dict[str, object] = {}
        # sha256 of a token -> (expiry, who or None when Babamul refused it).
        self.tokens: dict[str, tuple[float, Who | None]] = {}
        self._main_skipped_until = 0.0
        self._tools: list[dict] | None = None

    @asynccontextmanager
    async def running(self):
        """Holds the HTTP client for the model and Babamul while the app runs, and
        sweeps out expired keys."""
        sweeper = asyncio.create_task(self.sweep_keys()) if self.own else None
        try:
            if self.http is not None:
                yield
            else:
                async with settings["http_client"]() as http:
                    self.http = http
                    try:
                        yield
                    finally:
                        self.http = None
        finally:
            if sweeper is not None:
                sweeper.cancel()
                await asyncio.gather(sweeper, return_exceptions=True)

    async def sweep_keys(self):
        """Forget expired keys now and then. Reading a person's keys forgets theirs too,
        but someone who never comes back would keep theirs."""
        while True:
            try:
                if gone := await asyncio.to_thread(self.own.store.sweep):
                    log.info("forgot %d expired model key%s", gone, "s" * (gone > 1))
            except sqlite3.Error as exc:
                log.error("could not sweep expired model keys: %s", type(exc).__name__)
            await asyncio.sleep(SWEEP_EVERY)

    # Routes.

    async def health(self, request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "model": self.config.llm.model,
                "vision": self.config.vision,
                "own_keys": self.own.public() if self.own else None,
            }
        )

    async def chat(self, request: Request):
        try:
            token = bearer(request)
            messages, context, use = parse_body(await read_body(request))
            who = await self.identify(token)
            model = await self.model_for(who, token, use)
        except BadRequest as exc:
            return error(exc.status, str(exc), headers=exc.headers, code=exc.code)
        except ModelError as exc:
            # Their chosen key cannot be used. Nothing moves to the shared model by
            # itself: they would not know which model answered, or why.
            log.info("user %s cannot use their own key: %s", who.person, exc.code)
            return error(exc.status, exc.message, code=exc.code)
        except sqlite3.Error as exc:
            return store_down(exc)
        person = who.person
        if model.provider:
            line, limiter = self.own_line, self.own_limiter
        else:
            line, limiter = self.line, self.limiter

        if (wait := limiter.retry_after(person)) > 0:
            minutes = math.ceil(wait / 60)
            return error(
                429,
                f"That is a lot of questions in a short time. Try again in {minutes} minute{'s' * (minutes > 1)}.",
                headers={"Retry-After": str(math.ceil(wait))},
            )
        # One question per person: a new one replaces the one running. The panel
        # shows one answer at a time, and the old one may have no reader left (a
        # tunnel can drop the browser's stream without closing ours), so turning
        # the new one away would lock the person out until the old one ends.
        claim = object()
        self.claims[person] = claim
        try:
            await self.replace(person, claim)
            if self.claims.get(person) is not claim:
                # A newer question of theirs came in meanwhile, and it wins. This
                # one was asked all the same, so it still counts toward the limit.
                return error(409, REPLACED)
            # Taken here, after the last await, so a burst cannot overfill the line.
            if (place := line.join()) is None:
                limiter.refund(person)
                log.warning("turned user %s away: %d questions already waiting", person, len(line.waiting))
                return error(429, LINE_FULL, headers={"Retry-After": "60"})

            ticket = object()
            queue: asyncio.Queue = asyncio.Queue()
            task = asyncio.create_task(self.run(messages, context, token, person, queue.put_nowait, place, model, line))

            async def stop():
                # The Failure ends the old stream even if its task never started.
                queue.put_nowait(Failure(409, REPLACED))
                task.cancel()
                await asyncio.wait({task}, timeout=self.stop_grace)
                if task.done():
                    # A task cancelled before it started never left the line itself.
                    line.leave(place)

            self.active[person] = (ticket, stop)
        finally:
            if self.claims.get(person) is claim:
                del self.claims[person]

        def close():
            task.cancel()
            # Also here: a task cancelled before it starts never runs its own cleanup.
            line.leave(place)
            if self.active.get(person, (None,))[0] is ticket:
                del self.active[person]

        try:
            first = await self.first_event(request, queue)
        except BaseException:
            close()
            raise
        if first is None:
            close()
            return error(499, "The client left before the answer started.")
        if isinstance(first, Failure):
            close()
            if first.message != REPLACED:
                # Our failure, not their choice: it does not count. A replaced one does,
                # or asking again before each answer starts would never reach the limit.
                limiter.refund(person)
            return error(first.status, first.message, code=first.code)
        return EventStream(self.frames(first, queue), close)

    async def first_event(self, request: Request, queue: asyncio.Queue):
        """The question's first event, waited for so that a model that is down
        gets a 503 instead of a stream with only an error in it.

        None if the client leaves first: nothing else would notice before the
        stream starts, and the question would keep its place in line. A model
        slow to take it still starts the stream after HEARTBEAT seconds.
        """
        getter = asyncio.ensure_future(queue.get())
        give_up = time.monotonic() + HEARTBEAT
        try:
            while True:
                done, _ = await asyncio.wait({getter}, timeout=0.5)
                if done:
                    return getter.result()
                if await request.is_disconnected():
                    return None
                if time.monotonic() >= give_up:
                    return {"type": "status", "message": "Waiting for the model"}
        finally:
            getter.cancel()

    async def frames(self, first, queue: asyncio.Queue):
        item = first
        while item is not None:
            if isinstance(item, Failure):
                yield sse({"type": "error", "message": item.message, **({"code": item.code} if item.code else {})})
                return
            if not item["type"].startswith("_"):
                yield sse(item)
            try:
                async with asyncio.timeout(HEARTBEAT):
                    item = await queue.get()
            except TimeoutError:
                item = {"type": "status", "message": "Still working"}

    # Who is asking.

    async def person(self, token: str) -> str | None:
        """The id a person's limits are kept by, or None when Babamul refuses the token."""
        who = await self.who(token)
        return who.person if who else None

    async def identify(self, token: str, fresh: bool = False) -> "Who":
        """Who the token belongs to, or BadRequest with the status to answer."""
        try:
            who = await self.who(token, fresh)
        except BabamulDown:
            raise BadRequest("Could not reach Babamul to check your login. Try again in a minute.", 503) from None
        if who is None:
            raise BadRequest("Babamul did not accept your login. Log in again.", 401)
        return who

    async def who(self, token: str, fresh: bool = False) -> "Who | None":
        """Who the token belongs to, or None when Babamul refuses it.

        GET /profile is the cheapest call Babamul authenticates, and the id lets
        one person's login and API tokens share their limits. Answers are kept
        for a while; fresh asks again, for changes to a person's keys.
        """
        key = hashlib.sha256(token.encode()).hexdigest()
        now = time.monotonic()
        if not fresh and (hit := self.tokens.get(key)) and hit[0] > now:
            return hit[1]
        try:
            response = await self.http.get(
                f"{self.babamul_url}/profile",
                headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT},
                timeout=15,
            )
        except httpx.HTTPError as exc:
            # The type only: an httpx error can quote the Authorization header.
            raise BabamulDown(type(exc).__name__) from None
        if response.status_code in (401, 403):
            person, ttl = None, BAD_TOKEN_TTL
        elif response.status_code >= 400:
            raise BabamulDown(f"profile returned {response.status_code}")
        else:
            try:
                body = response.json()
            except ValueError as exc:
                raise BabamulDown("profile is not JSON") from exc
            data = body.get("data", body) if isinstance(body, dict) else {}
            account = (data or {}).get("id") if isinstance(data, dict) else None
            account = str(account) if account else None
            # Without an id, limits are kept by the token; its keys, if any, are not reachable.
            person, ttl = Who(account or key, account), TOKEN_TTL
        if len(self.tokens) >= TOKEN_CACHE:
            self.tokens = {k: v for k, v in self.tokens.items() if v[0] > now}
            if len(self.tokens) >= TOKEN_CACHE:
                self.tokens = dict(list(self.tokens.items())[TOKEN_CACHE // 2 :])
        self.tokens[key] = (now + ttl, person)
        return person

    async def replace(self, person: str, claim: object) -> None:
        """Stop person's running question, if any, and give it a moment to wind down.

        A loop, since another of their questions may start while this one waits;
        it ends once a newer question has claimed the person, which then wins.
        """
        while self.claims.get(person) is claim and (entry := self.active.pop(person, None)) is not None:
            log.info("user %s asked again, so their running question stops", person)
            await entry[1]()

    # People's own keys. Write-only: no route answers with a key, or anything sealed
    # from one. Only the web app's login tokens may use them, and a change asks
    # Babamul afresh who the token belongs to.

    async def keys_route(self, request: Request, handle, fresh: bool = True) -> Response:
        """Run handle(request, account) for a person logged in to the web app."""
        if self.own is None:
            return error(404, "Personal model keys are not turned on here.", code="not_found")
        try:
            token = bearer(request)
            if not is_login(token):
                raise BadRequest(
                    "Manage model keys from the Babamul web app. Babamul API tokens cannot use them.",
                    403,
                    "login_required",
                )
            who = await self.identify(token, fresh)
            if not who.account:
                # Never by the token's stand-in id: a new login would lose the keys, or find another's.
                raise BadRequest(
                    "Babamul did not say which account this is, so keys cannot be kept for it.", 403, "no_account"
                )
            return await handle(request, who.account)
        except BadRequest as exc:
            return error(exc.status, str(exc), headers=exc.headers, code=exc.code)
        except sqlite3.Error as exc:
            return store_down(exc)

    def provider(self, name) -> p.Provider:
        if isinstance(name, str) and (provider := self.own.providers.get(name)):
            return provider
        raise BadRequest("That model provider is not offered here.", 404, "not_found")

    def shown_keys(self, account: str) -> dict:
        """The person's choice and keys, as GET /keys answers. A choice of a provider no
        longer offered counts as the shared model."""
        store, offered = self.own.store, list(self.own.providers)
        use = store.use(account)
        keys = [e for e in store.entries(account) if e["provider"] in offered]
        keys.sort(key=lambda e: offered.index(e["provider"]))
        return {"use": use if use in offered else SHARED, "keys": keys}

    async def list_keys(self, request: Request) -> Response:
        async def handle(request, account):
            return JSONResponse(await asyncio.to_thread(self.shown_keys, account))

        return await self.keys_route(request, handle, fresh=False)

    async def save_key(self, request: Request) -> Response:
        async def handle(request, account):
            provider = self.provider(request.path_params.get("provider"))
            body = await key_body(request)
            try:
                key = provider.check_format(body.get("key"))
            except p.BadKey as exc:
                raise BadRequest(str(exc), 400, "bad_format") from None
            model = self.model_named(provider, body.get("model", provider.default_model))
            days = body.get("days", self.own.default_days)
            if isinstance(days, bool) or not isinstance(days, int) or days not in self.own.days:
                choices = ", ".join(map(str, self.own.days))
                raise BadRequest(f"Keep a key for one of {choices} days.", 400, "bad_days")
            workspace = body.get("workspace_id") if provider.workspace_field else None
            if isinstance(workspace, str) and workspace.strip():
                workspace = workspace.strip()
                if not p.WORKSPACE_ID.fullmatch(workspace):
                    raise BadRequest("A workspace id looks like wrkspc_ and letters and digits.", 400, "bad_format")
            elif workspace is not None and workspace != "":
                raise BadRequest("workspace_id must be text.", 400, "bad_format")
            else:
                workspace = None
            use = body.get("use", True)
            if not isinstance(use, bool):
                raise BadRequest("use must be true or false.", 400, "bad_format")

            # Counted before the provider is asked, failed checks included.
            if (wait := self.saves.retry_after(account)) > 0:
                minutes = math.ceil(wait / 60)
                raise BadRequest(
                    f"That is a lot of keys in an hour. Try again in {minutes} minute{'s' * (minutes > 1)}.",
                    429,
                    "too_many_saves",
                    {"Retry-After": str(math.ceil(wait))},
                )
            try:
                await provider.check_key(self.http, key, model, workspace)
            except p.KeyRejected as exc:
                raise BadRequest(str(exc), 422, "rejected") from None
            except p.ProviderUnreachable as exc:
                raise BadRequest(str(exc), 502, "provider_unreachable") from None
            entry = await asyncio.to_thread(self.own.store.save, account, provider.id, key, model, days, workspace, use)
            log.info("user %s saved their %s key for %d days", account, provider.name, days)
            return JSONResponse(entry)

        return await self.keys_route(request, handle)

    def model_named(self, provider: p.Provider, model) -> str:
        if model not in provider.models:
            offered = ", ".join(provider.models)
            raise BadRequest(f"Pick one of the {provider.name} models offered: {offered}.", 400, "bad_model")
        return model

    async def choose(self, request: Request) -> Response:
        async def handle(request, account):
            body = await key_body(request)
            use = body.get("use")
            if not isinstance(use, str):
                raise BadRequest('use is "shared" or a provider.', 400, "bad_format")
            if use == SHARED:
                await asyncio.to_thread(self.own.store.set_use, account, SHARED)
                return JSONResponse({"use": SHARED})
            provider = self.provider(use)
            model = body.get("model")
            if model is not None:
                self.model_named(provider, model)
            entries = await asyncio.to_thread(self.own.store.entries, account)
            if not any(e["provider"] == provider.id and e["state"] == "ok" for e in entries):
                raise BadRequest(f"Save your {provider.name} key first.", 409, "no_key")
            await asyncio.to_thread(self.own.store.set_use, account, provider.id, model)
            log.info("user %s chose their %s key", account, provider.name)
            return JSONResponse({"use": provider.id})

        return await self.keys_route(request, handle)

    async def delete_key(self, request: Request) -> Response:
        async def handle(request, account):
            provider = self.provider(request.path_params.get("provider"))
            await asyncio.to_thread(self.own.store.delete, account, provider.id)
            log.info("user %s removed their %s key", account, provider.name)
            return Response(status_code=204)

        return await self.keys_route(request, handle)

    async def delete_keys(self, request: Request) -> Response:
        async def handle(request, account):
            await asyncio.to_thread(self.own.store.delete_all, account)
            log.info("user %s removed all their keys", account)
            return Response(status_code=204)

        return await self.keys_route(request, handle)

    # Answering.

    async def run(self, messages, context, token, person, emit, place: Place, model: Model, line: Line):
        """Answer one question once it has its turn in line, every event and then None into emit."""
        started = time.monotonic()
        try:
            if not await self.wait_turn(place, emit, line):
                log.warning("user %s gave up waiting in line after %.0f s", person, time.monotonic() - started)
                emit(Failure(429, WAITED_TOO_LONG))
                return
            waited = time.monotonic() - started
            if model.provider:
                await self.note_key(person, model, "touch")
            calls = await self.answer(messages, context, token, emit, model)
            log.info(
                "answered user %s in %.1f s (%.1f s in line) with %d tool calls%s",
                person,
                time.monotonic() - started,
                waited,
                calls,
                f" on their {model.name} key" if model.provider else "",
            )
        except ModelError as exc:
            if model.provider:
                exc = own_error(exc, model)
            log.warning("model failed for user %s: %s", person, exc)
            if exc.code == p.KEY_REFUSED:
                # Shown on their Profile, and not tried again until they save it anew.
                await self.note_key(person, model, "mark", keystore.REFUSED)
            emit(Failure(exc.status, exc.message, exc.code))
        except Exception as exc:
            if model.provider:
                # The traceback, without the key: an error's message may quote a request.
                text = "".join(traceback.format_exception(exc))
                if key := model.endpoints[0].api_key:
                    text = text.replace(key, "(their key)")
                log.error("the assistant failed for user %s on their own key:\n%s", person, text.rstrip())
            else:
                log.exception("the assistant failed for user %s", person)
            emit(Failure(500, BROKEN))
        finally:
            line.leave(place)
            emit(None)

    async def note_key(self, person: str, model: Model, what: str, *args):
        """Write down the use of a person's key; a store that fails does not stop the answer."""
        try:
            await asyncio.to_thread(getattr(self.own.store, what), person, model.provider, *args)
        except sqlite3.Error as exc:
            log.error("could not note the use of a key: %s", type(exc).__name__)

    async def wait_turn(self, place: Place, emit, line: Line) -> bool:
        """Wait for place to reach the model, telling the person where they are in
        line. False when that takes longer than the queue timeout."""
        give_up = time.monotonic() + self.config.queue_timeout
        while not place.admitted:
            if (left := give_up - time.monotonic()) <= 0:
                return False
            emit({"type": "status", "message": waiting_message(line.ahead(place))})
            place.moved.clear()
            try:
                # Said again now and then, or the heartbeat's "Still working" replaces it.
                async with asyncio.timeout(min(left, HEARTBEAT / 2)):
                    await place.moved.wait()
            except TimeoutError:
                pass
        return True

    async def tools(self) -> list[dict]:
        """The read-only tools, as the chat API wants them."""
        if self._tools is None:
            listed = await mcp.list_tools()
            self._tools = [openai_tool(t) for t in listed if t.annotations and t.annotations.read_only_hint]
        return self._tools

    def shared_model(self, label: str = "") -> Model:
        """The operator's model, which answers everyone who has not chosen their own.
        With a label, the status line says so."""
        cfg = self.config
        return Model(
            vision=cfg.vision,
            thinking=cfg.thinking,
            max_tokens=cfg.max_tokens,
            context_budget=cfg.context_budget,
            label=label,
        )

    async def model_for(self, who: Who, token: str, use: str | None) -> Model:
        """The model that answers this question: the shared one, or the person's own key
        when they chose it and ask from the web app, not with an API token.

        Raises ModelError (a 422 with a code) when they chose their key and it
        cannot be used. The key is opened here for this question only, and goes
        no further than its Endpoint.
        """
        if self.own is None or use == SHARED or not who.account or not is_login(token):
            return self.shared_model()
        store = self.own.store
        try:
            choice = await asyncio.to_thread(store.use, who.account)
        except sqlite3.Error as exc:
            # The shared model never depends on the key store. The status line says
            # why, since this person may have chosen their own key.
            log.error("the key store failed, so user %s gets the shared model: %s", who.person, type(exc).__name__)
            return self.shared_model(STORE_DOWN_LABEL)
        provider = self.own.providers.get(choice)
        if provider is None:  # the shared model, or a provider no longer offered
            return self.shared_model()
        saved = await asyncio.to_thread(store.key, who.account, provider.id)
        if saved is None or saved.state == keystore.UNREADABLE:
            raise own_failure(p.KEY_MISSING, provider.name)
        if saved.state == keystore.REFUSED:
            raise own_failure(p.KEY_REFUSED, provider.name)
        if saved.model not in provider.models:
            raise own_failure(p.MODEL_UNAVAILABLE, provider.name, saved.model)
        try:
            endpoint = Endpoint(
                provider.base_url,
                saved.model,
                saved.key,
                kind="openai",
                headers=provider.request_headers(saved.workspace_id),
                body=provider.body,
            )
        except ValueError:
            # Its form was checked when it was saved, so this is a row gone wrong.
            raise own_failure(p.KEY_MISSING, provider.name) from None
        return Model(
            endpoints=(endpoint,),
            vision=True,
            max_tokens=self.own.max_tokens,
            context_budget=self.config.context_budget,
            max_images=self.own.max_images,
            provider=provider.id,
            name=provider.name,
            label=f"your {provider.name} key {saved.hint} ({saved.model})",
        )

    async def answer(self, messages, context, token, emit, model: Model | None = None) -> int:
        """Run the tool loop for the last question in messages, emitting events.

        Returns the number of tool calls it made.
        """
        model = model or self.shared_model()
        if model.label:
            emit = announcing(emit, f"Answering with {model.label}")
        tools = await self.tools()
        offered = {t["function"]["name"] for t in tools}
        cfg = self.config
        convo = [{"role": "system", "content": system_prompt(context, model.vision)}, *messages[-cfg.max_messages :]]
        # The question has its place on the model by now, so time spent in line
        # is not taken from its answer.
        start = time.monotonic()
        deadline = start + cfg.answer_timeout
        tools_until = start + cfg.answer_timeout * TOOL_SHARE
        # Tokens a person's own key spent on this question, as the provider counts them.
        spent: dict[str, int] = {}

        def finish(text, note="", empty=NO_ANSWER):
            text = strip_tool_markup(text)
            if not text:
                text = empty
            elif note:
                # Say so, or a sentence cut off mid-word reads like the whole answer.
                emit({"type": "text", "delta": note})
                text += note
            done = {"type": "done", "content": text}
            if spent:
                done["usage"] = {name: spent.get(name, 0) for name in ("input_tokens", "output_tokens")}
            emit(done)

        def ending(message):
            add_usage(spent, message.pop("usage", None))
            return TOO_LONG if message.pop("finish_reason", None) == "length" else ""

        reset = caller.set(Babamul(self.http, token, self.babamul_url))
        try:
            # One round can ask for several tools at once, so count the calls
            # themselves. Counting rounds lets a paging model run on and on.
            calls_made = 0
            while calls_made < cfg.max_tool_calls and time.monotonic() < tools_until:
                try:
                    message = await self.round(model, convo, tools, tools_until, emit, talking_until=deadline)
                except OutOfTime as exc:
                    if strip_tool_markup(exc.said):
                        # It was answering, and ran on to the deadline: keep what it said.
                        finish(exc.said, CUT_SHORT)
                        return calls_made
                    break
                note = ending(message)
                calls = message.get("tool_calls") or []
                if not calls:
                    finish(message.get("content"), note)
                    return calls_made
                for call in calls:
                    call["id"] = call["id"] or f"call_{uuid.uuid4().hex[:12]}"
                convo.append(message)
                seen = []
                for call in calls:
                    # Every call needs an answer, or the server refuses the next round.
                    if calls_made >= cfg.max_tool_calls:
                        content = "Not run: this question reached its limit of tool calls."
                    elif time.monotonic() >= tools_until:
                        content = "Not run: this question ran out of time for tools."
                    else:
                        calls_made += 1
                        content, images = await self.call(call, offered, tools_until, emit, model)
                        seen += images
                    convo.append({"role": "tool", "tool_call_id": call["id"], "content": content})
                if seen and model.vision:
                    convo.append(image_message(seen))

            # The cap or the clock stopped the tools: ask once more without them,
            # or the model asks for another instead of answering. A provider keeps
            # them with tool_choice "none" (see chat_payload); llama.cpp, as ever, is not offered them.
            convo.append({"role": "user", "content": ANSWER_NOW})
            try:
                if model.provider:
                    message = await self.round(model, convo, tools, deadline, emit, tool_choice="none")
                else:
                    message = await self.round(model, convo, [], deadline, emit)
                finish(message.get("content"), ending(message))
            except OutOfTime as exc:
                finish(exc.said, CUT_SHORT, empty=OUT_OF_TIME)
            return calls_made
        finally:
            caller.reset(reset)

    async def call(self, call, offered, until, emit, model: Model) -> tuple[str, list[dict]]:
        """Run one tool call; the text for the model and the images it returned."""
        name, call_id = call["function"]["name"], call["id"]
        try:
            arguments = json.loads(call["function"]["arguments"] or "{}")
            if not isinstance(arguments, dict):
                raise ValueError
        except ValueError:
            arguments = None
        emit({"type": "tool_call", "id": call_id, "name": name, "arguments": arguments or {}})

        texts, images, ok = [], [], False
        try:
            if arguments is None:
                raise ToolError("its arguments were not a JSON object")
            if name not in offered:
                raise ToolError(f"there is no tool called {name}")
            context = LocalContext(mcp_server=mcp, subscriptions=getattr(mcp, "_subscriptions", None))
            async with asyncio.timeout(until - time.monotonic()):
                result = await mcp.call_tool(name, arguments, context)
            texts, images = split_content(result)
            ok = not result.is_error
        except TimeoutError:
            texts = [f"tool {name} ran out of time and was stopped"]
        except Exception as exc:
            texts = [self.failure_text(name, exc)]
        text = "\n".join(texts)
        emit(
            {
                "type": "tool_result",
                "id": call_id,
                "name": name,
                "ok": ok,
                "summary": condense(text, SUMMARY_BUDGET),
                "images": images,
            }
        )
        content = condense_blocks(texts, self.config.result_budget)
        if images:
            # First, so that cutting an old result down to fit keeps it.
            n = f"{len(images)} image{'s' * (len(images) > 1)}"
            note = (
                f"[{n} shown to the person in the chat, and attached below for you.]"
                if model.vision
                else f"[{n} shown to the person in the chat. You cannot see {'them' if len(images) > 1 else 'it'}.]"
            )
            content = f"{note}\n{content}"
        return content, images

    @staticmethod
    def failure_text(name: str, exc: Exception) -> str:
        """What the model reads about a tool that failed, and the log line for a crash.

        The SDK wraps a crash so that only "Error executing tool X" is left, and
        the assistant calls tools without the MCP handler that would log it.
        """
        cause = crash_cause(exc)
        if isinstance(cause, ToolError):
            # Anticipated, such as Babamul answering 503, maybe from inside a task group.
            log.info("tool %s failed: %s", name, cause)
            message = str(cause)
            return message if message.startswith(f"Error executing tool {name}") else f"tool {name} failed: {message}"
        log.error("tool %s crashed", name, exc_info=exc)
        detail = f"{type(cause).__name__}: {cause}"[:200]
        return f"tool {name} crashed ({detail}). Calling it again with the same arguments will fail the same way."

    def endpoints(self, model: Model | None = None) -> list[Endpoint]:
        """Where to send one round, in the order to try them."""
        if model is not None and model.endpoints:
            # Each is given a second chance, as the operator's model is without a fallback.
            return [e for endpoint in model.endpoints for e in (endpoint, endpoint)]
        main, fallback = self.config.llm, self.config.fallback
        if fallback is None:
            return [main, main]
        if time.monotonic() < self._main_skipped_until:
            return [fallback, main]
        return [main, fallback]

    async def round(self, model: Model, messages, tools, until, emit, talking_until=None, tool_choice=None) -> dict:
        """One model round, moved to the other endpoint if the first fails before saying anything.

        Raises OutOfTime when the clock reaches until, and ModelError when no
        endpoint answers. A round that has started to answer in text may run on
        to talking_until: cutting it at until would throw away an answer the
        person is watching arrive, and ask for it again with less time.
        """
        fit(messages, model.context_budget, model.max_images)
        failure: ModelError | None = None
        tried, again = None, True
        for endpoint in self.endpoints(model):
            if endpoint is tried and not again:
                break
            if (left := until - time.monotonic()) <= 0:
                break
            tried, said = endpoint, []
            try:
                async with asyncio.timeout(left) as timer:
                    extend = None
                    if talking_until is not None and talking_until > until:
                        later = timer.when() + talking_until - until

                        def extend():
                            timer.reschedule(later)

                    return await self.stream(model, endpoint, messages, tools, emit, said, extend, tool_choice)
            except TimeoutError:
                raise OutOfTime("".join(said)) from None
            except Stalled as exc:
                if said:
                    raise ModelError(STALLED, detail=str(exc)) from None
                # Not asked again where it went quiet: that request may still be
                # running there, and a second one would wait behind it.
                failure, again = ModelError(STALLED, switchable=True, detail=str(exc)), False
            except ModelError as exc:
                if not exc.switchable or said:
                    raise
                failure = exc
            except httpx.TransportError as exc:
                # Unreachable, or the connection broke. Only the type is kept, and the
                # error is not chained: h11's message for a header it refuses quotes
                # the whole header, so it can hold the key.
                if said:
                    raise ModelError(STALLED, detail=type(exc).__name__) from None
                failure = ModelError(UNREACHABLE, switchable=True, detail=type(exc).__name__)
            log.warning("model at %s failed: %s", shown_url(endpoint.base_url), failure)
            if not model.endpoints and endpoint is self.config.llm and self.config.fallback is not None:
                # Only the operator's own failures move everyone to the fallback.
                self._main_skipped_until = time.monotonic() + self.config.fallback_period
        if failure is not None:
            raise failure
        raise OutOfTime()

    async def stream(
        self, model: Model, endpoint: Endpoint, messages, tools, emit, said: list[str], on_text=None, tool_choice=None
    ) -> dict:
        """One streamed chat completion, its text emitted as it arrives.

        on_text is called when the first text arrives. The message returned also
        has the reply's finish_reason, and on a person's own key the tokens it
        used, which the caller takes out.
        """
        cfg = self.config
        payload = chat_payload(
            endpoint.model, messages, tools, model.thinking, model.max_tokens, endpoint.kind, endpoint.body, tool_choice
        )
        payload["stream"] = True
        headers = {"Accept": "text/event-stream", **dict(endpoint.headers)}
        if endpoint.api_key:
            # Checked again here, so a key that cannot be sent never reaches h11.
            if not API_KEY.fullmatch(endpoint.api_key):
                raise ModelError(MISCONFIGURED, detail="the API key cannot go in a header")
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        stall = cfg.stall_timeout
        async with self.http.stream(
            "POST",
            f"{endpoint.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            # No read timeout here. Before its first byte the request waits in the
            # model server's own queue and then for its prompt to be read, which
            # only the round's clock limits. Silence after that is a stall, below.
            timeout=httpx.Timeout(None, connect=min(10.0, stall), write=stall),
        ) as response:
            if response.status_code >= 400:
                # The status alone says nothing: a refused request looks the same
                # whether the context was too long or a tool schema was rejected.
                # The error's type and code tell them apart; the rest of the body
                # is not logged, since a provider may quote the key it refused.
                body = await response.aread()
                why = error_fields(body)
                log.warning(
                    "model at %s refused the request: %s %s (messages=%d, tools=%d, %d chars)",
                    shown_url(endpoint.base_url),
                    response.status_code,
                    why,
                    len(messages),
                    len(tools),
                    len(json.dumps(messages)) + len(json.dumps(tools)),
                )
                if model.provider:
                    raise own_refused(model, body, response.status_code, f"{response.status_code} {why}", bool(said))
                raise refused(response.status_code, f"{response.status_code} {why}")
            kind = response.headers.get("content-type", "")
            if not kind.startswith("text/event-stream"):
                # A proxy's login or error page, sent with a 200. Its body is not
                # logged: it is not ours to keep, and the type says enough.
                log.warning("model at %s answered %r, not a stream", shown_url(endpoint.base_url), kind[:100])
                # A person's provider is asked again only when it failed outright.
                raise ModelError(UNREACHABLE, switchable=not model.provider, detail=f"content-type {kind[:100]!r}")
            emit({"type": "_started"})
            deltas, finish, ended, started, usage = [], None, False, False, None
            async with aclosing(response.aiter_lines()) as lines:
                while True:
                    try:
                        if started:
                            async with asyncio.timeout(stall):
                                line = await anext(lines)
                        else:
                            line = await anext(lines)
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        raise Stalled(f"silent for {stall:g} s") from None
                    started = True
                    if line.startswith("error:"):
                        # How llama.cpp reports a failure partway through a reply.
                        raise ModelError(STALLED, switchable=not said, detail=error_fields(line[6:].strip()))
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        ended = True
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if chunk.get("error"):
                        # A provider that has sent its 200 reports any failure here,
                        # out of credit or rate limited included.
                        status = p.status_of(p.error_of(chunk))
                        if model.provider and (p.classify(model.provider, chunk) or (status or 0) // 100 == 4):
                            raise own_refused(model, chunk, None, error_fields(chunk), bool(said))
                        again = not said and not model.provider
                        raise ModelError(STALLED, switchable=again, detail=error_fields(chunk))
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]  # in the last chunk, for the whole reply
                    for choice in (chunk.get("choices") or [])[:1]:
                        finish = choice.get("finish_reason") or finish
                        delta = choice.get("delta") or {}
                        deltas.append(delta)
                        if text := delta.get("content"):
                            if not said and on_text is not None:
                                on_text()
                            said.append(text)
                            emit({"type": "text", "delta": text})
        if not ended and finish is None:
            # A reply that just stops was cut off on the way, whatever its status said.
            log.warning(
                "model at %s stopped its reply without finishing it (%d chunks)",
                shown_url(endpoint.base_url),
                len(deltas),
            )
            raise ModelError(STALLED, switchable=not said, detail="the reply ended early")
        if not deltas:
            log.warning("model at %s sent an empty reply", shown_url(endpoint.base_url))
        message = merge_deltas(deltas)
        message["finish_reason"] = finish
        if model.provider:
            message["usage"] = usage
        return message


def bearer(request: Request) -> str:
    """The Babamul token a request carries, or BadRequest (401) without asking Babamul,
    when there is none or Babamul would refuse it."""
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token:
        raise BadRequest("Log in to Babamul to use the assistant.", 401)
    if not TOKEN.fullmatch(token):
        raise BadRequest("Babamul did not accept your login. Log in again.", 401)
    return token


async def read_body(request: Request, limit: int | None = None, too_long: str = "") -> bytes:
    """The request body, refused as soon as it is too long rather than once it is all in memory."""
    limit = MAX_BODY if limit is None else limit
    too_long = BadRequest(too_long or "The conversation is too long. Start a new chat.", 413)
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise too_long
    body = bytearray()
    async for part in request.stream():
        body += part
        if len(body) > limit:
            raise too_long
    return bytes(body)


async def key_body(request: Request) -> dict:
    """The JSON object a key route was sent. Errors never quote it: it may hold a key."""
    raw = await read_body(request, MAX_KEY_BODY, "That request is too long for a key.")
    try:
        body = json.loads(raw)
    except (ValueError, RecursionError):
        body = None
    if not isinstance(body, dict):
        raise BadRequest("The request body must be a JSON object.", 400, "bad_format")
    return body


def parse_body(raw: bytes) -> tuple[list[dict], dict | None, str | None]:
    """The conversation, the page's context, and "shared" when the person asks the
    shared model this once instead of their own key."""
    if len(raw) > MAX_BODY:
        raise BadRequest("The conversation is too long. Start a new chat.", 413)
    try:
        body = json.loads(raw)
    except (ValueError, RecursionError):
        # Deep nesting runs out of recursion before it is found not to be a conversation.
        raise BadRequest("The request body is not JSON.") from None
    if not isinstance(body, dict):
        raise BadRequest("The request body must be a JSON object.")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a list with at least one message.")
    turns = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ("user", "assistant")
            or not isinstance(message.get("content"), str)
        ):
            raise BadRequest('Each message needs a role, "user" or "assistant", and its text as content.')
        turns.append({"role": message["role"], "content": message["content"][:MAX_MESSAGE_CHARS]})
    if turns[-1]["role"] != "user" or not turns[-1]["content"].strip():
        raise BadRequest("The last message must be the user's question.")
    context = body.get("context")
    if context is not None and not isinstance(context, dict):
        raise BadRequest("context must be an object.")
    use = body.get("use")
    if use is not None and use != SHARED:
        raise BadRequest('use may only be "shared", to ask the shared model this once.')
    return turns, context, use


def store_down(exc: sqlite3.Error) -> JSONResponse:
    log.error("the key store failed: %s", type(exc).__name__)
    return error(503, "Saved keys cannot be reached right now. Try again in a minute.", code="store_unavailable")


def error(status: int, message: str, headers: dict | None = None, code: str = "") -> JSONResponse:
    body = {"error": message}
    if code:
        body["code"] = code
    return JSONResponse(body, status_code=status, headers=headers)


def mount(app, assistant: Assistant, prefix: str = "/assistant"):
    """Add the assistant's routes under prefix to a Starlette app, and run it with the app."""
    prefix = "/" + prefix.strip("/") if prefix.strip("/") else ""
    app.router.routes[0:0] = [
        Route(f"{prefix}/health", assistant.health, methods=["GET"]),
        Route(f"{prefix}/chat", assistant.chat, methods=["POST"]),
        # GET, POST and DELETE only: the CORS settings in server.py allow no others.
        Route(f"{prefix}/keys", assistant.list_keys, methods=["GET"]),
        Route(f"{prefix}/keys", assistant.delete_keys, methods=["DELETE"]),
        Route(f"{prefix}/keys/{{provider}}", assistant.save_key, methods=["POST"]),
        Route(f"{prefix}/keys/{{provider}}", assistant.delete_key, methods=["DELETE"]),
        Route(f"{prefix}/choice", assistant.choose, methods=["POST"]),
    ]
    inner = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with inner(app) as state, assistant.running():
            yield state

    app.router.lifespan_context = lifespan
    return app
