"""The model providers people may bring their own key for, and what each needs.

Only fixed https hosts are ever contacted with a person's key: no route takes
a URL. Each provider has its own key format, key check, request extras and
error shapes. Anthropic is reached through its OpenAI-compatible endpoint for
now; a native Messages adapter would add prompt caching and thinking.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("boom_mcp.assistant")

ANTHROPIC, OPENROUTER = "anthropic", "openrouter"
# Ids kept for providers that need an adapter first (OpenAI's Responses API,
# Gemini's thought signatures), so that turning one on later changes no client.
RESERVED = ("openai", "gemini")
DEFAULT_PROVIDERS = (ANTHROPIC, OPENROUTER)

# A pasted key: the characters keys are made of, and nothing an HTTP header or a log
# line could trip over. Stricter than the operator's keys, which may come from anywhere.
USER_KEY = re.compile(r"[A-Za-z0-9._~+/=-]{16,512}")
MODEL_ID = re.compile(r"[A-Za-z0-9._:/~-]{1,100}")
WORKSPACE_ID = re.compile(r"[A-Za-z0-9_-]{1,100}")
CHECK_TIMEOUT = 10.0
ANTHROPIC_VERSION = "2023-06-01"

# What went wrong with a person's own key, as the panel is told.
KEY_MISSING = "own_key_missing"
KEY_REFUSED = "own_key_refused"
NO_CREDIT = "own_key_no_credit"
BUSY = "own_key_busy"
MODEL_UNAVAILABLE = "own_model_unavailable"
DOWN = "own_provider_down"

# Anthropic names its errors by type rather than by a code in the body. A
# failure partway through a stream has no status of its own, so it comes from here.
ANTHROPIC_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "overloaded_error": 529,
}


class BadKey(ValueError):
    """A key that is not of the right form. The message is for the person and never quotes it."""


class KeyRejected(Exception):
    """The provider refused the key."""


class ProviderUnreachable(Exception):
    """The provider could not say whether the key works."""


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    # The OpenAI-compatible base URL, ending in /v1.
    base_url: str
    key_prefix: str
    # Where a person makes a key.
    key_url: str
    # Whether a key may need a workspace id sent with it.
    workspace_field: bool
    models: tuple[str, ...]
    headers: tuple[tuple[str, str], ...] = ()
    # Fields added to every chat request.
    body: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def default_model(self) -> str:
        return self.models[0]

    def public(self) -> dict:
        """What the web app needs to offer this provider."""
        return {
            "id": self.id,
            "name": self.name,
            "models": list(self.models),
            "default_model": self.default_model,
            "key_url": self.key_url,
            "key_prefix": self.key_prefix,
            "workspace_field": self.workspace_field,
        }

    def request_headers(self, workspace_id: str | None) -> tuple[tuple[str, str], ...]:
        if workspace_id and self.workspace_field:
            return (*self.headers, ("anthropic-workspace-id", workspace_id))
        return self.headers

    def check_format(self, raw) -> str:
        """The key, stripped, or BadKey saying what is wrong with it."""
        key = raw.strip() if isinstance(raw, str) else ""
        if not key:
            raise BadKey(f"Paste your {self.name} API key.")
        if not USER_KEY.fullmatch(key):
            raise BadKey(
                f"That does not look like an API key: it should be 16 to 512 letters, digits and dashes, "
                f"starting with {self.key_prefix}, with no spaces or line breaks."
            )
        if self.id == ANTHROPIC:
            if key.startswith("sk-ant-admin"):
                raise BadKey(
                    "That is an Anthropic Admin key, which cannot ask models. "
                    "Make an API key in the Claude Console instead."
                )
            if key.startswith("sk-ant-") and not key.startswith("sk-ant-api"):
                raise BadKey(
                    "This is not a Console API key. A Claude subscription cannot be used here: "
                    "make an API key in the Claude Console."
                )
        if not key.startswith(self.key_prefix):
            raise BadKey(f"{self.name} API keys start with {self.key_prefix}.")
        return key

    async def check_key(self, http: httpx.AsyncClient, key: str, model: str, workspace_id: str | None = None):
        """Ask the provider whether key works, with a call that runs no model.

        Anthropic's token count is free and also checks that the key may use
        the model; OpenRouter's /key reads the key's own limits. Raises
        KeyRejected or ProviderUnreachable. Neither the key nor the provider's
        answer is logged or kept: only the status, or the error's type.
        """
        try:
            if self.id == ANTHROPIC:
                headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
                headers.update(self.request_headers(workspace_id))
                response = await http.post(
                    f"{self.base_url}/messages/count_tokens",
                    headers=headers,
                    json={"model": model, "messages": [{"role": "user", "content": "Hi"}]},
                    timeout=CHECK_TIMEOUT,
                )
            else:
                response = await http.get(
                    f"{self.base_url}/key", headers={"Authorization": f"Bearer {key}"}, timeout=CHECK_TIMEOUT
                )
        except Exception as exc:
            # Not only httpx's errors: whatever fails here may quote the request, so
            # only its type goes anywhere, and it is not chained.
            log.warning("could not check a key with %s: %s", self.name, type(exc).__name__)
            raise ProviderUnreachable(f"Could not reach {self.name} to check the key. Try again in a minute.") from None
        status = response.status_code
        if status < 300:
            return
        log.info("%s refused a key check: %s", self.name, status)
        if status == 429 or status >= 500:
            raise ProviderUnreachable(f"{self.name} is busy and could not check the key. Try again in a minute.")
        if status in (401, 403):
            raise KeyRejected(f"{self.name} did not accept this key.")
        if status == 404:
            raise KeyRejected(f"{self.name} says this key cannot use {model}. Pick another model.")
        hint = " If the key is not tied to one workspace, add its workspace id." if self.workspace_field else ""
        raise KeyRejected(f"{self.name} did not accept this key for {model}.{hint}")


PRESETS = {
    ANTHROPIC: dict(
        name="Anthropic",
        base_url="https://api.anthropic.com/v1",
        key_prefix="sk-ant-api",
        key_url="https://platform.claude.com/settings/keys",
        workspace_field=True,
        models=("claude-haiku-5-5", "claude-sonnet-5-5"),
    ),
    OPENROUTER: dict(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        key_prefix="sk-or-",
        key_url="https://openrouter.ai/settings/keys",
        workspace_field=False,
        # The same two Claude models, as OpenRouter names them. Gemini and GPT models
        # wait for a live test of several tool rounds through OpenRouter.
        models=("anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"),
        headers=(("X-OpenRouter-Title", "Babamul"),),
        # Only providers that do not keep or train on what they are sent.
        body={"provider": {"data_collection": "deny"}},
    ),
}


def from_env(env=None) -> dict[str, Provider]:
    """The providers ASSISTANT_BYOK_PROVIDERS turns on, in its order, with their
    model lists from ASSISTANT_BYOK_<PROVIDER>_MODELS (the first is the default)."""
    env = os.environ if env is None else env
    names = env.get("ASSISTANT_BYOK_PROVIDERS", "").strip() or ",".join(DEFAULT_PROVIDERS)
    providers = {}
    for name in (n.strip().lower() for n in names.split(",")):
        if not name or name in providers:
            continue
        if name in RESERVED:
            raise ValueError(f"ASSISTANT_BYOK_PROVIDERS: {name} is not offered yet")
        if name not in PRESETS:
            raise ValueError(f"ASSISTANT_BYOK_PROVIDERS: {name!r} is not a provider; use {', '.join(PRESETS)}")
        preset = dict(PRESETS[name])
        variable = f"ASSISTANT_BYOK_{name.upper()}_MODELS"
        if raw := env.get(variable, "").strip():
            models = tuple(dict.fromkeys(m.strip() for m in raw.split(",") if m.strip()))
            if not models or not all(MODEL_ID.fullmatch(m) for m in models):
                raise ValueError(f"{variable} must be model ids separated by commas")
            preset["models"] = models
        providers[name] = Provider(id=name, **preset)
    if not providers:
        raise ValueError("ASSISTANT_BYOK_PROVIDERS names no provider")
    return providers


def error_of(body) -> dict:
    """The error object of a provider's error body, parsed if need be; {} if there is none."""
    if isinstance(body, (bytes, str)):
        try:
            body = json.loads(body)
        except (ValueError, RecursionError):
            return {}
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body["error"]
    return {}


def status_of(error: dict, status: int | None = None) -> int | None:
    """The HTTP status an error stands for: the response's, or for one partway
    through a stream, the code or type inside it."""
    if status is not None:
        return status
    code = error.get("code")
    if isinstance(code, int) and not isinstance(code, bool) and 100 <= code < 600:
        return code
    return ANTHROPIC_STATUS.get(error.get("type"))


def classify(provider: str, body, status: int | None = None) -> str | None:
    """What a provider's error means for the key's owner, as one of the own_* codes,
    or None for one the usual messages cover (a conversation it will not take).

    From the status and the error's type, code and metadata, and a few phrases
    of the message where the providers say nothing else: Anthropic's
    "credit balance is too low" and spend limits come as plain 400s. The
    message is only matched, never kept.
    """
    error = error_of(body)
    status = status_of(error, status)
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    labels = " ".join(
        str(value).lower()
        for value in (error.get("type"), error.get("code"), metadata.get("error_type"))
        if isinstance(value, str)
    )
    message = error.get("message").lower() if isinstance(error.get("message"), str) else ""
    limit = metadata.get("limit_source")

    if limit == "openrouter_in_flight_budget":
        return BUSY  # its running requests fill the budget for now; it frees up by itself
    if limit or any(word in labels for word in ("insufficient_quota", "spend_limit", "billing", "payment_required")):
        return NO_CREDIT
    if status == 402 or any(phrase in message for phrase in ("credit balance", "usage limit", "spend limit")):
        return NO_CREDIT
    if status == 403 and "key limit exceeded" in message:
        return NO_CREDIT
    if status == 401 or "authentication" in labels:
        return KEY_REFUSED
    if status == 403:
        # OpenRouter's 403 is moderation or a guardrail, about the question and not the key.
        return KEY_REFUSED if provider == ANTHROPIC else None
    if status == 404:
        return MODEL_UNAVAILABLE
    if status == 429:
        return BUSY
    if status is not None and (status == 408 or status >= 500):
        return DOWN
    return None
