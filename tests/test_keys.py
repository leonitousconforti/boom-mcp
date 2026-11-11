"""The routes where people save, choose and remove their own model keys."""

import asyncio
import base64
import logging
import sqlite3
from datetime import datetime

import httpx
import pytest
from starlette.applications import Starlette

from boom_mcp import assistant as a
from boom_mcp import providers
from boom_mcp.babamul import base_url_from_env
from boom_mcp.keystore import KeyStore
from tests.conftest import FAKE_ANTHROPIC_KEY, FAKE_OPENROUTER_KEY, KEYS_SECRET, LOGIN, LOGIN_NO_ID

pytestmark = pytest.mark.anyio

ANTHROPIC_PUBLIC = {
    "id": "anthropic",
    "name": "Anthropic",
    "models": ["claude-haiku-5-5", "claude-sonnet-5-5"],
    "default_model": "claude-haiku-5-5",
    "key_url": "https://platform.claude.com/settings/keys",
    "key_prefix": "sk-ant-api",
    "workspace_field": True,
}
OPENROUTER_PUBLIC = {
    "id": "openrouter",
    "name": "OpenRouter",
    "models": ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"],
    "default_model": "anthropic/claude-haiku-5.5",
    "key_url": "https://openrouter.ai/settings/keys",
    "key_prefix": "sk-or-",
    "workspace_field": False,
}


async def send(app, method, path, body=None, token=LOGIN, content=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://boom") as client:
        return await client.request(method, f"/assistant{path}", json=body, content=content, headers=headers)


def profiles(t) -> int:
    return len([r for r in t.babamul.requests if r.url.path.endswith("/profile")])


def days_between(entry) -> float:
    start, end = (datetime.fromisoformat(entry[f].replace("Z", "+00:00")) for f in ("created_at", "expires_at"))
    return (end - start).total_seconds() / 86400


async def test_health_tells_the_web_app_what_it_may_offer(byok):
    response = await send(byok().app, "GET", "/health", token=None)
    assert response.json()["own_keys"] == {
        "providers": [ANTHROPIC_PUBLIC, OPENROUTER_PUBLIC],
        "days": [1, 7, 30, 90],
        "default_days": 30,
    }
    t = byok(max_days=7, env={"ASSISTANT_BYOK_PROVIDERS": "openrouter"})
    own = (await send(t.app, "GET", "/health", token=None)).json()["own_keys"]
    assert own == {"providers": [OPENROUTER_PUBLIC], "days": [1, 7], "default_days": 7}


async def test_with_no_store_the_routes_are_not_found(fake):
    bot = a.Assistant(a.Config(llm=a.Endpoint("http://llm.test/v1", "qwen")), http=httpx.AsyncClient())
    app = a.mount(Starlette(), bot)
    assert (await send(app, "GET", "/health", token=None)).json()["own_keys"] is None
    for method, path in (("GET", "/keys"), ("DELETE", "/keys"), ("POST", "/keys/anthropic"),
                         ("DELETE", "/keys/anthropic"), ("POST", "/choice")):
        response = await send(app, method, path, {"key": FAKE_ANTHROPIC_KEY, "use": "anthropic"})
        assert response.status_code == 404 and response.json()["code"] == "not_found", path
    assert fake.requests == []


async def test_save_list_choose_and_remove_keys(byok, caplog):
    t = byok()
    with caplog.at_level(logging.DEBUG):
        saved = await send(t.app, "POST", "/keys/anthropic", {
            "key": f"  {FAKE_ANTHROPIC_KEY}\n", "model": "claude-sonnet-5-5", "days": 7, "workspace_id": " wrkspc_01FAKE ",
        })
        assert saved.status_code == 200, saved.text
        entry = saved.json()
        assert entry == {
            "provider": "anthropic", "hint": "…ab12", "model": "claude-sonnet-5-5", "workspace_id": "wrkspc_01FAKE",
            "created_at": entry["created_at"], "expires_at": entry["expires_at"], "last_used_at": None, "state": "ok",
        }
        assert entry["created_at"].endswith("Z") and days_between(entry) == 7
        check = t.checks.requests[-1]
        assert (check.method, str(check.url)) == ("POST", "https://api.anthropic.com/v1/messages/count_tokens")
        assert check.headers["x-api-key"] == FAKE_ANTHROPIC_KEY and check.headers["anthropic-workspace-id"] == "wrkspc_01FAKE"
        assert check.headers["anthropic-version"] and b'"model":"claude-sonnet-5-5"' in check.content.replace(b" ", b"")
        assert (await send(t.app, "GET", "/keys")).json() == {"use": "anthropic", "keys": [entry]}

        # A second provider, kept without becoming the choice, with the default model and days.
        other = await send(t.app, "POST", "/keys/openrouter", {"key": FAKE_OPENROUTER_KEY, "use": False})
        assert other.status_code == 200 and other.json()["model"] == "anthropic/claude-haiku-5.5"
        assert days_between(other.json()) == 30 and other.json()["workspace_id"] is None
        check = t.checks.requests[-1]
        assert (check.method, str(check.url)) == ("GET", "https://openrouter.ai/api/v1/key")
        assert check.headers["authorization"] == f"Bearer {FAKE_OPENROUTER_KEY}"
        listed = (await send(t.app, "GET", "/keys")).json()
        assert listed["use"] == "anthropic" and [k["provider"] for k in listed["keys"]] == ["anthropic", "openrouter"]

        chose = await send(t.app, "POST", "/choice", {"use": "openrouter", "model": "anthropic/claude-sonnet-5.5"})
        assert chose.status_code == 200 and chose.json() == {"use": "openrouter"}
        listed = (await send(t.app, "GET", "/keys")).json()
        assert listed["use"] == "openrouter" and listed["keys"][1]["model"] == "anthropic/claude-sonnet-5.5"
        assert (await send(t.app, "POST", "/choice", {"use": "shared"})).json() == {"use": "shared"}
        assert (await send(t.app, "GET", "/keys")).json()["use"] == "shared"
        assert (await send(t.app, "POST", "/choice", {"use": "openrouter"})).json() == {"use": "openrouter"}

        # Removing the key in use goes back to the shared model.
        removed = await send(t.app, "DELETE", "/keys/openrouter")
        assert removed.status_code == 204 and removed.content == b""
        listed = (await send(t.app, "GET", "/keys")).json()
        assert listed["use"] == "shared" and [k["provider"] for k in listed["keys"]] == ["anthropic"]
        assert (await send(t.app, "DELETE", "/keys/openrouter")).status_code == 204  # gone already: still fine
        assert (await send(t.app, "DELETE", "/keys")).status_code == 204
        assert (await send(t.app, "GET", "/keys")).json() == {"use": "shared", "keys": []}

    assert "user user-1 saved their Anthropic key for 7 days" in caplog.text
    assert "FAKE" not in caplog.text and "\u2026ab12" not in caplog.text


async def test_a_change_asks_babamul_afresh_and_a_listing_does_not(byok):
    t = byok()
    await send(t.app, "GET", "/keys")
    before = profiles(t)
    await send(t.app, "GET", "/keys")
    assert profiles(t) == before  # kept for a while, as for questions
    await send(t.app, "POST", "/choice", {"use": "shared"})
    await send(t.app, "DELETE", "/keys")
    await send(t.app, "POST", "/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY})
    assert profiles(t) == before + 3


async def test_api_tokens_and_unknown_logins_are_refused(byok):
    t = byok()
    routes = [("GET", "/keys"), ("DELETE", "/keys"), ("POST", "/keys/anthropic"), ("DELETE", "/keys/anthropic"),
              ("POST", "/choice")]
    body = {"key": FAKE_ANTHROPIC_KEY, "use": "anthropic"}
    for method, path in routes:
        response = await send(t.app, method, path, body, token="bbml_test")
        assert response.status_code == 403 and response.json()["code"] == "login_required", path
        assert "web app" in response.json()["error"]
    assert t.babamul.requests == [] and t.checks.requests == []  # not even asked who it is

    for token, status, code in ((None, 401, None), ("eyJhbGciOiJIUzI1NiJ9.eyJ9.bm9wZQ", 401, None),
                                ("not a token", 401, None), (LOGIN_NO_ID, 403, "no_account")):
        for method, path in routes:
            response = await send(t.app, method, path, body, token=token)
            assert response.status_code == status and response.json().get("code") == code, (token, path)
    assert t.checks.requests == [] and (await asyncio.to_thread(t.store.entries, "user-1")) == []


@pytest.mark.parametrize(
    "path, body, code, says",
    [
        ("/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY + "\nX-Injected: yes"}, "bad_format", "no spaces or line breaks"),
        ("/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY[:-4] + "café"}, "bad_format", "no spaces or line breaks"),
        ("/keys/anthropic", {"key": "sk-ant-api03-x"}, "bad_format", "16 to 512"),
        ("/keys/anthropic", {"key": "x" * 513}, "bad_format", "16 to 512"),
        ("/keys/anthropic", {"key": "sk-ant-admin01-FAKE" + "0" * 60}, "bad_format", "Admin key"),
        ("/keys/anthropic", {"key": "sk-ant-oat01-FAKE" + "0" * 60}, "bad_format", "not a Console API key"),
        ("/keys/anthropic", {"key": FAKE_OPENROUTER_KEY}, "bad_format", "start with sk-ant-api"),
        ("/keys/openrouter", {"key": FAKE_ANTHROPIC_KEY}, "bad_format", "start with sk-or-"),
        ("/keys/anthropic", {"key": ""}, "bad_format", "Paste"),
        ("/keys/anthropic", {}, "bad_format", "Paste"),
        ("/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY, "model": "claude-opus-5-5"}, "bad_model", "claude-haiku-5-5"),
        ("/keys/openrouter", {"key": FAKE_OPENROUTER_KEY, "model": "claude-haiku-5-5"}, "bad_model", "OpenRouter"),
        ("/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY, "days": 365}, "bad_days", "1, 7, 30, 90"),
        ("/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY, "days": "30"}, "bad_days", "days"),
        ("/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY, "days": True}, "bad_days", "days"),
        ("/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY, "workspace_id": "wrkspc_01\nX: y"}, "bad_format", "workspace"),
        ("/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY, "workspace_id": 7}, "bad_format", "workspace"),
        ("/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY, "use": "yes"}, "bad_format", "use"),
        ("/keys/anthropic", [FAKE_ANTHROPIC_KEY], "bad_format", "JSON object"),
        ("/choice", {"use": "anthropic", "model": "gpt-6"}, "bad_model", "Anthropic"),
        ("/choice", {}, "bad_format", "use"),
    ],
)
async def test_bad_requests_are_refused_without_asking_the_provider(byok, caplog, path, body, code, says):
    t = byok()
    with caplog.at_level(logging.DEBUG):
        response = await send(t.app, "POST", path, body)
    assert response.status_code == 400 and response.json()["code"] == code
    assert says in response.json()["error"]
    assert "FAKE" not in response.text and "FAKE" not in caplog.text
    assert t.checks.requests == [] and not t.bot.saves.hits.get("user-1")


async def test_bodies_that_are_not_json_or_too_long(byok):
    t = byok()
    for content in (b"not json", b"[" * 10_000, FAKE_ANTHROPIC_KEY.encode()):
        response = await send(t.app, "POST", "/keys/anthropic", content=content)
        assert response.status_code == 400 and response.json()["code"] == "bad_format"
        assert "FAKE" not in response.text
    response = await send(t.app, "POST", "/keys/anthropic", content=b'{"key": "' + b"x" * 20_000 + b'"}')
    assert response.status_code == 413
    assert t.checks.requests == []


async def test_providers_not_offered_are_not_found(byok):
    t = byok(env={"ASSISTANT_BYOK_PROVIDERS": "anthropic"})
    for name in ("openrouter", "openai", "gemini", "nope"):
        for method, path, body in (("POST", f"/keys/{name}", {"key": FAKE_OPENROUTER_KEY}), ("DELETE", f"/keys/{name}", None),
                                   ("POST", "/choice", {"use": name})):
            response = await send(t.app, method, path, body)
            assert response.status_code == 404 and response.json()["code"] == "not_found", (name, path)
    assert t.checks.requests == []


async def test_a_key_the_provider_refuses_is_not_kept(byok, caplog):
    t = byok()
    wrong = FAKE_ANTHROPIC_KEY[:-4] + "zz99"
    with caplog.at_level(logging.DEBUG):
        response = await send(t.app, "POST", "/keys/anthropic", {"key": wrong})
    assert response.status_code == 422 and response.json() == {"error": "Anthropic did not accept this key.", "code": "rejected"}
    assert t.checks.requests  # it was asked, and quoted the key back in its refusal
    assert "FAKE" not in caplog.text and "zz99" not in caplog.text

    for reply, status, code in ((404, 422, "rejected"), (400, 422, "rejected"), (403, 422, "rejected"),
                                (429, 502, "provider_unreachable"), (529, 502, "provider_unreachable"),
                                (httpx.ConnectTimeout(f"timed out: Bearer {FAKE_OPENROUTER_KEY}"), 502, "provider_unreachable"),
                                (httpx.LocalProtocolError(f"Illegal header value b'Bearer {FAKE_OPENROUTER_KEY}'"), 502,
                                 "provider_unreachable"),
                                (RuntimeError(FAKE_OPENROUTER_KEY), 502, "provider_unreachable")):
        t.checks.reply = reply
        with caplog.at_level(logging.DEBUG):
            response = await send(t.app, "POST", "/keys/openrouter", {"key": FAKE_OPENROUTER_KEY})
        assert (response.status_code, response.json()["code"]) == (status, code), reply
        assert "FAKE" not in response.text and "FAKE" not in caplog.text
    assert (await send(t.app, "GET", "/keys")).json() == {"use": "shared", "keys": []}
    assert "could not check a key with OpenRouter: RuntimeError" in caplog.text


async def test_saving_is_limited_per_person(byok):
    t = byok()
    t.babamul.tokens["eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyLTIifQ.dHdv"] = "user-2"
    t.checks.good.discard(FAKE_OPENROUTER_KEY)
    for i in range(a.SAVES_PER_HOUR):
        # A refused key counts too: this is how one would try keys found elsewhere.
        key, path = (FAKE_ANTHROPIC_KEY, "/keys/anthropic") if i % 2 else (FAKE_OPENROUTER_KEY, "/keys/openrouter")
        assert (await send(t.app, "POST", path, {"key": key})).status_code in (200, 422)
    asked = len(t.checks.requests)
    response = await send(t.app, "POST", "/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY})
    assert response.status_code == 429 and response.json()["code"] == "too_many_saves"
    assert 3500 < int(response.headers["retry-after"]) <= 3600 and len(t.checks.requests) == asked
    other = await send(t.app, "POST", "/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY},
                       token="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyLTIifQ.dHdv")
    assert other.status_code == 200  # another person's limit is their own


async def test_a_choice_needs_a_key_that_works(byok):
    t = byok()
    response = await send(t.app, "POST", "/choice", {"use": "anthropic"})
    assert response.status_code == 409 and response.json()["code"] == "no_key"
    await send(t.app, "POST", "/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY, "use": False})
    await asyncio.to_thread(t.store.mark, "user-1", "anthropic", "refused")
    assert (await send(t.app, "POST", "/choice", {"use": "anthropic"})).status_code == 409
    await send(t.app, "POST", "/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY, "use": False})  # saved again: fine
    assert (await send(t.app, "POST", "/choice", {"use": "anthropic"})).status_code == 200


async def test_keys_no_secret_opens_show_as_unreadable(byok, tmp_path):
    t = byok()
    await send(t.app, "POST", "/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY})
    # The operator lost the secret, or replaced it without keeping the old one.
    t.bot.own.store = KeyStore(t.store.path, bytes(32))
    listed = await send(t.app, "GET", "/keys")
    assert listed.status_code == 200 and listed.json()["keys"][0]["state"] == "unreadable"
    assert (await send(t.app, "POST", "/choice", {"use": "anthropic"})).status_code == 409


async def test_a_broken_store_is_a_503(byok, caplog):
    t = byok()
    t.store.path.unlink()
    t.store.path.mkdir()  # SQLite cannot open a directory
    response = await send(t.app, "GET", "/keys")
    assert response.status_code == 503 and response.json()["code"] == "store_unavailable"
    assert "the key store failed: OperationalError" in caplog.text


async def test_only_the_providers_and_babamul_are_contacted(byok):
    t = byok()
    await send(t.app, "POST", "/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY})
    await send(t.app, "POST", "/keys/openrouter", {"key": FAKE_OPENROUTER_KEY})
    babamul = httpx.URL(base_url_from_env()).host
    assert set(t.hosts) == {"api.anthropic.com", "openrouter.ai", babamul}
    assert {r.url.scheme for r in t.checks.requests} == {"https"}


async def test_expired_keys_are_swept_while_the_app_runs(byok, monkeypatch):
    t = byok()
    await send(t.app, "POST", "/keys/anthropic", {"key": FAKE_ANTHROPIC_KEY, "days": 1})
    monkeypatch.setattr(a, "SWEEP_EVERY", 0.05)
    t.store.clock = lambda: 4_000_000_000.0  # long after it expired
    async with t.bot.running():
        for _ in range(100):
            with sqlite3.connect(t.store.path) as db:
                if not db.execute("SELECT COUNT(*) FROM ai_keys").fetchone()[0]:
                    break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("the expired key was not swept")


def test_settings_from_the_environment(tmp_path):
    env = {"ASSISTANT_LLM_URL": "http://127.0.0.1:8081/v1"}
    assert a.Config.from_env(env).own_keys is None
    env["ASSISTANT_KEYS_DB"] = str(tmp_path / "keys.sqlite3")
    assert a.Config.from_env(env).own_keys is None  # a database, but no secret: still off
    env["ASSISTANT_KEYS_SECRET"] = base64.b64encode(KEYS_SECRET).decode()
    own = a.Config.from_env(env).own_keys
    assert list(own.providers) == ["anthropic", "openrouter"] and own.max_days == 90 and own.days == [1, 7, 30, 90]
    assert (own.concurrency, own.max_queue, own.rate_limit, own.rate_window) == (8, 16, 40, 600.0)
    assert (own.max_tokens, own.max_images) == (8192, 6)
    assert "AAAA" not in repr(own) and env["ASSISTANT_KEYS_SECRET"] not in repr(a.Config.from_env(env))

    own = a.Config.from_env({
        **env,
        "ASSISTANT_BYOK_PROVIDERS": " OpenRouter , anthropic,openrouter",
        "ASSISTANT_BYOK_ANTHROPIC_MODELS": "claude-sonnet-5-5, claude-haiku-5-5,",
        "ASSISTANT_BYOK_OPENROUTER_MODELS": "anthropic/claude-sonnet-5.5",
        "ASSISTANT_KEYS_MAX_DAYS": "30",
        "ASSISTANT_BYOK_CONCURRENCY": "3",
        "ASSISTANT_BYOK_MAX_TOKENS": "4096",
    }).own_keys
    assert list(own.providers) == ["openrouter", "anthropic"]
    assert own.providers["anthropic"].models == ("claude-sonnet-5-5", "claude-haiku-5-5")
    assert own.providers["openrouter"].default_model == "anthropic/claude-sonnet-5.5"
    assert own.days == [1, 7, 30] and own.default_days == 30 and own.concurrency == 3 and own.max_tokens == 4096

    for bad, says in (({"ASSISTANT_BYOK_PROVIDERS": "openai"}, "not offered yet"),
                      ({"ASSISTANT_BYOK_PROVIDERS": "gemini,anthropic"}, "not offered yet"),
                      ({"ASSISTANT_BYOK_PROVIDERS": "mistral"}, "not a provider"),
                      ({"ASSISTANT_BYOK_PROVIDERS": " , "}, "names no provider"),
                      ({"ASSISTANT_BYOK_ANTHROPIC_MODELS": "claude haiku"}, "ASSISTANT_BYOK_ANTHROPIC_MODELS"),
                      ({"ASSISTANT_KEYS_MAX_DAYS": "0"}, "ASSISTANT_KEYS_MAX_DAYS"),
                      ({"ASSISTANT_BYOK_RATE_LIMIT": "lots"}, "ASSISTANT_BYOK_RATE_LIMIT")):
        with pytest.raises(ValueError, match=says):
            a.Config.from_env({**env, **bad})


def test_the_presets_reach_only_fixed_https_hosts():
    for provider in providers.from_env({}).values():
        url = httpx.URL(provider.base_url)
        assert url.scheme == "https" and url.host in ("api.anthropic.com", "openrouter.ai")
        assert provider.default_model == provider.models[0]
    assert providers.from_env({})["openrouter"].body == {"provider": {"data_collection": "deny"}}
    assert providers.from_env({})["openrouter"].headers == (("X-OpenRouter-Title", "Babamul"),)
