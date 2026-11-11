"""Questions answered with a person's own key: its line, its request, its errors."""

import asyncio
import json
import logging
import time

import httpx
import pytest

from boom_mcp import assistant as a
from boom_mcp import providers as p
from boom_mcp.babamul import base_url_from_env
from tests.conftest import FAKE_ANTHROPIC_KEY, FAKE_OPENROUTER_KEY, LOGIN, LOGIN_NO_ID
from tests.test_assistant import OBJECT, chunk, events, say, stream, stream_bytes, types, until, use

pytestmark = pytest.mark.anyio

LOGIN_TWO = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyLTIifQ.dHdvLXNpZ25hdHVyZQ"
NEW_ANTHROPIC_KEY = "sk-ant-api03-FAKE" + "2" * 76 + "ne99"
ANTHROPIC_STATUS = "Answering with your Anthropic key …ab12 (claude-haiku-5-5)"
OPENROUTER_STATUS = "Answering with your OpenRouter key …cd34 (anthropic/claude-haiku-5.5)"
DONE = b"data: [DONE]\n\n"


async def ask(app, question="What is new tonight?", token=LOGIN, **extra):
    body = {"messages": [{"role": "user", "content": question}], **extra}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://boom") as client:
        return await client.post("/assistant/chat", json=body, headers={"Authorization": f"Bearer {token}"})


async def save(t, provider="anthropic", user="user-1", **kw):
    key = {"anthropic": FAKE_ANTHROPIC_KEY, "openrouter": FAKE_OPENROUTER_KEY}[provider]
    model = kw.pop("model", t.bot.own.providers[provider].default_model)
    await asyncio.to_thread(t.store.save, user, provider, key, model, kw.pop("days", 30), kw.pop("workspace_id", None))


def usage(prompt, completion) -> bytes:
    return f'data: {json.dumps({"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}})}\n\n'.encode()


def anthropic_error(status, kind, message):
    return httpx.Response(status, json={"type": "error", "error": {"type": kind, "message": message}})


def openrouter_error(status, message, **metadata):
    error = {"code": status, "message": message}
    if metadata:
        error["metadata"] = metadata
    return httpx.Response(status, json={"error": error})


def no_key_anywhere(t, *texts):
    """The key is in no response, no request to Babamul or the shared model, and no text given."""
    for text in texts:
        assert "FAKE" not in text
    for request in t.babamul.requests:
        assert "FAKE" not in str(request.url) + str(request.headers) + request.content.decode(errors="replace")
    for request in t.shared.requests:
        assert "FAKE" not in str(request.headers) + request.content.decode()


async def test_an_own_key_answers_with_a_standard_body_and_says_so(byok, caplog):
    first = use("get_object", {"object_id": OBJECT}, said="Let me look. ")
    t = byok(
        lambda body: stream_bytes(*[chunk(d) for d in first], usage(1200, 20), DONE),
        lambda body: stream_bytes(chunk({"content": "It is a "}), chunk({"content": "supernova."}), usage(1500, 8), DONE),
    )
    await save(t, workspace_id="wrkspc_01FAKE")
    with caplog.at_level(logging.DEBUG):
        response = await ask(t.app)
    evs = events(response)
    assert evs[0] == {"type": "status", "message": ANTHROPIC_STATUS}
    assert types(evs) == ["status", "text", "tool_call", "tool_result", "text", "text", "done"]
    assert evs[-1] == {"type": "done", "content": "It is a supernova.", "usage": {"input_tokens": 2700, "output_tokens": 28}}

    asked, again = t.own.requests
    assert [u for u in t.urls if "/chat/" in u] == ["https://api.anthropic.com/v1/chat/completions"] * 2
    assert list(asked.body) == ["model", "messages", "max_tokens", "tools", "stream_options", "stream"]
    assert asked.body["model"] == "claude-haiku-5-5" and asked.body["max_tokens"] == 8192
    assert asked.body["stream_options"] == {"include_usage": True}
    assert asked.headers["authorization"] == f"Bearer {FAKE_ANTHROPIC_KEY}"
    assert asked.headers["anthropic-workspace-id"] == "wrkspc_01FAKE" and "x-openrouter-title" not in asked.headers
    assert "you are given them too" in asked.body["messages"][0]["content"]  # own models see images
    assert "finish_reason" not in json.dumps(again.body) and "usage" not in json.dumps(again.body["messages"])

    # The shared model and its line were never involved, nor its fallback timer.
    assert t.shared.requests == [] and t.bot._main_skipped_until == 0 and not t.bot.limiter.hits.get("user-1")
    assert len(t.bot.own_limiter.hits["user-1"]) == 1
    assert set(t.hosts) == {"api.anthropic.com", httpx.URL(base_url_from_env()).host}
    entry = (await asyncio.to_thread(t.store.entries, "user-1"))[0]
    assert entry["last_used_at"] is not None and entry["state"] == "ok"
    assert "answered user user-1" in caplog.text and "on their Anthropic key" in caplog.text
    # The key goes in the Authorization header only, never into the conversation.
    assert all("FAKE" not in r.content.decode() for r in t.own.requests)
    no_key_anywhere(t, response.text, caplog.text)


async def test_openrouter_asks_for_providers_that_keep_nothing(byok):
    t = byok(say("Hi."))
    await save(t, "openrouter")
    evs = events(await ask(t.app))
    assert evs[0] == {"type": "status", "message": OPENROUTER_STATUS}
    assert evs[-1] == {"type": "done", "content": "Hi."}  # no usage reported, none shown
    sent = t.own.requests[0]
    assert [u for u in t.urls if "/chat/" in u] == ["https://openrouter.ai/api/v1/chat/completions"]
    assert list(sent.body) == ["model", "messages", "max_tokens", "tools", "stream_options", "provider", "stream"]
    assert sent.body["provider"] == {"data_collection": "deny"} and sent.body["model"] == "anthropic/claude-haiku-5.5"
    assert sent.headers["x-openrouter-title"] == "Babamul" and sent.headers["authorization"] == f"Bearer {FAKE_OPENROUTER_KEY}"
    assert "anthropic-workspace-id" not in sent.headers
    for field in ("temperature", "top_p", "chat_template_kwargs", "tool_choice"):
        assert field not in sent.body


async def test_api_tokens_and_the_shared_button_get_the_shared_model(byok):
    t = byok()
    await save(t)
    for token, extra in (("bbml_test", {}), (LOGIN, {"use": "shared"}), (LOGIN_NO_ID, {})):
        evs = events(await ask(t.app, token=token, **extra))
        assert evs[-1] == {"type": "done", "content": "From the shared model."}, token
        assert "status" not in types(evs)
    assert t.own.requests == [] and len(t.shared.requests) == 3
    assert "FAKE" not in json.dumps([r.body for r in t.shared.requests])
    bad = await ask(t.app, use="mine")
    assert bad.status_code == 400 and "shared" in bad.json()["error"]
    # Without a choice of their own, a login token gets the shared model too.
    await asyncio.to_thread(t.store.set_use, "user-1", "shared")
    assert events(await ask(t.app))[-1]["content"] == "From the shared model."


@pytest.mark.parametrize(
    "provider, reply, status, code",
    [
        ("anthropic", anthropic_error(401, "authentication_error", f"invalid x-api-key: {FAKE_ANTHROPIC_KEY}"), 422, "own_key_refused"),
        ("anthropic", anthropic_error(403, "permission_error", "not allowed"), 422, "own_key_refused"),
        ("anthropic", anthropic_error(400, "invalid_request_error", "Your credit balance is too low to access the API."),
         422, "own_key_no_credit"),
        ("anthropic", anthropic_error(400, "invalid_request_error", "You have reached your specified API usage limits."),
         422, "own_key_no_credit"),
        ("anthropic", anthropic_error(429, "rate_limit_error", "Your organization's enforced spend limit was reached."),
         422, "own_key_no_credit"),
        ("anthropic", anthropic_error(402, "billing_error", "billing"), 422, "own_key_no_credit"),
        ("anthropic", anthropic_error(429, "rate_limit_error", "Number of requests exceeded your rate limit."), 422, "own_key_busy"),
        ("anthropic", anthropic_error(404, "not_found_error", "model: claude-haiku-5-5"), 422, "own_model_unavailable"),
        ("anthropic", anthropic_error(529, "overloaded_error", "Overloaded"), 503, "own_provider_down"),
        ("anthropic", httpx.ConnectError("refused"), 503, "own_provider_down"),
        ("openrouter", openrouter_error(401, "User not found."), 422, "own_key_refused"),
        ("openrouter", openrouter_error(402, "Insufficient credits", limit_source="openrouter_credits"), 422, "own_key_no_credit"),
        ("openrouter", openrouter_error(403, "Key limit exceeded"), 422, "own_key_no_credit"),
        ("openrouter", openrouter_error(403, "Budget limit exceeded", limit_source="openrouter_key_limit"), 422, "own_key_no_credit"),
        ("openrouter", openrouter_error(402, "Busy", limit_source="openrouter_in_flight_budget"), 422, "own_key_busy"),
        ("openrouter", openrouter_error(429, "Rate limit exceeded"), 422, "own_key_busy"),
        ("openrouter", openrouter_error(404, "No endpoints found"), 422, "own_model_unavailable"),
        ("openrouter", openrouter_error(503, "No allowed providers are available"), 503, "own_provider_down"),
    ],
)
async def test_a_key_that_fails_before_the_answer_gets_a_code_and_never_a_401(byok, caplog, provider, reply, status, code):
    t = byok(reply)
    await save(t, provider)
    with caplog.at_level(logging.DEBUG):
        response = await ask(t.app)
    assert (response.status_code, response.json()["code"]) == (status, code)
    name = {"anthropic": "Anthropic", "openrouter": "OpenRouter"}[provider]
    assert name in response.json()["error"] and "shared model" in response.json()["error"]
    # Asked again at once only when the provider failed outright, never after a 429.
    assert len(t.own.requests) == (2 if code == "own_provider_down" else 1)
    assert t.shared.requests == [] and t.bot._main_skipped_until == 0
    assert not t.bot.own_limiter.hits.get("user-1")  # our failure or theirs, it does not count
    state = (await asyncio.to_thread(t.store.entries, "user-1"))[0]["state"]
    assert state == ("refused" if code == "own_key_refused" else "ok")
    no_key_anywhere(t, response.text, caplog.text)


async def test_a_failing_provider_leaves_the_operators_fallback_alone(byok):
    backup = a.Endpoint("http://llm.test/v1", "backup-model")
    t = byok(anthropic_error(529, "overloaded_error", "Overloaded"), config={"fallback": backup})
    await save(t)
    response = await ask(t.app)
    assert (response.status_code, response.json()["code"]) == (503, "own_provider_down")
    assert t.shared.requests == [] and t.bot._main_skipped_until == 0
    # Everyone else still starts with the operator's main model, not its fallback.
    evs = events(await ask(t.app, token="bbml_test"))
    assert evs[-1]["content"] == "From the shared model." and t.shared.requests[0].body["model"] == "test-model"


async def test_a_refused_key_is_not_tried_again_until_it_is_saved_again(byok):
    t = byok(anthropic_error(401, "authentication_error", "invalid"))
    await save(t)
    assert (await ask(t.app)).json()["code"] == "own_key_refused"
    again = await ask(t.app)
    assert (again.status_code, again.json()["code"]) == (422, "own_key_refused") and len(t.own.requests) == 1
    t.own.script = [say("Fixed.")]
    await save(t)
    assert events(await ask(t.app))[-1]["content"] == "Fixed."


def keeps_cookies(http: httpx.AsyncClient) -> bool:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    http.cookies.extract_cookies(httpx.Response(200, headers={"set-cookie": "s=FAKE; Path=/"}, request=request))
    return bool(http.cookies)


async def test_a_cookie_set_for_one_person_is_not_sent_for_another(byok):
    def with_cookie(body):
        headers = {"content-type": "text/event-stream", "set-cookie": "session=FAKE-COOKIE-OF-USER-1; Path=/; Secure"}
        return httpx.Response(200, headers=headers, content=stream(say("Done.")))

    t = byok(with_cookie)
    t.babamul.tokens[LOGIN_TWO] = "user-2"
    await save(t)
    await asyncio.to_thread(t.store.save, "user-2", "anthropic", NEW_ANTHROPIC_KEY, "claude-haiku-5-5", 30)
    for token in (LOGIN, LOGIN_TWO):
        assert events(await ask(t.app, token=token))[-1]["content"] == "Done."
    first, second = t.own.requests
    assert second.headers["authorization"] == f"Bearer {NEW_ANTHROPIC_KEY}"
    assert "cookie" not in first.headers and "cookie" not in second.headers

    # The client the assistant makes for itself keeps none either; httpx's own would.
    bot = a.Assistant(t.bot.config)
    async with bot.running():
        assert not keeps_cookies(bot.http)
    async with httpx.AsyncClient() as plain:
        assert keeps_cookies(plain)


async def test_a_refusal_marks_only_the_key_it_was_for(byok):
    asked, release = asyncio.Event(), asyncio.Event()

    async def revoked(body):
        asked.set()
        await release.wait()
        return anthropic_error(401, "authentication_error", "invalid x-api-key")

    t = byok(revoked)
    t.checks.good.add(NEW_ANTHROPIC_KEY)
    await save(t)
    question = asyncio.create_task(ask(t.app))
    await asyncio.wait_for(asked.wait(), 3)
    # While a question runs on the key they just revoked, they save a new one on Profile.
    body = {"key": NEW_ANTHROPIC_KEY, "model": "claude-haiku-5-5", "days": 7}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(t.app), base_url="http://boom") as client:
        saved = await client.post("/assistant/keys/anthropic", json=body, headers={"Authorization": f"Bearer {LOGIN}"})
    assert saved.status_code == 200 and saved.json()["hint"] == "…ne99"
    release.set()
    assert (await question).json()["code"] == "own_key_refused"
    entry = (await asyncio.to_thread(t.store.entries, "user-1"))[0]
    assert (entry["hint"], entry["state"]) == ("…ne99", "ok")
    t.own.script = [say("With the new key.")]
    assert events(await ask(t.app))[-1]["content"] == "With the new key."
    assert t.own.requests[-1].headers["authorization"] == f"Bearer {NEW_ANTHROPIC_KEY}"


async def test_one_failure_of_the_provider_is_asked_again(byok):
    t = byok(anthropic_error(529, "overloaded_error", "Overloaded"), say("Second try."))
    await save(t)
    assert events(await ask(t.app))[-1] == {"type": "done", "content": "Second try."}
    t = byok(httpx.ReadError("reset"), say("Second try."))
    await save(t)
    assert events(await ask(t.app))[-1]["content"] == "Second try."


async def test_other_refusals_are_about_the_conversation(byok):
    for provider, reply in (
        ("anthropic", anthropic_error(400, "invalid_request_error", "prompt is too long")),
        ("openrouter", openrouter_error(403, "Your input was flagged", reasons=["violence"])),
    ):
        t = byok(reply)
        await save(t, provider)
        response = await ask(t.app)
        assert response.status_code == 503 and response.json() == {"error": a.REFUSED}
        assert (await asyncio.to_thread(t.store.entries, "user-1"))[0]["state"] == "ok"  # the key is fine


async def test_a_key_that_fails_partway_ends_with_an_error_event_with_a_code(byok):
    error = {"code": 402, "message": "Insufficient credits", "metadata": {"error_type": "payment_required"}}
    choices = [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]
    mid = f"data: {json.dumps({'error': error, 'choices': choices})}\n\n".encode()
    t = byok(lambda body: stream_bytes(mid, DONE))
    await save(t, "openrouter")
    evs = events(await ask(t.app))
    assert evs[0] == {"type": "status", "message": OPENROUTER_STATUS}  # OpenRouter answered 200 first
    assert evs[-1]["type"] == "error" and evs[-1]["code"] == "own_key_no_credit" and "OpenRouter" in evs[-1]["message"]

    rate = b'data: {"error": {"code": 429, "message": "Rate limit exceeded", "metadata": {"error_type": "rate_limit_exceeded"}}}\n\n'
    t = byok(lambda body: stream_bytes(chunk({"content": "It is"}), rate, DONE))
    await save(t, "openrouter")
    evs = events(await ask(t.app))
    assert evs[-2:] == [{"type": "text", "delta": "It is"}, {**evs[-1], "type": "error", "code": "own_key_busy"}]
    assert len(t.own.requests) == 1

    # Anthropic's own error event, overloaded, after some text: a stall of their provider.
    overloaded = b'data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}\n\n'
    t = byok(lambda body: stream_bytes(chunk({"content": "It is"}), overloaded, DONE))
    await save(t)
    evs = events(await ask(t.app))
    assert evs[-1]["code"] == "own_provider_down" and len(t.own.requests) == 1

    # A reply that just stops, with nothing said: their provider stopped.
    t = byok(lambda body: stream_bytes(chunk({"role": "assistant"})))
    await save(t)
    evs = events(await ask(t.app))
    assert evs[-1] == {"type": "error", "message": a.STALLED, "code": "own_provider_down"}

    # A refusal in the second round, after a tool ran.
    t = byok(use("convert_time"), anthropic_error(401, "authentication_error", "revoked"))
    await save(t)
    evs = events(await ask(t.app))
    assert types(evs) == ["status", "tool_call", "tool_result", "error"]
    assert evs[-1]["code"] == "own_key_refused" and t.shared.requests == []


async def test_a_provider_that_fails_in_its_stream_before_any_text_is_asked_again(byok):
    # OpenRouter's failure after its 200: a code that is a string, and finish_reason "error".
    error = {"code": "server_error", "message": "Upstream disconnected"}
    choices = [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]
    failed = f"data: {json.dumps({'error': error, 'choices': choices})}\n\n".encode()
    t = byok(lambda body: stream_bytes(failed, DONE), say("Second."))
    await save(t, "openrouter")
    evs = events(await ask(t.app))
    assert evs == [
        {"type": "status", "message": OPENROUTER_STATUS},
        {"type": "text", "delta": "Second."},
        {"type": "done", "content": "Second."},
    ]
    assert len(t.own.requests) == 2

    # Twice: their provider is down.
    t = byok(lambda body: stream_bytes(failed, DONE))
    await save(t, "openrouter")
    evs = events(await ask(t.app))
    assert evs[-1]["code"] == "own_provider_down" and "OpenRouter" in evs[-1]["message"] and len(t.own.requests) == 2

    # After some text, never: the person is reading the answer.
    t = byok(lambda body: stream_bytes(chunk({"content": "It is"}), failed, DONE), say("Second."))
    await save(t, "openrouter")
    evs = events(await ask(t.app))
    assert evs[-1] == {"type": "error", "message": a.STALLED, "code": "own_provider_down"}
    assert len(t.own.requests) == 1


async def test_a_chosen_key_that_cannot_be_used_never_falls_back(byok):
    t = byok()
    await save(t, days=1)
    t.store.clock = lambda: time.time() + 2 * 86400  # it expired
    response = await ask(t.app)
    assert (response.status_code, response.json()["code"]) == (422, "own_key_missing")
    assert "Anthropic" in response.json()["error"] and t.own.requests == [] and t.shared.requests == []
    assert await asyncio.to_thread(t.store.use, "user-1") == "anthropic"  # still their choice

    t = byok()
    await save(t)
    t.bot.own.store = type(t.store)(t.store.path, bytes(32))  # the secret changed: the row no longer opens
    assert (await ask(t.app)).json()["code"] == "own_key_missing"

    t = byok(env={"ASSISTANT_BYOK_ANTHROPIC_MODELS": "claude-sonnet-5-5"})
    await save(t, model="claude-haiku-5-5")  # a model the operator no longer offers
    response = await ask(t.app)
    assert response.json()["code"] == "own_model_unavailable" and "claude-haiku-5-5" in response.json()["error"]

    t = byok(env={"ASSISTANT_BYOK_PROVIDERS": "openrouter"})
    await asyncio.to_thread(t.store.save, "user-1", "anthropic", FAKE_ANTHROPIC_KEY, "claude-haiku-5-5", 30)
    # A provider the operator stopped offering: the shared model, as GET /keys says.
    assert events(await ask(t.app))[-1]["content"] == "From the shared model."
    assert t.shared.requests and t.own.requests == []


async def test_a_broken_store_leaves_everyone_the_shared_model(byok, caplog):
    t = byok()
    await save(t)
    t.store.path.unlink()  # removed while the server runs
    for token in ("bbml_test", LOGIN):
        evs = events(await ask(t.app, token=token))
        assert evs[-1] == {"type": "done", "content": "From the shared model."}, token
    # They may have chosen their own key, so the status line says why it is not used.
    assert evs[0] == {"type": "status", "message": f"Answering with {a.STORE_DOWN_LABEL}"}
    assert t.own.requests == [] and len(t.shared.requests) == 2
    assert "the key store failed, so user user-1 gets the shared model: OperationalError" in caplog.text
    assert not t.store.path.exists()  # not made again, empty

    t.store.path.mkdir()  # SQLite cannot open a directory
    t.babamul.tokens[LOGIN_TWO] = "user-2"  # who never saved a key
    assert events(await ask(t.app, token=LOGIN_TWO))[-1]["content"] == "From the shared model."


async def test_an_own_key_question_does_not_wait_for_the_shared_line(byok):
    gate = asyncio.Event()
    t = byok(say("From your own key."), shared=[stream(say("Busy "), gate, say("done."))],
             config={"concurrency": 1, "max_queue": 0})
    t.babamul.tokens.update({"bbml_two": "user-2", "bbml_three": "user-3"})
    await save(t)
    busy = asyncio.create_task(ask(t.app, token="bbml_two"))
    await until(lambda: t.shared.requests)
    full = await ask(t.app, token="bbml_three")
    assert full.status_code == 429 and full.json() == {"error": a.LINE_FULL}  # the shared line is full

    started = time.monotonic()
    evs = events(await asyncio.wait_for(ask(t.app), 3))
    assert evs[-1] == {"type": "done", "content": "From your own key."} and time.monotonic() - started < 2
    assert t.bot.own_line.answering == set() and len(t.bot.line.answering) == 1
    gate.set()
    assert events(await busy)[-1]["content"] == "Busy done."


async def test_own_key_questions_have_a_line_and_limit_of_their_own(byok):
    gate = asyncio.Event()
    t = byok(stream(say("Slow "), gate, say("answer.")), say("Quick."), concurrency=1, max_queue=0, rate_limit=2)
    t.babamul.tokens[LOGIN_TWO] = "user-2"
    await save(t)
    await save(t, user="user-2")
    first = asyncio.create_task(ask(t.app))
    await until(lambda: t.own.requests)
    full = await ask(t.app, token=LOGIN_TWO)
    assert full.status_code == 429 and full.json() == {"error": a.LINE_FULL}
    shared = await ask(t.app, token="bbml_test", question="meanwhile")  # the shared line is free
    assert events(shared)[-1]["content"] == "From the shared model."  # and it replaced the slow one
    gate.set()
    await first
    assert t.bot.own_line.answering == set()

    assert events(await ask(t.app))[-1]["content"] == "Quick."
    limited = await ask(t.app)
    assert limited.status_code == 429 and "Try again in" in limited.json()["error"]
    assert len(t.bot.own_limiter.hits["user-1"]) == 2 and len(t.bot.limiter.hits["user-1"]) == 1


async def test_own_models_see_the_newest_images_only(byok):
    t = byok(use("get_cutouts", {"object_id": OBJECT}), use("get_cutouts", {"object_id": OBJECT}),
             use("get_cutouts", {"object_id": OBJECT}), say("Three point sources."), max_images=2)
    await save(t)
    evs = events(await ask(t.app, question="Show me the cutouts"))
    assert evs[-1]["content"] == "Three point sources."
    last = t.own.requests[-1].body["messages"]
    image_messages = [m for m in last if m["role"] == "user" and m is not last[1]]
    assert [m["content"] if isinstance(m["content"], str) else "images" for m in image_messages] == [
        a.IMAGES_LEFT_OUT, "images", "images",
    ]
    parts = [part for m in last if isinstance(m["content"], list) for part in m["content"]]
    urls = [part["image_url"]["url"] for part in parts if part["type"] == "image_url"]
    assert len(urls) == 2 and all(u.startswith("data:image/png;base64,") for u in urls)
    assert all("attached below for you" in m["content"] for m in last if m["role"] == "tool")


def test_fit_keeps_the_newest_images():
    def images(n):
        return a.image_message([{"mime_type": "image/png", "data": f"IMG{i}"} for i in range(n)])

    messages = [{"role": "system", "content": "s"}, images(2), {"role": "tool", "content": "x"}, images(3)]
    a.fit(messages, 10_000, max_images=4)
    assert messages[1]["content"][0]["type"] == "text" and len(messages[1]["content"]) == 2  # one of two kept
    assert len(messages[3]["content"]) == 4  # all three of the newest
    a.fit(messages, 10_000, max_images=1)
    assert messages[1]["content"] == a.IMAGES_LEFT_OUT and len(messages[3]["content"]) == 2
    unlimited = [images(9)]
    a.fit(unlimited, 10_000)
    assert len(unlimited[0]["content"]) == 10


async def test_the_last_round_keeps_the_tools_but_calls_none(byok):
    t = byok(lambda body: use("convert_time") if body.get("tool_choice") != "none" else say("It is late."),
             config={"max_tool_calls": 1})
    await save(t)
    evs = events(await ask(t.app, question="What time is it?"))
    assert evs[-1] == {"type": "done", "content": "It is late."}
    final = t.own.requests[-1].body
    assert final["tool_choice"] == "none" and final["tools"] == t.own.requests[0].body["tools"]
    assert final["messages"][-1] == {"role": "user", "content": a.ANSWER_NOW}
    assert "tool_choice" not in t.own.requests[0].body


async def test_keys_stay_out_of_logs_even_when_something_breaks(byok, caplog):
    leak = httpx.LocalProtocolError(f"Illegal header value b'Bearer {FAKE_ANTHROPIC_KEY}\\n'")
    t = byok(leak)
    await save(t)
    with caplog.at_level(logging.DEBUG):
        response = await ask(t.app)
    assert (response.status_code, response.json()["code"]) == (503, "own_provider_down")
    assert "LocalProtocolError" in caplog.text
    no_key_anywhere(t, response.text, caplog.text)
    caplog.clear()

    # Not an httpx error at all, so it reaches the catch-all, which logs a traceback.
    t = byok(RuntimeError(f"something quoted {FAKE_ANTHROPIC_KEY}"))
    await save(t)
    with caplog.at_level(logging.DEBUG):
        response = await ask(t.app)
    assert response.status_code == 500 and response.json() == {"error": a.BROKEN}
    assert "the assistant failed for user user-1 on their own key" in caplog.text
    assert "RuntimeError: something quoted (their key)" in caplog.text
    no_key_anywhere(t, response.text, caplog.text)


def test_classify():
    cases = [
        (p.ANTHROPIC, {"error": {"type": "authentication_error"}}, 401, p.KEY_REFUSED),
        (p.ANTHROPIC, {"type": "error", "error": {"type": "overloaded_error"}}, None, p.DOWN),
        (p.ANTHROPIC, {"type": "error", "error": {"type": "rate_limit_error"}}, None, p.BUSY),
        (p.ANTHROPIC, {"error": {"type": "invalid_request_error", "message": "prompt is too long"}}, 400, None),
        (p.ANTHROPIC, b"<html>not json</html>", 500, p.DOWN),
        (p.ANTHROPIC, b"", 408, p.DOWN),
        (p.OPENROUTER, {"error": {"code": 403, "message": "flagged"}}, 403, None),
        (p.OPENROUTER, {"error": {"code": 401, "metadata": {"error_type": "authentication"}}}, None, p.KEY_REFUSED),
        (p.OPENROUTER, {"error": {"code": 400, "metadata": {"error_type": "context_length_exceeded"}}}, None, None),
        (p.OPENROUTER, {"error": {"message": "x", "type": "insufficient_quota", "code": "insufficient_quota"}}, 429, p.NO_CREDIT),
        (p.OPENROUTER, {"error": {"code": "organization_spend_limit_exceeded"}}, 429, p.NO_CREDIT),
        (p.OPENROUTER, {"error": "not an object"}, 418, None),
    ]
    for provider, body, status, code in cases:
        assert p.classify(provider, body, status) == code, (provider, body, status)
