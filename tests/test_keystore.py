"""The store of people's model keys: sealed, bound to their owner, private on disk."""

import base64
import hashlib
import os
import sqlite3
import stat
from contextlib import closing

import pytest

from boom_mcp import keystore as k

FAKE_KEY = "sk-ant-api03-FAKE0000000000000000000000000000000000000000000000000000XyZ9"
OTHER_KEY = "sk-or-v1-FAKE000000000000000000000000000000000000000000000000000000ab12"
SECRET = bytes(range(32))
NEW_SECRET = bytes(range(32, 64))


class Clock:
    def __init__(self, now=1_800_000_000):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return k.KeyStore(tmp_path / "run" / "assistant" / "keys.sqlite3", SECRET, clock=clock)


def test_a_saved_key_opens_for_its_owner_and_shows_only_metadata(store, clock):
    entry = store.save("user-1", "anthropic", FAKE_KEY, "claude-haiku-5-5", 30, workspace_id="wrkspc_01FAKE")
    assert entry == {
        "provider": "anthropic",
        "hint": "…XyZ9",
        "model": "claude-haiku-5-5",
        "workspace_id": "wrkspc_01FAKE",
        "created_at": "2027-01-15T08:00:00Z",
        "expires_at": "2027-02-14T08:00:00Z",
        "last_used_at": None,
        "state": "ok",
    }
    assert store.entries("user-1") == [entry] and store.entries("user-2") == []
    assert store.use("user-1") == "anthropic" and store.use("user-2") == "shared"

    saved = store.key("user-1", "anthropic")
    assert saved.key == FAKE_KEY and saved.model == "claude-haiku-5-5" and saved.state == "ok"
    assert FAKE_KEY not in repr(saved) and store.key("user-2", "anthropic") is None

    clock.now += 60
    store.touch("user-1", "anthropic")
    assert store.entries("user-1")[0]["last_used_at"] == "2027-01-15T08:01:00Z"
    store.mark("user-1", "anthropic", k.REFUSED)
    assert store.entries("user-1")[0]["state"] == "refused" and store.key("user-1", "anthropic").state == "refused"
    # Saving again replaces the key and starts it afresh.
    store.save("user-1", "anthropic", FAKE_KEY[:-4] + "Abcd", "claude-sonnet-5-5", 7, use=False)
    assert [(e["hint"], e["model"], e["state"], e["last_used_at"]) for e in store.entries("user-1")] == [
        ("…Abcd", "claude-sonnet-5-5", "ok", None)
    ]


def test_touch_and_mark_with_a_nonce_change_only_that_key(tmp_path, clock):
    store = k.KeyStore(tmp_path / "keys.sqlite3", SECRET, clock=clock)
    store.save("user-1", "anthropic", FAKE_KEY, "claude-haiku-5-5", 30)
    old = store.key("user-1", "anthropic")
    assert old.nonce and "nonce" not in repr(old)
    # Saved anew while a question on the old key was running.
    store.save("user-1", "anthropic", FAKE_KEY[:-4] + "Abcd", "claude-haiku-5-5", 30)
    clock.now += 60
    store.touch("user-1", "anthropic", nonce=old.nonce)
    store.mark("user-1", "anthropic", k.REFUSED, nonce=old.nonce)
    assert [(e["hint"], e["state"], e["last_used_at"]) for e in store.entries("user-1")] == [("…Abcd", "ok", None)]

    # The nonce key() hands out is the row's, after it is sealed again under a new secret too.
    rotated = k.KeyStore(store.path, NEW_SECRET, old_secrets=[SECRET], clock=clock)
    new = rotated.key("user-1", "anthropic")
    assert new.nonce != old.nonce
    rotated.touch("user-1", "anthropic", nonce=new.nonce)
    rotated.mark("user-1", "anthropic", k.REFUSED, nonce=new.nonce)
    assert [(e["state"], e["last_used_at"]) for e in rotated.entries("user-1")] == [("refused", "2027-01-15T08:01:00Z")]


def test_a_row_moved_to_another_person_or_provider_does_not_open(store):
    store.save("user-1", "anthropic", FAKE_KEY, "claude-haiku-5-5", 30)
    store.save("user-2", "anthropic", OTHER_KEY, "claude-haiku-5-5", 30)
    with sqlite3.connect(store.path) as db:
        # Someone who can write to the file, but has no secret, swaps the two ciphertexts
        # and copies one row to another provider.
        rows = dict(db.execute("SELECT user_id, ciphertext FROM ai_keys"))
        nonces = dict(db.execute("SELECT user_id, nonce FROM ai_keys"))
        db.execute("UPDATE ai_keys SET ciphertext = ?, nonce = ? WHERE user_id = 'user-2'", (rows["user-1"], nonces["user-1"]))
        db.execute(
            "INSERT INTO ai_keys SELECT user_id, 'openrouter', nonce, ciphertext, kid, hint, model, workspace_id, "
            "created_at, expires_at, last_used_at, state FROM ai_keys WHERE user_id = 'user-1'"
        )
    assert store.key("user-1", "anthropic").key == FAKE_KEY  # the original still opens
    moved = store.key("user-2", "anthropic")
    assert moved.key == "" and moved.state == "unreadable"
    assert store.key("user-1", "openrouter").state == "unreadable"
    assert [e["state"] for e in store.entries("user-1")] == ["ok", "unreadable"]
    assert [e["state"] for e in store.entries("user-2")] == ["unreadable"]


def test_a_new_secret_still_opens_rows_under_the_old_one_and_seals_them_again(tmp_path, clock):
    path = tmp_path / "keys.sqlite3"
    k.KeyStore(path, SECRET, clock=clock).save("user-1", "anthropic", FAKE_KEY, "claude-haiku-5-5", 30)

    without_old = k.KeyStore(path, NEW_SECRET, clock=clock)
    assert without_old.entries("user-1")[0]["state"] == "unreadable"  # no 500, just "enter it again"
    assert without_old.key("user-1", "anthropic").state == "unreadable"

    rotated = k.KeyStore(path, NEW_SECRET, old_secrets=[SECRET], clock=clock)
    assert rotated.entries("user-1")[0]["state"] == "ok"
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT kid FROM ai_keys").fetchone()[0] == k.key_id(SECRET)  # listing changes nothing
    assert rotated.key("user-1", "anthropic").key == FAKE_KEY
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT kid FROM ai_keys").fetchone()[0] == k.key_id(NEW_SECRET)
    # Sealed again under the new secret, it no longer needs the old one.
    assert k.KeyStore(path, NEW_SECRET, clock=clock).key("user-1", "anthropic").key == FAKE_KEY
    assert k.key_id(SECRET) == hashlib.sha256(SECRET).hexdigest()[:8]


def test_expired_keys_are_forgotten_when_read_and_by_the_sweep(store, clock):
    store.save("user-1", "anthropic", FAKE_KEY, "claude-haiku-5-5", 1)
    store.save("user-1", "openrouter", OTHER_KEY, "anthropic/claude-haiku-5.5", 7)
    store.save("user-2", "anthropic", FAKE_KEY, "claude-haiku-5-5", 1)
    clock.now += k.DAY
    assert [e["provider"] for e in store.entries("user-1")] == ["openrouter"]
    assert store.key("user-2", "anthropic") is None
    store.save("user-3", "anthropic", FAKE_KEY, "claude-haiku-5-5", 1)
    clock.now += 7 * k.DAY
    assert store.sweep() == 2  # user-1's OpenRouter key and user-3's
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM ai_keys").fetchone()[0] == 0
    # The choice stays, so a question finds the key gone instead of quietly using another model.
    assert store.use("user-1") == "openrouter"


def test_choices_and_deleting(store):
    store.save("user-1", "anthropic", FAKE_KEY, "claude-haiku-5-5", 30)
    store.save("user-1", "openrouter", OTHER_KEY, "anthropic/claude-haiku-5.5", 30, use=False)
    assert store.use("user-1") == "anthropic"
    store.set_use("user-1", "openrouter", model="anthropic/claude-sonnet-5.5")
    assert store.use("user-1") == "openrouter" and store.key("user-1", "openrouter").model == "anthropic/claude-sonnet-5.5"
    store.delete("user-1", "anthropic")  # not the one in use
    assert store.use("user-1") == "openrouter" and [e["provider"] for e in store.entries("user-1")] == ["openrouter"]
    store.delete("user-1", "openrouter")  # the one in use: back to the shared model
    assert store.use("user-1") == "shared" and store.entries("user-1") == []

    store.save("user-1", "anthropic", FAKE_KEY, "claude-haiku-5-5", 30)
    store.save("user-2", "anthropic", FAKE_KEY, "claude-haiku-5-5", 30)
    store.delete_all("user-1")
    assert store.entries("user-1") == [] and store.use("user-1") == "shared"
    assert store.use("user-2") == "anthropic"
    store.set_use("user-2", "shared")
    assert store.use("user-2") == "shared" and store.entries("user-2")[0]["provider"] == "anthropic"


@pytest.fixture
def open_umask():
    """What a shell usually has: new files readable by everyone."""
    old = os.umask(0o022)
    yield
    os.umask(old)


def test_the_file_is_private_and_holds_no_key(tmp_path, clock, open_umask):
    store = k.KeyStore(tmp_path / "new" / "keys.sqlite3", SECRET, clock=clock)
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(store.path.parent).st_mode) == 0o700
    for i in range(20):
        store.save(f"user-{i}", "anthropic", FAKE_KEY, "claude-haiku-5-5", 30)
        store.save(f"user-{i}", "openrouter", OTHER_KEY, "anthropic/claude-haiku-5.5", 30)
    store.delete_all("user-0")
    raw = store.path.read_bytes()
    assert b"FAKE" not in raw and FAKE_KEY[-12:].encode() not in raw and OTHER_KEY[-12:].encode() not in raw
    assert not (tmp_path / "new" / "keys.sqlite3-wal").exists()
    with sqlite3.connect(store.path) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"

    # A file left readable by others is made private again.
    os.chmod(store.path, 0o644)
    k.KeyStore(store.path, SECRET, clock=clock)
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600


def test_a_file_removed_while_running_is_not_made_again(store):
    store.save("user-1", "anthropic", FAKE_KEY, "claude-haiku-5-5", 30)
    store.path.unlink()
    with pytest.raises(sqlite3.OperationalError):
        store.use("user-1")
    # Made again, it would have no tables, and the umask's mode rather than 0600.
    assert not store.path.exists()


def test_a_deleted_key_is_overwritten_in_the_file(tmp_path, clock, monkeypatch):
    # Some SQLite builds overwrite deleted rows by default, and most do not: start
    # every connection with it off, so the store has to turn it on itself.
    real = sqlite3.connect

    def connect(*args, **kwargs):
        db = real(*args, **kwargs)
        db.execute("PRAGMA secure_delete=OFF")
        return db

    monkeypatch.setattr(k.sqlite3, "connect", connect)
    store = k.KeyStore(tmp_path / "keys.sqlite3", SECRET, clock=clock)
    for i in range(5):
        store.save(f"user-{i}", "anthropic", FAKE_KEY, "claude-haiku-5-5", 30)
    with closing(real(store.path)) as db:
        sealed = db.execute("SELECT ciphertext FROM ai_keys WHERE user_id = 'user-2'").fetchone()[0]
    assert sealed in store.path.read_bytes()
    store.delete("user-2", "anthropic")
    assert sealed not in store.path.read_bytes()  # overwritten, not just unlinked from its page


def test_settings_from_the_environment(tmp_path):
    assert k.KeyStore.from_env({}) is None
    db = str(tmp_path / "keys.sqlite3")
    encoded = base64.b64encode(SECRET).decode()
    assert k.KeyStore.from_env({"ASSISTANT_KEYS_DB": db}) is None  # no secret: off
    assert k.KeyStore.from_env({"ASSISTANT_KEYS_DB": db, "ASSISTANT_KEYS_SECRET": " "}) is None
    assert k.KeyStore.from_env({"ASSISTANT_KEYS_SECRET": encoded}) is None  # no database: off

    store = k.KeyStore.from_env({"ASSISTANT_KEYS_DB": db, "ASSISTANT_KEYS_SECRET": encoded + "\n"})
    assert store.kid == k.key_id(SECRET)

    secret_file, old_file = tmp_path / "secret", tmp_path / "old"
    for path, secret in ((secret_file, NEW_SECRET), (old_file, SECRET)):
        path.write_text(base64.b64encode(secret).decode() + "\n")
        path.chmod(0o600)
    env = {"ASSISTANT_KEYS_DB": db, "ASSISTANT_KEYS_SECRET_FILE": str(secret_file), "ASSISTANT_KEYS_SECRET": encoded}
    assert k.KeyStore.from_env(env).kid == k.key_id(NEW_SECRET)  # the file wins
    env["ASSISTANT_KEYS_OLD_SECRET_FILE"] = str(old_file)
    assert set(k.KeyStore.from_env(env).ciphers) == {k.key_id(SECRET), k.key_id(NEW_SECRET)}

    secret_file.chmod(0o640)
    with pytest.raises(ValueError, match="chmod 600"):
        k.KeyStore.from_env(env)
    secret_file.chmod(0o600)
    for bad in ("not base64!", base64.b64encode(b"short").decode()):
        with pytest.raises(ValueError) as failed:
            k.KeyStore.from_env({"ASSISTANT_KEYS_DB": db, "ASSISTANT_KEYS_SECRET": bad})
        assert "ASSISTANT_KEYS_SECRET" in str(failed.value) and bad not in str(failed.value)
    with pytest.raises(ValueError, match="cannot be read"):
        k.KeyStore.from_env({"ASSISTANT_KEYS_DB": db, "ASSISTANT_KEYS_SECRET_FILE": str(tmp_path / "missing")})
