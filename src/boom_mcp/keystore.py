"""Model API keys people save for the assistant, encrypted at rest in one SQLite file.

Keys are write-only: nothing here hands back a key except key(), which the
assistant calls to answer one question with it. A key is sealed with
AES-256-GCM under a secret only this server has, and the row's user and
provider are bound in as associated data, so a row moved to another person
or provider no longer opens.

sqlite3 blocks, so the assistant calls these methods in a thread. Each opens
its own short connection: the file is small, and a connection per call needs
no lock between threads.
"""

import base64
import hashlib
import os
import sqlite3
import stat
import time
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Bound into every ciphertext with the row's user and provider. A new layout gets a new version.
AAD_PREFIX = b"boom-mcp/ai-key/v1\0"
SECRET_BYTES = 32
NONCE_BYTES = 12
DAY = 86400

SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_keys (
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    nonce BLOB NOT NULL,
    ciphertext BLOB NOT NULL,
    kid TEXT NOT NULL,
    hint TEXT NOT NULL,
    model TEXT NOT NULL,
    workspace_id TEXT,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    last_used_at INTEGER,
    state TEXT NOT NULL DEFAULT 'ok',
    PRIMARY KEY (user_id, provider)
);
CREATE TABLE IF NOT EXISTS ai_choice (
    user_id TEXT PRIMARY KEY,
    use TEXT NOT NULL
);
"""

SHARED = "shared"
# A row's state. "unreadable" is never stored: it is what a row no secret here opens shows as.
OK, REFUSED, UNREADABLE = "ok", "refused", "unreadable"


def key_id(secret: bytes) -> str:
    """Which secret sealed a row, without saying anything about the secret."""
    return hashlib.sha256(secret).hexdigest()[:8]


def aad(user_id: str, provider: str) -> bytes:
    return AAD_PREFIX + user_id.encode() + b"\0" + provider.encode()


def hint(key: str) -> str:
    """The last 4 characters, enough for a person to tell their keys apart."""
    return "…" + key[-4:]


def iso(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_secret(raw: str, what: str) -> bytes:
    """A secret from its base64 text. The error names where it came from, never what it holds."""
    try:
        secret = base64.b64decode(raw.strip(), validate=True)
    except ValueError:
        raise ValueError(f"{what} is not base64") from None
    if len(secret) != SECRET_BYTES:
        raise ValueError(f"{what} must hold {SECRET_BYTES} random bytes, base64-encoded")
    return secret


def read_secret_file(path: str, what: str) -> bytes:
    """A secret from a file only this user can read. A file others can read is refused,
    since whoever reads it can open every key."""
    try:
        mode = os.stat(path).st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ValueError(f"{what} ({path}) can be read by other users: chmod 600 it")
        with open(path, encoding="ascii") as f:
            raw = f.read()
    except OSError as exc:
        raise ValueError(f"{what} ({path}) cannot be read: {type(exc).__name__}") from None
    except UnicodeDecodeError:
        raise ValueError(f"{what} ({path}) is not base64") from None
    return parse_secret(raw, what)


@dataclass(frozen=True)
class Saved:
    """One person's key for one provider, opened for one question. key is empty
    when the row could not be opened."""

    provider: str
    key: str = field(repr=False)
    model: str
    workspace_id: str | None
    hint: str
    state: str
    # The sealed row's nonce, new with every save: which key touch() and mark() are about.
    nonce: bytes = field(default=b"", repr=False)


class KeyStore:
    def __init__(self, path, secret: bytes, old_secrets=(), clock=time.time):
        if len(secret) != SECRET_BYTES:
            raise ValueError(f"the key store's secret must be {SECRET_BYTES} bytes")
        self.path = Path(path)
        self.clock = clock
        self.kid = key_id(secret)
        # Rows sealed under an old secret still open, and are sealed again when next used.
        self.ciphers = {key_id(s): AESGCM(s) for s in old_secrets if len(s) == SECRET_BYTES}
        self.ciphers[self.kid] = AESGCM(secret)
        self._create()

    @classmethod
    def from_env(cls, env=None) -> "KeyStore | None":
        """The store in ASSISTANT_KEYS_* variables, or None when the database or the secret is unset."""
        env = os.environ if env is None else env
        path = env.get("ASSISTANT_KEYS_DB", "").strip()
        secret_file = env.get("ASSISTANT_KEYS_SECRET_FILE", "").strip()
        secret_text = env.get("ASSISTANT_KEYS_SECRET", "").strip()
        if not path or not (secret_file or secret_text):
            return None
        if secret_file:
            secret = read_secret_file(secret_file, "ASSISTANT_KEYS_SECRET_FILE")
        else:
            secret = parse_secret(secret_text, "ASSISTANT_KEYS_SECRET")
        old = []
        if old_file := env.get("ASSISTANT_KEYS_OLD_SECRET_FILE", "").strip():
            old.append(read_secret_file(old_file, "ASSISTANT_KEYS_OLD_SECRET_FILE"))
        return cls(path, secret, old)

    def _create(self):
        # Made private before SQLite first writes to it, and the journal SQLite
        # keeps next to it takes the same mode.
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.close(os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600))
        os.chmod(self.path, 0o600)
        with self._connect() as db:
            # A rollback journal, not WAL, so that secure_delete overwrites removed rows
            # in the file itself rather than leaving them in a log beside it.
            db.execute("PRAGMA journal_mode=DELETE")
            db.executescript(SCHEMA)

    def _connect(self):
        # Never made here (mode=rw): a file removed while the server runs would come
        # back empty, with no tables and the umask's mode. _create makes it.
        db = sqlite3.connect(f"{self.path.absolute().as_uri()}?mode=rw", uri=True, timeout=10)
        db.execute("PRAGMA secure_delete=ON")
        return _Connection(db)

    def _seal(self, user_id: str, provider: str, key: str) -> tuple[bytes, bytes, str]:
        nonce = os.urandom(NONCE_BYTES)
        return nonce, self.ciphers[self.kid].encrypt(nonce, key.encode(), aad(user_id, provider)), self.kid

    def _open(self, user_id: str, provider: str, nonce: bytes, ciphertext: bytes, kid: str) -> str | None:
        cipher = self.ciphers.get(kid)
        if cipher is None:
            return None
        try:
            return cipher.decrypt(nonce, ciphertext, aad(user_id, provider)).decode("ascii")
        except (InvalidTag, ValueError):
            return None

    def _forget_expired(self, db, user_id: str):
        db.execute("DELETE FROM ai_keys WHERE user_id = ? AND expires_at <= ?", (user_id, int(self.clock())))

    # What the routes call.

    def save(self, user_id: str, provider: str, key: str, model: str, days: int, workspace_id=None, use=True) -> dict:
        """Store key in place of any the person had for provider, and make it their choice if use."""
        now = int(self.clock())
        nonce, ciphertext, kid = self._seal(user_id, provider, key)
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO ai_keys (user_id, provider, nonce, ciphertext, kid, hint, model, "
                "workspace_id, created_at, expires_at, last_used_at, state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                (user_id, provider, nonce, ciphertext, kid, hint(key), model, workspace_id, now, now + days * DAY, OK),
            )
            if use:
                self._set_use(db, user_id, provider)
            return self._entries(db, user_id, provider)[0]

    def entries(self, user_id: str) -> list[dict]:
        """What the person may see of their keys: never a key or its ciphertext."""
        with self._connect() as db:
            self._forget_expired(db, user_id)
            return self._entries(db, user_id)

    def _entries(self, db, user_id: str, provider: str | None = None) -> list[dict]:
        query = (
            "SELECT provider, nonce, ciphertext, kid, hint, model, workspace_id, created_at, expires_at, "
            "last_used_at, state FROM ai_keys WHERE user_id = ?"
        )
        args = [user_id]
        if provider is not None:
            query += " AND provider = ?"
            args.append(provider)
        entries = []
        for name, nonce, ciphertext, kid, shown, model, workspace, created, expires, used, state in db.execute(
            query + " ORDER BY provider", args
        ):
            # Opened only to tell whether it still opens; the key itself goes nowhere.
            if self._open(user_id, name, nonce, ciphertext, kid) is None:
                state = UNREADABLE
            entries.append(
                {
                    "provider": name,
                    "hint": shown,
                    "model": model,
                    "workspace_id": workspace,
                    "created_at": iso(created),
                    "expires_at": iso(expires),
                    "last_used_at": iso(used),
                    "state": state if state in (OK, REFUSED, UNREADABLE) else OK,
                }
            )
        return entries

    def key(self, user_id: str, provider: str) -> Saved | None:
        """The person's key for provider, opened to answer a question; None if they have none
        that has not expired. One sealed under an old secret is sealed again under the current one."""
        with self._connect() as db:
            self._forget_expired(db, user_id)
            row = db.execute(
                "SELECT nonce, ciphertext, kid, hint, model, workspace_id, state FROM ai_keys "
                "WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            ).fetchone()
            if row is None:
                return None
            nonce, ciphertext, kid, shown, model, workspace, state = row
            key = self._open(user_id, provider, nonce, ciphertext, kid)
            if key is None:
                return Saved(provider, "", model, workspace, shown, UNREADABLE, nonce)
            if kid != self.kid:
                nonce, ciphertext, kid = self._seal(user_id, provider, key)
                db.execute(
                    "UPDATE ai_keys SET nonce = ?, ciphertext = ?, kid = ? WHERE user_id = ? AND provider = ?",
                    (nonce, ciphertext, kid, user_id, provider),
                )
            return Saved(provider, key, model, workspace, shown, state if state in (OK, REFUSED) else OK, nonce)

    def use(self, user_id: str) -> str:
        """What the person chose to answer with: "shared", or a provider."""
        with self._connect() as db:
            row = db.execute("SELECT use FROM ai_choice WHERE user_id = ?", (user_id,)).fetchone()
        return row[0] if row else SHARED

    def set_use(self, user_id: str, use: str, model: str | None = None):
        """Choose what answers the person's questions, and the model of that key if given."""
        with self._connect() as db:
            self._set_use(db, user_id, use)
            if model is not None and use != SHARED:
                db.execute("UPDATE ai_keys SET model = ? WHERE user_id = ? AND provider = ?", (model, user_id, use))

    def _set_use(self, db, user_id: str, use: str):
        if use == SHARED:
            db.execute("DELETE FROM ai_choice WHERE user_id = ?", (user_id,))
        else:
            db.execute("INSERT OR REPLACE INTO ai_choice (user_id, use) VALUES (?, ?)", (user_id, use))

    def touch(self, user_id: str, provider: str, nonce: bytes | None = None):
        """Note that the key answered a question, so its owner can see when it was last used.
        With nonce (Saved.nonce), only while that key is still the one saved, not one saved since."""
        self._update(user_id, provider, nonce, "last_used_at", int(self.clock()))

    def mark(self, user_id: str, provider: str, state: str, nonce: bytes | None = None):
        """Set the key's state; with nonce, as touch()."""
        self._update(user_id, provider, nonce, "state", state)

    def _update(self, user_id: str, provider: str, nonce: bytes | None, column: str, value):
        query, args = f"UPDATE ai_keys SET {column} = ? WHERE user_id = ? AND provider = ?", [value, user_id, provider]
        if nonce is not None:
            query += " AND nonce = ?"
            args.append(nonce)
        with self._connect() as db:
            db.execute(query, args)

    def delete(self, user_id: str, provider: str):
        """Forget the key; if it was the person's choice, they go back to the shared model."""
        with self._connect() as db:
            db.execute("DELETE FROM ai_keys WHERE user_id = ? AND provider = ?", (user_id, provider))
            db.execute("DELETE FROM ai_choice WHERE user_id = ? AND use = ?", (user_id, provider))

    def delete_all(self, user_id: str):
        with self._connect() as db:
            db.execute("DELETE FROM ai_keys WHERE user_id = ?", (user_id,))
            db.execute("DELETE FROM ai_choice WHERE user_id = ?", (user_id,))

    def sweep(self) -> int:
        """Forget every expired key; how many there were."""
        with self._connect() as db:
            return db.execute("DELETE FROM ai_keys WHERE expires_at <= ?", (int(self.clock()),)).rowcount


class _Connection:
    """A connection used for one transaction and closed after it, committed unless it failed."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def __enter__(self) -> sqlite3.Connection:
        return self.db

    def __exit__(self, kind, exc, tb):
        with closing(self.db):
            if kind is None:
                self.db.commit()
            else:
                self.db.rollback()
        return False
