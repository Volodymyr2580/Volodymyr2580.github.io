#!/usr/bin/env python3
"""
Encrypt private Christmas letters and videos into playground/christmas/vault/
using zero-dependency Python stdlib + macOS /usr/bin/openssl (AES-256-CTR + HMAC-SHA256).

Usage:
  python3 _playground/christmas/build_vault.py
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_DIR = ROOT / "private-christmas"
LETTERS_FILE = PRIVATE_DIR / "letters.json"
VAULT_DIR = ROOT / "playground" / "christmas" / "vault"

PBKDF2_ITERATIONS = 120_000
MAX_RECOMMENDED_VIDEO_MB = 25


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").strip().lower()
    return "".join(text.split())


def normalize_code(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").strip().upper()
    return "".join(text.split())


def aes_ctr_xor(enc_key: bytes, iv: bytes, data: bytes) -> bytes:
    proc = subprocess.run(
        [
            "openssl",
            "enc",
            "-aes-256-ctr",
            "-K",
            enc_key.hex(),
            "-iv",
            iv.hex(),
        ],
        input=data,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def encrypt_blob(enc_key: bytes, mac_key: bytes, plaintext: bytes) -> bytes:
    """Return IV (16B: 8B random nonce + 8B zero counter) || Ciphertext || HMAC-SHA256 (32B)."""
    iv = secrets.token_bytes(8) + b"\x00" * 8
    ciphertext = aes_ctr_xor(enc_key, iv, plaintext)
    mac = hmac.new(mac_key, iv + ciphertext, hashlib.sha256).digest()
    return iv + ciphertext + mac


def derive_slot_keys(norm_name: str, norm_code: str, salt: bytes) -> tuple[str, bytes, bytes]:
    secret_input = f"{norm_name}\0{norm_code}".encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", secret_input, salt, PBKDF2_ITERATIONS, dklen=96)
    tag = hashlib.sha256(dk[:32]).hexdigest()[:32]
    return tag, dk[32:64], dk[64:96]


def main() -> int:
    if not LETTERS_FILE.exists():
        print(f"Error: {LETTERS_FILE} not found.", file=sys.stderr)
        return 1

    raw = json.loads(LETTERS_FILE.read_text(encoding="utf-8"))
    entries = raw.get("letters", [])
    if not isinstance(entries, list):
        print("Error: letters.json must contain a 'letters' list.", file=sys.stderr)
        return 1

    # Recreate vault cleanly so removed or disabled drafts never stay published
    if VAULT_DIR.exists():
        shutil.rmtree(VAULT_DIR)
    VAULT_DIR.mkdir(parents=True, exist_ok=True)

    salt = secrets.token_bytes(16)
    slots: list[dict[str, str]] = []
    seen_tags: set[str] = set()
    published_count = 0
    skipped: list[str] = []

    for item in entries:
        item_id = item.get("id") or "unnamed"
        if not item.get("enabled", False):
            skipped.append(f"{item_id} (enabled=false)")
            continue

        code = item.get("code", "")
        norm_code = normalize_code(code)
        if not norm_code or norm_code.startswith("TODO"):
            skipped.append(f"{item_id} (code still placeholder)")
            continue

        names = [n for n in item.get("names", []) if normalize_name(n)]
        if not names:
            skipped.append(f"{item_id} (no valid names)")
            continue

        paragraphs = [p for p in item.get("paragraphs", []) if str(p).strip()]
        if not paragraphs:
            skipped.append(f"{item_id} (empty paragraphs)")
            continue

        letter_payload: dict[str, object] = {
            "envelopeRecipient": item.get("envelopeRecipient") or f"To: {names[0]}",
            "greeting": item.get("greeting") or f"Dear {names[0]},",
            "paragraphs": paragraphs,
            "signaturePrefix": item.get("signaturePrefix") or "Warm wishes,",
            "signatureName": item.get("signatureName") or "Wenzhe",
            "date": item.get("date") or "Christmas 2026",
            "video": None,
        }

        video_cfg = item.get("video") or {}
        video_src = str(video_cfg.get("src", "")).strip() if isinstance(video_cfg, dict) else ""
        if video_src:
            video_path = (PRIVATE_DIR / video_src).resolve()
            if not video_path.is_file():
                print(f"Error [{item_id}]: video file not found -> {video_path}", file=sys.stderr)
                return 1
            size_mb = video_path.stat().st_size / (1024 * 1024)
            if size_mb > MAX_RECOMMENDED_VIDEO_MB:
                print(
                    f"Warning [{item_id}]: {video_src} is {size_mb:.1f} MB "
                    f"(recommended <= {MAX_RECOMMENDED_VIDEO_MB} MB for fast mobile playback)."
                )
            ext = video_path.suffix.lower()
            mime_type = "video/webm" if ext == ".webm" else "video/mp4"
            video_enc_key = secrets.token_bytes(32)
            video_mac_key = secrets.token_bytes(32)
            encrypted_video = encrypt_blob(video_enc_key, video_mac_key, video_path.read_bytes())
            video_filename = f"v_{secrets.token_hex(8)}.vbin"
            (VAULT_DIR / video_filename).write_bytes(encrypted_video)
            letter_payload["video"] = {
                "vaultFile": f"vault/{video_filename}",
                "mimeType": mime_type,
                "keyB64": base64.b64encode(video_enc_key + video_mac_key).decode("ascii"),
                "caption": str(video_cfg.get("caption", "")).strip(),
            }

        letter_enc_key = secrets.token_bytes(32)
        letter_mac_key = secrets.token_bytes(32)
        letter_bytes = json.dumps(letter_payload, ensure_ascii=False).encode("utf-8")
        encrypted_letter = encrypt_blob(letter_enc_key, letter_mac_key, letter_bytes)
        letter_filename = f"l_{secrets.token_hex(8)}.lbin"
        (VAULT_DIR / letter_filename).write_bytes(encrypted_letter)

        slot_payload = json.dumps(
            {
                "letterFile": f"vault/{letter_filename}",
                "letterKeyB64": base64.b64encode(letter_enc_key + letter_mac_key).decode("ascii"),
            },
            ensure_ascii=False,
        ).encode("utf-8")

        for alias in names:
            norm_name = normalize_name(alias)
            tag, slot_enc_key, slot_mac_key = derive_slot_keys(norm_name, norm_code, salt)
            if tag in seen_tags:
                print(
                    f"Error: duplicate (name, code) combination detected for '{alias}' in [{item_id}].",
                    file=sys.stderr,
                )
                return 1
            seen_tags.add(tag)
            encrypted_slot = encrypt_blob(slot_enc_key, slot_mac_key, slot_payload)
            slots.append(
                {
                    "tag": tag,
                    "payload": base64.b64encode(encrypted_slot).decode("ascii"),
                }
            )

        published_count += 1

    # Shuffle slots so order does not reveal anything
    secrets.SystemRandom().shuffle(slots)

    index_data = {
        "version": 1,
        "kdf": "PBKDF2-SHA256",
        "iterations": PBKDF2_ITERATIONS,
        "saltB64": base64.b64encode(salt).decode("ascii"),
        "slots": slots,
    }
    (VAULT_DIR / "index.json").write_text(
        json.dumps(index_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Vault built at {VAULT_DIR}: {published_count} letter(s) published ({len(slots)} name alias slot(s)).")
    if skipped:
        print("Skipped drafts:")
        for line in skipped:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
