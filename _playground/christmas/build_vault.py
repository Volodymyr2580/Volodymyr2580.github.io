#!/usr/bin/env python3
"""
Encrypt private Christmas Markdown letters (private-christmas/letters/*.md)
and videos (private-christmas/videos/*) into playground/christmas/vault/
using zero-dependency Python stdlib + macOS /usr/bin/openssl (AES-256-CTR + HMAC-SHA256).

Identical video files (e.g. a shared piano video across letters) are encrypted ONCE
and shared securely so the vault stays compact.

Usage:
  python3 _playground/christmas/build_vault.py
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_DIR = ROOT / "private-christmas"
LETTERS_DIR = PRIVATE_DIR / "letters"
CONFIG_FILE = PRIVATE_DIR / "config.json"
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


def parse_names(raw_names: str) -> list[str]:
    text = raw_names.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = re.split(r"[,，、]+", text)
    cleaned: list[str] = []
    for part in parts:
        item = part.strip().strip('"').strip("'")
        if item:
            cleaned.append(item)
    return cleaned


def parse_markdown_letter(path: Path) -> dict[str, object]:
    content = path.read_text(encoding="utf-8").lstrip("\ufeff")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if not match:
        raise ValueError(f"{path.name}: missing frontmatter block (--- ... ---).")

    fm_text, body = match.group(1), match.group(2)
    meta: dict[str, str] = {}
    for line in fm_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, val = stripped.split(":", 1)
        meta[key.strip()] = val.strip().strip('"').strip("'")

    # Strip HTML comments so template helper notes never enter the letter
    clean_body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    paragraphs = [line.strip() for line in clean_body.splitlines() if line.strip()]

    enabled_str = meta.get("enabled", "false").lower()
    enabled = enabled_str in {"true", "yes", "1", "on"}

    return {
        "id": path.stem,
        "enabled": enabled,
        "names": parse_names(meta.get("names", "")),
        "code": meta.get("code", ""),
        "envelopeRecipient": meta.get("envelope_recipient", ""),
        "greeting": meta.get("greeting", ""),
        "paragraphs": paragraphs,
        "signaturePrefix": meta.get("signature_prefix", ""),
        "signatureName": meta.get("signature_name", ""),
        "date": meta.get("date", ""),
        "video": meta.get("video", "default"),
        "videoCaption": meta.get("video_caption", ""),
    }


def main() -> int:
    if not LETTERS_DIR.is_dir():
        print(f"Error: {LETTERS_DIR} not found.", file=sys.stderr)
        return 1

    config: dict[str, str] = {}
    if CONFIG_FILE.is_file():
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    default_video_src = str(config.get("defaultVideo", "")).strip()
    default_video_caption = str(config.get("defaultVideoCaption", "")).strip()

    md_files = sorted(LETTERS_DIR.glob("*.md"))
    if not md_files:
        print(f"Warning: no .md files found in {LETTERS_DIR}.")

    # Recreate vault cleanly so removed or disabled drafts never stay published
    if VAULT_DIR.exists():
        shutil.rmtree(VAULT_DIR)
    VAULT_DIR.mkdir(parents=True, exist_ok=True)

    salt = secrets.token_bytes(16)
    slots: list[dict[str, str]] = []
    seen_tags: set[str] = set()
    encrypted_videos: dict[Path, dict[str, str]] = {}
    published_count = 0
    skipped: list[str] = []
    warned_missing_default_video = False

    for md_path in md_files:
        item = parse_markdown_letter(md_path)
        item_id = str(item["id"])

        if not item["enabled"]:
            skipped.append(f"{item_id} (enabled=false)")
            continue

        norm_code = normalize_code(str(item["code"]))
        if not norm_code or norm_code.startswith("TODO"):
            skipped.append(f"{item_id} (code still placeholder)")
            continue

        names = [n for n in item["names"] if normalize_name(n)]  # type: ignore[arg-type]
        if not names:
            skipped.append(f"{item_id} (no valid names)")
            continue

        paragraphs = item["paragraphs"]
        if not paragraphs:
            skipped.append(f"{item_id} (empty body)")
            continue

        video_field = str(item["video"]).strip()
        video_caption = str(item["videoCaption"]).strip()
        is_default_video = video_field.lower() in {"", "default"}
        is_no_video = video_field.lower() in {"none", "null", "false", "no"}

        video_entry = None
        if not is_no_video:
            resolved_src = default_video_src if is_default_video else video_field
            final_caption = video_caption or (default_video_caption if is_default_video else "")
            if resolved_src:
                video_path = (PRIVATE_DIR / resolved_src).resolve()
                if not video_path.is_file():
                    if is_default_video:
                        if not warned_missing_default_video:
                            print(
                                f"Info: default video '{resolved_src}' not placed yet; "
                                "encrypting text-only for now."
                            )
                            warned_missing_default_video = True
                    else:
                        print(f"Error [{item_id}]: custom video not found -> {video_path}", file=sys.stderr)
                        return 1
                else:
                    if video_path not in encrypted_videos:
                        size_mb = video_path.stat().st_size / (1024 * 1024)
                        if size_mb > MAX_RECOMMENDED_VIDEO_MB:
                            print(
                                f"Warning [{item_id}]: {resolved_src} is {size_mb:.1f} MB "
                                f"(recommended <= {MAX_RECOMMENDED_VIDEO_MB} MB for fast mobile playback)."
                            )
                        ext = video_path.suffix.lower()
                        mime_type = "video/webm" if ext == ".webm" else "video/mp4"
                        v_enc = secrets.token_bytes(32)
                        v_mac = secrets.token_bytes(32)
                        encrypted_video = encrypt_blob(v_enc, v_mac, video_path.read_bytes())
                        v_filename = f"v_{secrets.token_hex(8)}.vbin"
                        (VAULT_DIR / v_filename).write_bytes(encrypted_video)
                        encrypted_videos[video_path] = {
                            "vaultFile": f"vault/{v_filename}",
                            "mimeType": mime_type,
                            "keyB64": base64.b64encode(v_enc + v_mac).decode("ascii"),
                        }
                    shared_v = encrypted_videos[video_path]
                    video_entry = {
                        **shared_v,
                        "caption": final_caption,
                    }

        letter_payload: dict[str, object] = {
            "envelopeRecipient": item["envelopeRecipient"] or f"To: {names[0]}",
            "greeting": item["greeting"] or f"Dear {names[0]},",
            "paragraphs": paragraphs,
            "signaturePrefix": item["signaturePrefix"],
            "signatureName": item["signatureName"] or "文哲",
            "date": item["date"] or "2026 年圣诞",
            "video": video_entry,
        }

        l_enc = secrets.token_bytes(32)
        l_mac = secrets.token_bytes(32)
        letter_bytes = json.dumps(letter_payload, ensure_ascii=False).encode("utf-8")
        encrypted_letter = encrypt_blob(l_enc, l_mac, letter_bytes)
        l_filename = f"l_{secrets.token_hex(8)}.lbin"
        (VAULT_DIR / l_filename).write_bytes(encrypted_letter)

        slot_payload = json.dumps(
            {
                "letterFile": f"vault/{l_filename}",
                "letterKeyB64": base64.b64encode(l_enc + l_mac).decode("ascii"),
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

    print(
        f"Vault built at {VAULT_DIR}: {published_count} letter(s) published "
        f"({len(slots)} alias slot(s), {len(encrypted_videos)} unique video(s))."
    )
    if skipped:
        print("Skipped drafts:")
        for line in skipped:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
