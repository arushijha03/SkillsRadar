"""Safely set a single key in .env (paste in terminal — nothing is printed back)."""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")

ALLOWED_KEYS = {
    "OPENAI_API_KEY",
    "RAPIDAPI_KEY",
    "PINECONE_API_KEY",
    "LANGCHAIN_API_KEY",
    "ANTHROPIC_API_KEY",
}


def update_env(key: str, value: str) -> None:
    value = value.strip()
    if not value:
        raise ValueError("Value cannot be empty.")

    lines: list[str] = []
    if os.path.isfile(ENV_PATH):
        with open(ENV_PATH, encoding="utf-8") as f:
            lines = f.read().splitlines()

    pattern = re.compile(rf"^{re.escape(key)}=")
    replaced = False
    new_lines = []
    for line in lines:
        if pattern.match(line):
            new_lines.append(f"{key}={value}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        new_lines.append(f"{key}={value}")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/set_env_key.py RAPIDAPI_KEY")
        print("Allowed keys:", ", ".join(sorted(ALLOWED_KEYS)))
        sys.exit(1)

    key = sys.argv[1].strip()
    if key not in ALLOWED_KEYS:
        print(f"Unknown key: {key}")
        sys.exit(1)

    print(f"Paste your {key} value, then press Enter.")
    print("(Right-click or Ctrl+V to paste in this terminal. Input is hidden.)")
    value = __import__("getpass").getpass("> ")

    update_env(key, value)
    print(f"Saved {key} to .env — restart uvicorn if it is running.")


if __name__ == "__main__":
    main()
