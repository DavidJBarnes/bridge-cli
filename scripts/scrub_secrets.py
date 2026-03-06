#!/usr/bin/env python3
"""Scrub secret-like patterns from JSONL dataset files.

Removes any line containing strings that match common secret patterns
(API keys, OAuth credentials, private keys, tokens) to prevent GitHub
push protection from blocking commits.
"""

import re
import sys

SECRET_PATTERNS = [
    r"GOCSPX-[A-Za-z0-9_-]+",
    r"client_secret",
    r"GOOGLE_CLIENT",
    r"pat[A-Za-z0-9]{14}\.[A-Za-z0-9]+",
    r"[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com",
    r"AIza[A-Za-z0-9_-]{35}",
    r"sk-[A-Za-z0-9]{20,}",
    r"ghp_[A-Za-z0-9]{36}",
    r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
    r"airtable.*pat[A-Za-z0-9]",
    r"AIRTABLE_API_KEY",
]

COMBINED = re.compile("|".join(SECRET_PATTERNS), re.IGNORECASE)


def scrub_file(filepath):
    """Remove lines matching secret patterns from a file."""
    with open(filepath) as f:
        lines = f.readlines()

    clean = [line for line in lines if not COMBINED.search(line)]
    removed = len(lines) - len(clean)

    with open(filepath, "w") as f:
        f.writelines(clean)

    print(f"{filepath}: {len(lines)} -> {len(clean)} ({removed} removed)")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        scrub_file(path)
