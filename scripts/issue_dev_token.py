#!/usr/bin/env python3
"""Mint a dev JWT for testing the API. Usage: python scripts/issue_dev_token.py [user_id]"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.security import create_access_token  # noqa: E402

if __name__ == "__main__":
    user_id = sys.argv[1] if len(sys.argv) > 1 else "user-dev-1"
    print(create_access_token(user_id, {"email": f"{user_id}@example.com"}))
