"""Generate a user API key without storing it anywhere."""

from __future__ import annotations

import secrets


def main() -> None:
    print(f"fn_{secrets.token_urlsafe(32)}")


if __name__ == "__main__":
    main()
