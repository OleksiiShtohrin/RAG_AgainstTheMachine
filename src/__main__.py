"""Application main entrypoint executing Python Fire CLI."""

import fire
from src.cli import CLI


def main() -> None:
    """Run CLI application."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
