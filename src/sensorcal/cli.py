"""SensorCal CLI entrypoint with subcommands."""

from __future__ import annotations

import argparse
import sys

from . import app as recalibrate_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="sensorcal")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Recalibrate subcommand passes through to app.main
    subparsers.add_parser("recalibrate", help="Run the LiDAR/camera recalibration UI")

    args, unknown = parser.parse_known_args()

    if args.command == "recalibrate":
        sys.argv = ["sensorcal recalibrate", *unknown]
        recalibrate_app.main()


if __name__ == "__main__":
    main()
