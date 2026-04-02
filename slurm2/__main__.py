from __future__ import annotations

import argparse

COMMAND_HELP = {
    "generate": "Generate experiment metadata, point metadata, and cfg files from an experiment spec",
    "scan": "Rebuild per-point status by inspecting HDF5 outputs on disk",
    "submit": "Plan pending seed chunks and submit them as Slurm job arrays",
    "worker": "Run one submitted Slurm array chunk and update per-seed runtime state",
}


# Dispatch the top-level `python -m slurm2 ...` command to a subcommand.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Entry point for the slurm2 workflow. Choose one subcommand such as generate, scan, "
            "submit, or worker, then pass the remaining command-specific arguments after it."
        )
    )
    parser.add_argument("command", choices=sorted(COMMAND_HELP), help="Workflow subcommand to run. Use one of: generate, scan, submit, or worker.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded verbatim to the chosen subcommand. You can insert '--' before them if needed.")
    parsed = parser.parse_args(argv)

    if parsed.args and parsed.args[0] == "--":
        parsed.args = parsed.args[1:]

    if parsed.command == "generate":
        from slurm2 import generate

        return generate.main(parsed.args)
    if parsed.command == "scan":
        from slurm2 import scan

        return scan.main(parsed.args)
    if parsed.command == "submit":
        from slurm2 import submit

        return submit.main(parsed.args)
    if parsed.command == "worker":
        from slurm2 import worker

        return worker.main(parsed.args)
    raise AssertionError(f"Unhandled command: {parsed.command}")


if __name__ == "__main__":
    raise SystemExit(main())
