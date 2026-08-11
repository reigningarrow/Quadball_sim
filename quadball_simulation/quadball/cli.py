"""Command-line entry point for matches and headless training."""

from __future__ import annotations

import argparse

from .agents import ScriptedPolicy
from .config import FlagConfig, RulesConfig, SimulationConfig
from .environment import QuadballEnvironment


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description="Run a deterministic 2D quadball simulation.")
    parser.add_argument("--render", action="store_true", help="Open the optional Pygame renderer.")
    parser.add_argument("--flag-runner", action="store_true", help="Enable the neutral flag runner and seekers.")
    parser.add_argument("--matches", type=int, default=1, help="Number of headless matches to run.")
    parser.add_argument("--seed", type=int, default=7, help="Base deterministic seed.")
    parser.add_argument("--no-flag-end", action="store_true", help="Do not end the match after a flag catch.")
    return parser


def run_match(environment: QuadballEnvironment, render: bool = False) -> dict:
    """Run one scripted baseline match.

    Parameters
    ----------
    environment : QuadballEnvironment
        Match environment.
    render : bool
        Display the optional Pygame renderer.

    Returns
    -------
    dict
        Final score and elapsed time.
    """

    policy = ScriptedPolicy()
    renderer = None
    if render:
        from .renderer import PygameRenderer
        renderer = PygameRenderer(environment)
    while not environment.done:
        observation = environment.observe()
        actions = {pid: policy.act(pid, observation) for pid in environment.players}
        environment.step(actions)
        if renderer is not None and not renderer.draw():
            break
    if renderer is not None:
        renderer.close()
    return {"blue": environment.score[0], "red": environment.score[1], "seconds": environment.time}


def main() -> None:
    """Parse arguments and execute one or more matches."""
    args = build_parser().parse_args()
    config = SimulationConfig(seed=args.seed, flag=FlagConfig(enabled=args.flag_runner), rules=RulesConfig(flag_ends_match=not args.no_flag_end))
    for index in range(args.matches):
        environment = QuadballEnvironment(config)
        environment.reset(seed=args.seed + index)
        result = run_match(environment, render=args.render and index == 0)
        print(f"Match {index + 1}: Blue {result['blue']} - {result['red']} Red ({result['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
