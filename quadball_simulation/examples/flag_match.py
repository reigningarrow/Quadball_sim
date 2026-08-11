"""Run a visual match with the neutral flag runner enabled."""

from quadball.agents import ScriptedPolicy
from quadball.cli import run_match
from quadball.config import FlagConfig, SimulationConfig
from quadball.environment import QuadballEnvironment


if __name__ == "__main__":
    environment = QuadballEnvironment(SimulationConfig(flag=FlagConfig(enabled=True)))
    print(run_match(environment, render=True))
