"""Core deterministic environment tests."""

import numpy as np
import pytest

from quadball.config import FlagConfig, SimulationConfig, SubstitutionConfig
from quadball.entities import AgentAction, Role
from quadball.environment import QuadballEnvironment


def test_default_roster_has_six_players_per_team() -> None:
    """Default teams contain exactly the requested six roles."""
    env = QuadballEnvironment()
    assert len(env.players) == 12
    for team in (0, 1):
        roles = [p.role for p in env.players.values() if int(p.team) == team]
        assert roles.count(Role.CHASER) == 3
        assert roles.count(Role.KEEPER) == 1
        assert roles.count(Role.BEATER) == 2
        assert roles.count(Role.SEEKER) == 0


def test_flag_adds_one_seeker_per_team() -> None:
    """Flag mode adds exactly one seeker to each team."""
    env = QuadballEnvironment(SimulationConfig(flag=FlagConfig(enabled=True)))
    assert len(env.players) == 14
    assert sum(p.role == Role.SEEKER for p in env.players.values()) == 2


def test_seeded_steps_are_deterministic() -> None:
    """Identically seeded simulations remain numerically identical."""
    a, b = QuadballEnvironment(), QuadballEnvironment()
    actions = {pid: AgentAction(move=np.array([1.0, 0.25]), sprint=True) for pid in a.players}
    for _ in range(10):
        oa, _, _, _ = a.step(actions)
        ob, _, _, _ = b.step(actions)
    for pid in a.players:
        np.testing.assert_allclose(oa["players"][pid]["position"], ob["players"][pid]["position"])


def test_substitution_is_opt_in() -> None:
    """Substitution calls fail safely unless enabled."""
    env = QuadballEnvironment()
    with pytest.raises(RuntimeError):
        env.substitute(0)
    enabled = QuadballEnvironment(SimulationConfig(substitutions=SubstitutionConfig(enabled=True)))
    enabled.players[0].stamina = 1.0
    enabled.substitute(0)
    assert enabled.players[0].stamina == enabled.config.player.stamina_capacity
