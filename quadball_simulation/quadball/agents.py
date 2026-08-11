"""Scripted baseline policies and agent interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from .entities import AgentAction, Role
from .physics import norm, unit


class AgentPolicy(ABC):
    """Abstract policy interface for one simulated player."""

    @abstractmethod
    def act(self, player_id: int, observation: dict) -> AgentAction:
        """Choose the next action.

        Parameters
        ----------
        player_id : int
            Controlled player identifier.
        observation : dict
            Read-only structured world observation.

        Returns
        -------
        AgentAction
            Desired action for the next simulation tick.
        """


class ScriptedPolicy(AgentPolicy):
    """Role-aware deterministic baseline policy.

    Notes
    -----
    The policy provides a meaningful opponent for learning experiments. It
    values spacing, passes under pressure, hoop defence, retrieval of loose
    bludgers, legal contact, and seeker pursuit.
    """

    def act(self, player_id: int, observation: dict) -> AgentAction:
        """Choose a rule-based action for one player.

        Parameters
        ----------
        player_id : int
            Controlled player identifier.
        observation : dict
            Environment observation produced by ``observe``.

        Returns
        -------
        AgentAction
            Role-appropriate action.
        """

        me = observation["players"][player_id]
        if not me["active"]:
            return AgentAction()
        teammates = [p for p in observation["players"].values() if p["team"] == me["team"] and p["id"] != player_id and p["active"]]
        opponents = [p for p in observation["players"].values() if p["team"] != me["team"] and p["active"]]
        attack_x = observation["field"]["width"] if me["team"] == 0 else 0.0
        own_x = 0.0 if me["team"] == 0 else observation["field"]["width"]
        pos = np.asarray(me["position"], dtype=float)

        if me["role"] == Role.SEEKER.name and observation["flag"]["active"]:
            return AgentAction(move=unit(np.asarray(observation["flag"]["position"]) - pos), sprint=True)

        if me["role"] == Role.BEATER.name:
            if me["held_bludger"] is not None and opponents:
                target = min(opponents, key=lambda p: norm(np.asarray(p["position"]) - pos))
                return AgentAction(move=unit(np.asarray(target["position"]) - pos), throw_bludger_target=target["id"])
            loose = [b for b in observation["balls"] if b["kind"] == "BLUDGER" and b["carrier_id"] is None]
            if loose:
                target = min(loose, key=lambda b: norm(np.asarray(b["position"]) - pos))
                return AgentAction(move=unit(np.asarray(target["position"]) - pos), sprint=True)

        if me["has_quaffle"]:
            pressure = [p for p in opponents if norm(np.asarray(p["position"]) - pos) < 3.0]
            distance_to_goal = abs(attack_x - pos[0])
            if distance_to_goal < 9.0:
                return AgentAction(move=np.array([1.0 if attack_x > pos[0] else -1.0, 0.0]), shoot=True)
            if pressure and teammates:
                ahead = sorted(teammates, key=lambda p: abs(attack_x - p["position"][0]))
                return AgentAction(pass_target=ahead[0]["id"], move=unit(np.array([attack_x, 18.0]) - pos))
            return AgentAction(move=unit(np.array([attack_x, 18.0]) - pos), sprint=me["stamina"] > 35)

        carrier = next((p for p in observation["players"].values() if p["has_quaffle"]), None)
        if carrier is not None and carrier["team"] != me["team"]:
            delta = np.asarray(carrier["position"]) - pos
            return AgentAction(move=unit(delta), sprint=True, tackle_target=carrier["id"] if norm(delta) < 1.7 else None)
        if me["role"] == Role.KEEPER.name:
            return AgentAction(move=unit(np.array([own_x + (3.5 if own_x == 0 else -3.5), 18.0]) - pos))
        quaffle = observation["balls"][0]
        if quaffle["carrier_id"] is None:
            return AgentAction(move=unit(np.asarray(quaffle["position"]) - pos), sprint=True)
        lane_y = 7.0 + (player_id % 3) * 11.0
        return AgentAction(move=unit(np.array([(attack_x + own_x) / 2.0, lane_y]) - pos))
