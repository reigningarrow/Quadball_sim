"""Dependency-light learning helpers and optional Gymnasium adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .entities import AgentAction, Role


@dataclass
class RoleQLearner:
    """Tabular Q-learning policy shared by all players of one role.

    Parameters
    ----------
    role : Role
        Role whose experience is shared.
    learning_rate : float
        Temporal-difference update rate.
    discount : float
        Future reward discount factor.
    epsilon : float
        Exploration probability.
    seed : int
        Random seed.

    Notes
    -----
    The compact state bins distance to the relevant ball, possession, stamina,
    attacking half and pressure. Actions are eight movement directions plus
    idle. Higher-level pass, shoot and tackle decisions can be layered on by
    an experiment policy.
    """

    role: Role
    learning_rate: float = 0.12
    discount: float = 0.97
    epsilon: float = 0.15
    seed: int = 0
    q: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the learner-specific random generator."""
        self.rng = np.random.default_rng(self.seed)

    def encode(self, player: dict, observation: dict) -> tuple[int, ...]:
        """Encode an observation into a compact role-shared state.

        Parameters
        ----------
        player : dict
            Controlled player observation.
        observation : dict
            Full environment observation.

        Returns
        -------
        tuple[int, ...]
            Discrete state key.
        """

        pos = np.asarray(player["position"])
        ball = observation["balls"][0 if self.role != Role.BEATER else 1]
        distance_bin = min(5, int(np.linalg.norm(np.asarray(ball["position"]) - pos) // 5))
        stamina_bin = min(3, int(player["stamina"] // 25))
        attacking = int((player["team"] == 0 and pos[0] > observation["field"]["width"] / 2) or (player["team"] == 1 and pos[0] < observation["field"]["width"] / 2))
        return distance_bin, int(player["has_quaffle"]), stamina_bin, attacking

    def act(self, state: tuple[int, ...]) -> tuple[int, AgentAction]:
        """Select an epsilon-greedy movement action.

        Parameters
        ----------
        state : tuple[int, ...]
            Encoded state.

        Returns
        -------
        index : int
            Discrete action index.
        action : AgentAction
            Corresponding simulation action.
        """

        values = self.q.setdefault(state, np.zeros(9, dtype=float))
        index = int(self.rng.integers(9)) if self.rng.random() < self.epsilon else int(np.argmax(values))
        angle = index * np.pi / 4.0
        movement = np.zeros(2) if index == 8 else np.array([np.cos(angle), np.sin(angle)])
        return index, AgentAction(move=movement, sprint=index != 8)

    def update(self, state: tuple[int, ...], action: int, reward: float, next_state: tuple[int, ...], terminal: bool) -> None:
        """Apply one Q-learning update.

        Parameters
        ----------
        state : tuple[int, ...]
            Previous state key.
        action : int
            Chosen action index.
        reward : float
            Immediate reward.
        next_state : tuple[int, ...]
            Successor state key.
        terminal : bool
            Whether the episode ended.
        """

        current = self.q.setdefault(state, np.zeros(9, dtype=float))
        future = 0.0 if terminal else float(np.max(self.q.setdefault(next_state, np.zeros(9, dtype=float))))
        current[action] += self.learning_rate * (reward + self.discount * future - current[action])


class GymnasiumAdapter:
    """Minimal optional Gymnasium-style wrapper around one controlled player.

    Parameters
    ----------
    environment : object
        A :class:`quadball.environment.QuadballEnvironment` instance.
    player_id : int
        Player exposed to the external learner.
    opponent_policy : object
        Policy supplying actions for all other players.
    """

    def __init__(self, environment, player_id: int, opponent_policy) -> None:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError("Install the 'rl' extra to use GymnasiumAdapter.") from exc
        self.environment = environment
        self.player_id = player_id
        self.opponent_policy = opponent_policy
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the wrapped environment.

        Parameters
        ----------
        seed : int | None
            Optional random seed.

        Returns
        -------
        observation : numpy.ndarray
            Flattened controlled-player observation.
        info : dict
            Empty reset information.
        """

        observation = self.environment.reset(seed)
        return self._flatten(observation), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one Gymnasium-compatible step.

        Parameters
        ----------
        action : int
            Movement action in ``[0, 8]``.

        Returns
        -------
        observation : numpy.ndarray
            Flattened successor observation.
        reward : float
            Controlled-player reward.
        terminated : bool
            Match termination flag.
        truncated : bool
            Always false; time limits are match terminations.
        info : dict
            Simulation event information.
        """

        observation = self.environment.observe()
        actions = {pid: self.opponent_policy.act(pid, observation) for pid in self.environment.players if pid != self.player_id}
        angle = action * np.pi / 4.0
        move = np.zeros(2) if action == 8 else np.array([np.cos(angle), np.sin(angle)])
        actions[self.player_id] = AgentAction(move=move, sprint=action != 8)
        next_observation, rewards, done, info = self.environment.step(actions)
        return self._flatten(next_observation), rewards[self.player_id], done, False, info

    def _flatten(self, observation: dict) -> np.ndarray:
        """Flatten the controlled player's most relevant state."""
        player = observation["players"][self.player_id]
        ball = observation["balls"][0]
        return np.asarray([*player["position"], *player["velocity"], player["stamina"], float(player["has_quaffle"]), *ball["position"], *ball["velocity"]], dtype=np.float32)
