"""Entity and action definitions used by the simulation core."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


class Team(IntEnum):
    """Identify the two competing teams."""

    BLUE = 0
    RED = 1


class Role(Enum):
    """Identify a player's fixed match role."""

    CHASER = auto()
    KEEPER = auto()
    BEATER = auto()
    SEEKER = auto()


class BallKind(Enum):
    """Identify a ball's gameplay purpose."""

    QUAFFLE = auto()
    BLUDGER = auto()


@dataclass(slots=True)
class AgentAction:
    """Represent one player's desired action for a simulation tick.

    Parameters
    ----------
    move : numpy.ndarray
        Desired normalized movement direction.
    sprint : bool
        Request maximum-intensity movement.
    pass_target : int | None
        Player identifier receiving a pass, if any.
    shoot : bool
        Attempt a shot towards the best visible hoop.
    tackle_target : int | None
        Opponent identifier to tackle.
    throw_bludger_target : int | None
        Opponent identifier targeted by a beater.
    """

    move: Vector = field(default_factory=lambda: np.zeros(2, dtype=float))
    sprint: bool = False
    pass_target: int | None = None
    shoot: bool = False
    tackle_target: int | None = None
    throw_bludger_target: int | None = None


@dataclass(slots=True)
class Player:
    """Store mutable player state.

    Parameters
    ----------
    player_id : int
        Stable match identifier.
    team : Team
        Team membership.
    role : Role
        Fixed playing role.
    position : numpy.ndarray
        Current planar coordinates.
    velocity : numpy.ndarray
        Current planar velocity.
    heading : float
        Facing angle in radians.
    stamina : float
        Remaining stamina units.
    """

    player_id: int
    team: Team
    role: Role
    position: Vector
    velocity: Vector = field(default_factory=lambda: np.zeros(2, dtype=float))
    heading: float = 0.0
    stamina: float = 100.0
    active: bool = True
    knocked_out_until: float = 0.0
    return_required: bool = False
    has_quaffle: bool = False
    held_bludger: int | None = None
    tackle_ready_at: float = 0.0
    possession_protected_until: float = 0.0


@dataclass(slots=True)
class Ball:
    """Store mutable ball state.

    Parameters
    ----------
    ball_id : int
        Stable match identifier.
    kind : BallKind
        Gameplay purpose.
    position : numpy.ndarray
        Current planar coordinates.
    velocity : numpy.ndarray
        Current planar velocity.
    carrier_id : int | None
        Player currently carrying the ball.
    """

    ball_id: int
    kind: BallKind
    position: Vector
    velocity: Vector = field(default_factory=lambda: np.zeros(2, dtype=float))
    carrier_id: int | None = None
    last_team: Team | None = None


@dataclass(slots=True)
class FlagRunner:
    """Store the neutral flag runner state."""

    position: Vector
    velocity: Vector = field(default_factory=lambda: np.zeros(2, dtype=float))
    active: bool = False
    caught: bool = False
