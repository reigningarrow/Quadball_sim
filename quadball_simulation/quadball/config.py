"""Configuration objects for the simulation."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field


@dataclass(slots=True)
class FieldConfig:
    """Describe the playing area and scoring geometry.

    Parameters
    ----------
    width : float
        Field width in simulation metres.
    height : float
        Field height in simulation metres.
    goal_x_margin : float
        Distance between each goal line and its three hoops.
    hoop_y : tuple[float, float, float]
        Vertical hoop-centre coordinates.
    hoop_radius : float
        Effective scoring radius of every hoop.
    """

    width: float = 60.0
    height: float = 36.0
    goal_x_margin: float = 3.0
    hoop_y: tuple[float, float, float] = (12.0, 18.0, 24.0)
    hoop_radius: float = 1.35


@dataclass(slots=True)
class RulesConfig:
    """Configure the organisation-neutral match rules.

    Parameters
    ----------
    goal_points : int
        Points awarded for a quaffle goal.
    flag_points : int
        Points awarded for a successful flag catch.
    flag_ends_match : bool
        Whether catching the flag immediately ends the match.
    match_seconds : float
        Maximum simulated match duration.
    knockout_seconds : float
        Minimum inactive period after a bludger hit.
    tackle_cooldown : float
        Seconds before a player may attempt another tackle.
    possession_immunity : float
        Anti-chain-contact grace period after gaining possession.
    """

    goal_points: int = 10
    flag_points: int = 30
    flag_ends_match: bool = True
    match_seconds: float = 420.0
    knockout_seconds: float = 2.0
    tackle_cooldown: float = 1.2
    possession_immunity: float = 0.35


@dataclass(slots=True)
class FlagConfig:
    """Configure the optional neutral flag runner.

    Parameters
    ----------
    enabled : bool
        Enable one neutral evading runner and one seeker per team.
    speed : float
        Maximum runner speed.
    catch_radius : float
        Distance at which an eligible seeker can catch the flag.
    release_time : float
        Match time at which the runner enters play.
    """

    enabled: bool = False
    speed: float = 7.2
    catch_radius: float = 1.15
    release_time: float = 45.0


@dataclass(slots=True)
class PlayerConfig:
    """Configure movement, stamina and interaction parameters.

    Parameters
    ----------
    max_speed : float
        Maximum speed at full stamina.
    acceleration : float
        Maximum planar acceleration.
    turn_rate : float
        Maximum angular velocity in radians per second.
    stamina_capacity : float
        Maximum stamina units.
    sprint_drain : float
        Stamina drained per second while moving intensely.
    recovery_rate : float
        Stamina restored per second while moving gently.
    tackle_range : float
        Maximum tackle initiation range.
    pass_speed : float
        Initial speed of a pass.
    shot_speed : float
        Initial speed of a shot.
    """

    max_speed: float = 7.0
    acceleration: float = 15.0
    turn_rate: float = 5.2
    stamina_capacity: float = 100.0
    sprint_drain: float = 9.0
    recovery_rate: float = 7.0
    tackle_range: float = 1.6
    pass_speed: float = 13.0
    shot_speed: float = 16.0


@dataclass(slots=True)
class SubstitutionConfig:
    """Describe optional substitution support.

    Parameters
    ----------
    enabled : bool
        Permit programmatic replacements during stoppages.
    bench_per_role : int
        Number of reserve templates maintained per role.

    Notes
    -----
    The default match has no substitutions. Enabling this feature exposes the
    :meth:`quadball.environment.QuadballEnvironment.substitute` API; automated
    substitution strategy is deliberately left to agents or experiment code.
    """

    enabled: bool = False
    bench_per_role: int = 1


@dataclass(slots=True)
class SimulationConfig:
    """Top-level immutable-by-convention simulation configuration.

    Parameters
    ----------
    dt : float
        Fixed simulation time step in seconds.
    seed : int
        Random seed used for deterministic resets.
    field : FieldConfig
        Field geometry.
    rules : RulesConfig
        Match rules.
    flag : FlagConfig
        Optional neutral runner settings.
    player : PlayerConfig
        Shared player movement settings.
    substitutions : SubstitutionConfig
        Optional substitution settings.
    """

    dt: float = 0.05
    seed: int = 7
    field: FieldConfig = dataclass_field(default_factory=FieldConfig)
    rules: RulesConfig = dataclass_field(default_factory=RulesConfig)
    flag: FlagConfig = dataclass_field(default_factory=FlagConfig)
    player: PlayerConfig = dataclass_field(default_factory=PlayerConfig)
    substitutions: SubstitutionConfig = dataclass_field(default_factory=SubstitutionConfig)
