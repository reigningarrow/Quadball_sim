"""Deterministic quadball match environment."""

from __future__ import annotations

from collections.abc import Mapping
import math
import numpy as np

from .config import SimulationConfig
from .entities import AgentAction, Ball, BallKind, FlagRunner, Player, Role, Team
from .physics import clamp_position, norm, segment_point_distance, unit


class QuadballEnvironment:
    """Simulate a complete organisation-neutral quadball match.

    Parameters
    ----------
    config : SimulationConfig | None
        Configuration. Defaults are used when omitted.

    Notes
    -----
    Each team always fields three chasers, one keeper and two beaters. One
    seeker per team is added only when the neutral flag runner is enabled.
    Fouls and penalties are intentionally outside the model. Tackles are
    resolved from range, closing speed, facing and stamina.
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.players: dict[int, Player] = {}
        self.balls: list[Ball] = []
        self.flag = FlagRunner(np.zeros(2, dtype=float))
        self.time = 0.0
        self.score = {Team.BLUE: 0, Team.RED: 0}
        self.done = False
        self.events: list[dict] = []
        self.reset()

    def reset(self, seed: int | None = None) -> dict:
        """Reset the match and return its initial observation.

        Parameters
        ----------
        seed : int | None
            Optional replacement random seed.

        Returns
        -------
        dict
            Structured initial observation.
        """

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time, self.done = 0.0, False
        self.score = {Team.BLUE: 0, Team.RED: 0}
        self.events = []
        self.players = {}
        roles = [Role.CHASER] * 3 + [Role.KEEPER] + [Role.BEATER] * 2
        if self.config.flag.enabled:
            roles.append(Role.SEEKER)
        player_id = 0
        for team in Team:
            x = 10.0 if team == Team.BLUE else self.config.field.width - 10.0
            for index, role in enumerate(roles):
                y = 5.0 + index * (26.0 / max(1, len(roles) - 1))
                self.players[player_id] = Player(player_id, team, role, np.array([x, y]), stamina=self.config.player.stamina_capacity)
                player_id += 1
        centre = np.array([self.config.field.width / 2.0, self.config.field.height / 2.0])
        self.balls = [Ball(0, BallKind.QUAFFLE, centre.copy())]
        self.balls.extend([Ball(i + 1, BallKind.BLUDGER, centre + np.array([0.0, (-1) ** i * 3.0])) for i in range(2)])
        self.flag = FlagRunner(centre.copy())
        return self.observe()

    def observe(self) -> dict:
        """Return a serializable full-state observation.

        Returns
        -------
        dict
            Players, balls, flag state, scores, time and field dimensions.
        """

        return {
            "time": self.time,
            "score": {int(k): v for k, v in self.score.items()},
            "field": {"width": self.config.field.width, "height": self.config.field.height},
            "players": {pid: {"id": pid, "team": int(p.team), "role": p.role.name, "position": p.position.copy(), "velocity": p.velocity.copy(), "stamina": p.stamina, "active": p.active, "has_quaffle": p.has_quaffle, "held_bludger": p.held_bludger} for pid, p in self.players.items()},
            "balls": [{"id": b.ball_id, "kind": b.kind.name, "position": b.position.copy(), "velocity": b.velocity.copy(), "carrier_id": b.carrier_id} for b in self.balls],
            "flag": {"enabled": self.config.flag.enabled, "active": self.flag.active, "caught": self.flag.caught, "position": self.flag.position.copy()},
            "done": self.done,
        }

    def step(self, actions: Mapping[int, AgentAction]) -> tuple[dict, dict[int, float], bool, dict]:
        """Advance the match by one fixed simulation step.

        Parameters
        ----------
        actions : Mapping[int, AgentAction]
            At most one desired action per player.

        Returns
        -------
        observation : dict
            State after the step.
        rewards : dict[int, float]
            Dense role-aware learning rewards.
        terminated : bool
            Whether the match has ended.
        info : dict
            Events emitted during this step.
        """

        if self.done:
            raise RuntimeError("Cannot step a completed match; call reset().")
        start_events = len(self.events)
        rewards = {pid: -0.001 for pid in self.players}
        self._update_activity()
        self._move_players(actions)
        self._handle_actions(actions, rewards)
        self._move_balls(rewards)
        self._resolve_pickups(rewards)
        self._update_flag(rewards)
        self.time += self.config.dt
        self.done = self.done or self.time >= self.config.rules.match_seconds
        return self.observe(), rewards, self.done, {"events": self.events[start_events:]}

    def _move_players(self, actions: Mapping[int, AgentAction]) -> None:
        cfg, dt = self.config.player, self.config.dt
        for pid, player in self.players.items():
            action = actions.get(pid, AgentAction())
            if not player.active:
                continue
            desired = unit(np.asarray(action.move, dtype=float))
            intensity = 1.0 if action.sprint and player.stamina > 0.0 else 0.72
            stamina_factor = 0.58 + 0.42 * player.stamina / cfg.stamina_capacity
            target_velocity = desired * cfg.max_speed * intensity * stamina_factor
            acceleration = target_velocity - player.velocity
            acceleration = unit(acceleration) * min(norm(acceleration), cfg.acceleration * dt)
            player.velocity += acceleration
            if norm(desired) > 0:
                desired_heading = math.atan2(desired[1], desired[0])
                delta = (desired_heading - player.heading + math.pi) % (2 * math.pi) - math.pi
                player.heading += float(np.clip(delta, -cfg.turn_rate * dt, cfg.turn_rate * dt))
            player.position = clamp_position(player.position + player.velocity * dt, self.config.field.width, self.config.field.height)
            if action.sprint and norm(desired) > 0.2:
                player.stamina = max(0.0, player.stamina - cfg.sprint_drain * dt)
            elif norm(desired) < 0.6:
                player.stamina = min(cfg.stamina_capacity, player.stamina + cfg.recovery_rate * dt)

    def _handle_actions(self, actions: Mapping[int, AgentAction], rewards: dict[int, float]) -> None:
        for pid, action in actions.items():
            if pid not in self.players or not self.players[pid].active:
                continue
            player = self.players[pid]
            if action.pass_target is not None:
                self._pass(player, action.pass_target, rewards)
            if action.shoot:
                self._shoot(player)
            if action.tackle_target is not None:
                self._tackle(player, action.tackle_target, rewards)
            if action.throw_bludger_target is not None:
                self._throw_bludger(player, action.throw_bludger_target)

    def _pass(self, player: Player, target_id: int, rewards: dict[int, float]) -> None:
        if not player.has_quaffle or target_id not in self.players:
            return
        target = self.players[target_id]
        if target.team != player.team or not target.active:
            return
        ball = self.balls[0]
        player.has_quaffle = False
        ball.carrier_id = None
        ball.position = player.position.copy()
        ball.velocity = unit(target.position + target.velocity * 0.35 - player.position) * self.config.player.pass_speed
        ball.last_team = player.team
        rewards[player.player_id] += 0.08
        self.events.append({"type": "pass", "from": player.player_id, "to": target_id, "time": self.time})

    def _shoot(self, player: Player) -> None:
        if not player.has_quaffle:
            return
        attack_x = self.config.field.width - self.config.field.goal_x_margin if player.team == Team.BLUE else self.config.field.goal_x_margin
        target_y = min(self.config.field.hoop_y, key=lambda y: abs(y - player.position[1]))
        ball = self.balls[0]
        player.has_quaffle = False
        ball.carrier_id = None
        ball.position = player.position.copy()
        ball.velocity = unit(np.array([attack_x, target_y]) - player.position) * self.config.player.shot_speed
        ball.last_team = player.team
        self.events.append({"type": "shot", "player": player.player_id, "time": self.time})

    def _tackle(self, tackler: Player, target_id: int, rewards: dict[int, float]) -> None:
        if target_id not in self.players or self.time < tackler.tackle_ready_at:
            return
        target = self.players[target_id]
        delta = target.position - tackler.position
        if target.team == tackler.team or not target.active or norm(delta) > self.config.player.tackle_range:
            return
        facing = max(0.0, float(np.dot(unit(delta), np.array([math.cos(tackler.heading), math.sin(tackler.heading)]))))
        closing = max(0.0, float(np.dot(tackler.velocity - target.velocity, unit(delta))))
        stamina = tackler.stamina / self.config.player.stamina_capacity
        probability = float(np.clip(0.22 + 0.3 * facing + 0.04 * closing + 0.18 * stamina, 0.05, 0.92))
        tackler.tackle_ready_at = self.time + self.config.rules.tackle_cooldown
        if self.time >= target.possession_protected_until and self.rng.random() < probability:
            target.velocity *= 0.25
            if target.has_quaffle:
                target.has_quaffle = False
                quaffle = self.balls[0]
                quaffle.carrier_id = None
                quaffle.position = target.position.copy()
                quaffle.velocity = 2.5 * unit(target.position - tackler.position)
                rewards[tackler.player_id] += 0.25
            self.events.append({"type": "tackle", "by": tackler.player_id, "target": target_id, "time": self.time})

    def _throw_bludger(self, beater: Player, target_id: int) -> None:
        if beater.role != Role.BEATER or beater.held_bludger is None or target_id not in self.players:
            return
        target = self.players[target_id]
        if target.team == beater.team or not target.active:
            return
        ball = next(b for b in self.balls if b.ball_id == beater.held_bludger)
        beater.held_bludger = None
        ball.carrier_id = None
        ball.position = beater.position.copy()
        ball.velocity = unit(target.position + 0.2 * target.velocity - beater.position) * 15.0
        ball.last_team = beater.team
        self.events.append({"type": "bludger_throw", "by": beater.player_id, "target": target_id, "time": self.time})

    def _move_balls(self, rewards: dict[int, float]) -> None:
        dt = self.config.dt
        for ball in self.balls:
            previous = ball.position.copy()
            if ball.carrier_id is not None:
                ball.position = self.players[ball.carrier_id].position.copy()
                ball.velocity[:] = 0.0
                continue
            ball.position += ball.velocity * dt
            ball.velocity *= max(0.0, 1.0 - 0.5 * dt)
            if ball.kind == BallKind.QUAFFLE:
                self._check_goal(ball, previous, rewards)
            else:
                self._check_bludger_hits(ball, previous, rewards)
            if ball.position[0] < 0 or ball.position[0] > self.config.field.width:
                ball.velocity[0] *= -0.55
            if ball.position[1] < 0 or ball.position[1] > self.config.field.height:
                ball.velocity[1] *= -0.55
            ball.position = clamp_position(ball.position, self.config.field.width, self.config.field.height)

    def _check_goal(self, ball: Ball, previous: np.ndarray, rewards: dict[int, float]) -> None:
        if ball.last_team is None:
            return
        goal_x = self.config.field.width - self.config.field.goal_x_margin if ball.last_team == Team.BLUE else self.config.field.goal_x_margin
        crossed = (previous[0] - goal_x) * (ball.position[0] - goal_x) <= 0 and abs(ball.position[0] - previous[0]) > 1e-8
        if crossed and any(segment_point_distance(previous, ball.position, np.array([goal_x, y])) <= self.config.field.hoop_radius for y in self.config.field.hoop_y):
            self.score[ball.last_team] += self.config.rules.goal_points
            for player in self.players.values():
                rewards[player.player_id] += 1.0 if player.team == ball.last_team else -0.5
            self.events.append({"type": "goal", "team": int(ball.last_team), "time": self.time})
            self._reset_quaffle(ball)

    def _check_bludger_hits(self, ball: Ball, previous: np.ndarray, rewards: dict[int, float]) -> None:
        if norm(ball.velocity) < 3.0 or ball.last_team is None:
            return
        candidates = [p for p in self.players.values() if p.team != ball.last_team and p.active and segment_point_distance(previous, ball.position, p.position) < 0.7]
        if not candidates:
            return
        target = min(candidates, key=lambda p: norm(p.position - ball.position))
        target.active = False
        target.return_required = True
        target.knocked_out_until = self.time + self.config.rules.knockout_seconds
        self._drop_held_balls(target)
        ball.velocity[:] = 0.0
        team_beaters = [p for p in self.players.values() if p.team == ball.last_team and p.role == Role.BEATER]
        if team_beaters:
            rewards[min(team_beaters, key=lambda p: norm(p.position - target.position)).player_id] += 0.35
        self.events.append({"type": "knockout", "target": target.player_id, "time": self.time})

    def _resolve_pickups(self, rewards: dict[int, float]) -> None:
        for ball in self.balls:
            if ball.carrier_id is not None or norm(ball.velocity) > 4.0:
                continue
            eligible = [p for p in self.players.values() if p.active and norm(p.position - ball.position) < 0.85 and ((ball.kind == BallKind.QUAFFLE and p.role in (Role.CHASER, Role.KEEPER)) or (ball.kind == BallKind.BLUDGER and p.role == Role.BEATER and p.held_bludger is None))]
            if not eligible:
                continue
            player = min(eligible, key=lambda p: norm(p.position - ball.position))
            ball.carrier_id = player.player_id
            ball.velocity[:] = 0.0
            if ball.kind == BallKind.QUAFFLE:
                player.has_quaffle = True
                player.possession_protected_until = self.time + self.config.rules.possession_immunity
                rewards[player.player_id] += 0.03
            else:
                player.held_bludger = ball.ball_id

    def _update_activity(self) -> None:
        for player in self.players.values():
            if not player.active and self.time >= player.knocked_out_until:
                home_x = 0.0 if player.team == Team.BLUE else self.config.field.width
                delta = np.array([home_x, self.config.field.height / 2]) - player.position
                player.position += unit(delta) * self.config.player.max_speed * self.config.dt
                if norm(delta) < 1.2:
                    player.active, player.return_required = True, False
                    player.stamina = min(self.config.player.stamina_capacity, player.stamina + 15.0)

    def _update_flag(self, rewards: dict[int, float]) -> None:
        if not self.config.flag.enabled or self.flag.caught:
            return
        if self.time >= self.config.flag.release_time:
            self.flag.active = True
        if not self.flag.active:
            return
        seekers = [p for p in self.players.values() if p.role == Role.SEEKER and p.active]
        if seekers:
            nearest = min(seekers, key=lambda p: norm(p.position - self.flag.position))
            flee = unit(self.flag.position - nearest.position)
            centre = np.array([self.config.field.width / 2, self.config.field.height / 2])
            boundary_bias = unit(centre - self.flag.position) * 0.45
            jitter = self.rng.normal(0.0, 0.12, 2)
            self.flag.velocity = unit(flee + boundary_bias + jitter) * self.config.flag.speed
            self.flag.position = clamp_position(self.flag.position + self.flag.velocity * self.config.dt, self.config.field.width, self.config.field.height)
            catcher = next((p for p in seekers if norm(p.position - self.flag.position) <= self.config.flag.catch_radius), None)
            if catcher is not None:
                self.flag.caught = True
                self.flag.active = False
                self.score[catcher.team] += self.config.rules.flag_points
                rewards[catcher.player_id] += 2.0
                self.events.append({"type": "flag_catch", "team": int(catcher.team), "player": catcher.player_id, "time": self.time})
                self.done = self.config.rules.flag_ends_match

    def _drop_held_balls(self, player: Player) -> None:
        for ball in self.balls:
            if ball.carrier_id == player.player_id:
                ball.carrier_id = None
                ball.position = player.position.copy()
                ball.velocity = player.velocity * 0.35
        player.has_quaffle = False
        player.held_bludger = None

    def _reset_quaffle(self, ball: Ball) -> None:
        for player in self.players.values():
            player.has_quaffle = False
        ball.carrier_id = None
        ball.last_team = None
        ball.position = np.array([self.config.field.width / 2, self.config.field.height / 2])
        ball.velocity[:] = 0.0

    def substitute(self, outgoing_id: int) -> None:
        """Replace a player with a fresh same-role reserve template.

        Parameters
        ----------
        outgoing_id : int
            Identifier of the player to refresh.

        Raises
        ------
        RuntimeError
            If substitutions are disabled.
        KeyError
            If the player does not exist.
        """

        if not self.config.substitutions.enabled:
            raise RuntimeError("Substitutions are disabled in SimulationConfig.")
        player = self.players[outgoing_id]
        self._drop_held_balls(player)
        home_x = 1.0 if player.team == Team.BLUE else self.config.field.width - 1.0
        player.position = np.array([home_x, self.config.field.height / 2])
        player.velocity[:] = 0.0
        player.stamina = self.config.player.stamina_capacity
        player.active = True
        self.events.append({"type": "substitution", "player": outgoing_id, "time": self.time})
