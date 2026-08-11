# Quadball Learning Simulation

A documented, deterministic 2D quadball simulation for reinforcement-learning research and visual experimentation. It uses an organisation-neutral competitive rules approximation rather than claiming conformance with any governing body's rules.

## Included gameplay

- Fixed active roster per team: **3 chasers, 1 keeper and 2 beaters**.
- Optional **1 seeker per team**, added only when the neutral flag runner is enabled.
- Neutral autonomous flag runner that actively evades the nearest seeker.
- Quaffle possession, loose-ball collection, passing, shooting, hoop scoring and interceptions through loose-ball pickups.
- Range-, angle-, speed- and stamina-sensitive tackles; successful possession tackles force a drop.
- Two loose bludgers, beater retrieval, aimed throws, knockouts, forced ball drops and return-to-hoops re-entry.
- Moderate acceleration, momentum, turning rates, drag, projectile travel, stamina use and recovery.
- No foul, penalty or injury system.
- Optional substitutions, disabled by default and exposed programmatically.
- Scripted role-aware baseline agents and a NumPy-only shared-role Q-learning example.
- Optional Gymnasium adapter and optional Pygame visualisation.
- Deterministic fixed-step headless mode suitable for running many independent matches.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
pip install -e .[all]
```

For headless use with only NumPy:

```bash
pip install -e .
```

## Quick start

Run a fast headless match:

```bash
quadball --matches 1
```

Open the Pygame renderer:

```bash
quadball --render
```

Enable the neutral flag runner and both seekers:

```bash
quadball --render --flag-runner
```

Keep playing after a flag catch:

```bash
quadball --flag-runner --no-flag-end
```

Run several deterministic headless matches:

```bash
quadball --matches 20 --seed 100
```

Run the NumPy-only Q-learning example:

```bash
python examples/train_role_q.py
```

## Environment API

```python
from quadball import QuadballEnvironment, SimulationConfig
from quadball.agents import ScriptedPolicy

simulation = QuadballEnvironment(SimulationConfig())
policy = ScriptedPolicy()

while not simulation.done:
    observation = simulation.observe()
    actions = {player_id: policy.act(player_id, observation)
               for player_id in simulation.players}
    observation, rewards, terminated, info = simulation.step(actions)
```

`step` returns a full structured observation, a reward dictionary keyed by player ID, a termination flag and newly emitted match events. The state and renderer are separate, so headless execution does not import Pygame.

## Flag runner design

Flag mode is **off by default**. When enabled, one seeker is added to each team. After `FlagConfig.release_time`, a neutral runner enters play, flees the nearest seeker, receives a soft field-centre bias and uses seeded movement variation. A catch awards `RulesConfig.flag_points` and ends the match by default. Set `RulesConfig.flag_ends_match=False` to continue.

## Learning design

`RoleQLearner` is intentionally small and inspectable. All players of a role can share a Q-table, encouraging reusable behaviour and faster learning. Its compact state includes ball distance, possession, stamina and field progression. Dense rewards cover time cost, collection, passing, tackles, bludger knockouts, scoring and flag catches.

For advanced experiments, `GymnasiumAdapter` exposes one controlled player with a discrete movement action space; scripted agents fill the remaining roles. Researchers can instead call the native multi-agent `step` method to train centralized critics, role-shared policies or independent agents.

### Recommended experiment progression

1. Validate scripted-vs-scripted balance across many seeds.
2. Train one chaser movement policy while scripted agents control all other players.
3. Add learned high-level pass, shot and tackle actions.
4. Share parameters among same-role agents.
5. Train beaters, then enable seekers and the flag runner.
6. Evaluate against held-out seeds and scripted baselines.

## Configuration

The dataclasses in `quadball/config.py` configure field geometry, scoring, timings, movement, stamina, flag behaviour and substitutions. The complete simulation is reproducible when the same configuration, seed and action sequence are used.

Optional substitution support is deliberately conservative: enabling `SubstitutionConfig.enabled` allows `environment.substitute(player_id)` to replace a player with a fresh same-role reserve template. There are no automatic substitutions or roster-size changes.

## Project structure

```text
quadball/
  agents.py       scripted baseline policy and policy interface
  cli.py          command-line runner
  config.py       documented configuration dataclasses
  entities.py     players, balls, roles and actions
  environment.py  rules, contacts, scoring and match lifecycle
  learning.py     role-shared Q-learning and Gymnasium adapter
  physics.py      numerical movement/contact helpers
  renderer.py     optional Pygame renderer
examples/          training and flag-match examples
tests/             deterministic unit tests
```

## Tests

```bash
pytest
```

## Extension points

- Implement `AgentPolicy.act` for custom policies.
- Add richer observations without changing physics.
- Wrap multiple controlled players for PettingZoo-style parallel training.
- Replace the Q-table with a neural policy through Gymnasium.
- Tune contact, stamina and reward parameters through dataclass configuration.

## Scope and limitations

This is a realistic **simulation abstraction**, not a biomechanics engine or an official rules implementation. Contact is non-injury-producing, foul-free and probabilistic. Vertical ball flight and hoop height are projected into 2D effective scoring circles. The neutral flag runner is an artificial game agent rather than a human participant.
