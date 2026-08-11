"""Train a small shared chaser movement policy without external RL packages."""

from quadball import QuadballEnvironment, SimulationConfig
from quadball.agents import ScriptedPolicy
from quadball.entities import Role
from quadball.learning import RoleQLearner


def main(episodes: int = 10) -> None:
    """Run a compact role-shared Q-learning demonstration.

    Parameters
    ----------
    episodes : int
        Number of training matches.
    """

    environment = QuadballEnvironment(SimulationConfig())
    baseline = ScriptedPolicy()
    learner = RoleQLearner(Role.CHASER, seed=11)
    controlled = next(pid for pid, p in environment.players.items() if p.role == Role.CHASER)
    for episode in range(episodes):
        observation = environment.reset(seed=episode)
        while not environment.done:
            state = learner.encode(observation["players"][controlled], observation)
            action_index, action = learner.act(state)
            actions = {pid: baseline.act(pid, observation) for pid in environment.players}
            actions[controlled] = action
            next_observation, rewards, done, _ = environment.step(actions)
            next_state = learner.encode(next_observation["players"][controlled], next_observation)
            learner.update(state, action_index, rewards[controlled], next_state, done)
            observation = next_observation
        print(f"Episode {episode + 1}: {environment.score}")


if __name__ == "__main__":
    main()
