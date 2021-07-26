import numpy as np

from envs.base import StochasticMDPEnv, STATE


def one_step_lookahead(
        env: StochasticMDPEnv,
        state: STATE,
        value_function: np.array,
        discount_factor: float
) -> np.array:

    n_actions = len(env.idx_action_map)
    action_values = np.zeros(n_actions)
    for action_idx in range(n_actions):
        for prob, next_state, reward in env.get_transitions(
                state=state,
                action=env.idx_action_map[action_idx]
        ):
            action_values[action_idx] += prob * (reward + discount_factor * value_function[next_state.value])
    return action_values
