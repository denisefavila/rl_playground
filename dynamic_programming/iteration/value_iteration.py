import numpy as np

from iteration.utils import one_step_lookahead
from envs.base import StochasticMDPEnv


def value_iteration(env: StochasticMDPEnv, theta: float = 0.0001, discount_factor: float = 1.0) \
        -> np.array:

    n_states = len(env.idx_state_map)
    n_actions = len(env.idx_action_map)

    value_function = np.zeros(n_states)

    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for state_idx in range(n_states):
            # Do a one-step lookahead to find the best action
            q_values = one_step_lookahead(
                env=env,
                state=env.idx_state_map[state_idx],
                value_function=value_function,
                discount_factor=discount_factor
            )

            best_action_value = np.max(q_values)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - value_function[state_idx]))
            # Update the value function.
            value_function[state_idx] = best_action_value

        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([n_states, n_actions])
    for state_idx in range(n_states):
        # One step lookahead to find the best action for this state
        q_values = one_step_lookahead(
            env=env,
            state=env.idx_state_map[state_idx],
            value_function=value_function,
            discount_factor=discount_factor,
        )
        best_action = np.argmax(q_values)
        # Always take the best action
        policy[state_idx, best_action] = 1.0

    return policy, value_function
