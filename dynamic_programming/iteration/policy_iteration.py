from typing import Tuple

import numpy as np

from iteration.utils import one_step_lookahead
from envs.base import StochasticMDPEnv


def eval_policy(
        policy: np.array,
        env: StochasticMDPEnv,
        discount_factor: float = 1.0,
        theta: float = 0.0001
) -> np.array:

    n_states = len(env.idx_state_map)

    # Start with all zero value function
    value_function = np.zeros(n_states)
    while True:
        delta = 0
        for state_idx in range(n_states):
            v = 0
            for action_idx, action_prob in enumerate(policy[state_idx]):
                for prob, next_state, reward in env.get_transitions(
                        state=env.idx_state_map[state_idx],
                        action=env.idx_action_map[action_idx]
                ):
                    # Expected value
                    v += action_prob * prob * (reward + discount_factor * value_function[next_state.value])
            # How much the value function changed
            delta = max(delta, np.abs(v - value_function[state_idx]))
            value_function[state_idx] = v

        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(value_function)


def policy_iteration(
        env: StochasticMDPEnv,
        policy,
        theta: float = 0.0001,
        discount_factor: float = 1.0
) -> Tuple[np.array, np.array]:

    n_states = len(env.idx_state_map)
    n_actions = len(env.idx_action_map)

    while True:
        value_function = eval_policy(policy, env, discount_factor, theta)

        policy_stable = True
        for state_idx in range(n_states):
            # Best action  under the current policy
            action = np.argmax(policy[state_idx])
            # Best action by one-step ahead
            q_values = one_step_lookahead(
                env=env,
                state=env.idx_state_map[state_idx],
                value_function=value_function,
                discount_factor=discount_factor
            )

            best_action = np.argmax(q_values)

            # Greedly update the policy
            if action != best_action:
                policy_stable = False

            policy[state_idx] = np.eye(n_actions)[best_action]

        if policy_stable:
            return policy, value_function
