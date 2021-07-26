from envs.vigilant_robot import VigilantRobotEnv, RobotStates, RobotActions
from iteration.policy_iteration import policy_iteration

import numpy as np


def main():

    vigilant_robot_env = VigilantRobotEnv()

    n_states = len(vigilant_robot_env.idx_state_map)
    n_actions = len(vigilant_robot_env.idx_action_map)

    initial_policy = np.zeros([n_states, n_actions])
    initial_policy[RobotStates.LEFT.value][RobotActions.STAY.value] = 1
    initial_policy[RobotStates.CENTER.value][RobotActions.GO_RIGHT.value] = 1
    initial_policy[RobotStates.RIGHT.value][RobotActions.STAY.value] = 1

    policy, value_function = policy_iteration(
        env=vigilant_robot_env,
        policy=initial_policy,
        discount_factor=0.9
    )

    print(f'Value Function: {value_function}')
    print(f'Policy: {policy}')


if __name__ == "__main__":

    main()
