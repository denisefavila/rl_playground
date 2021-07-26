from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from envs.base import StochasticMDPEnv


class RobotStates(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class RobotActions(Enum):
    STAY = 0
    GO_RIGHT = 1
    GO_LEFT = 2


STATES_REWARDS = {
    RobotStates.LEFT: 1,
    RobotStates.CENTER: 0,
    RobotStates.RIGHT: 4,
}

ACTIONS_COST = {
    RobotActions.STAY: 0,
    RobotActions.GO_LEFT: 1,
    RobotActions.GO_RIGHT: 1,
}


@dataclass
class VigilantRobotEnv(StochasticMDPEnv):

    idx_state_map = {i.value: i for i in RobotStates}
    idx_action_map = {i.value: i for i in RobotActions}

    transition = {
        RobotStates.LEFT: {
            RobotActions.STAY: {
                RobotStates.LEFT: 1.0,
                RobotStates.CENTER: 0.0,
                RobotStates.RIGHT: 0.0,
            },
            RobotActions.GO_LEFT: {
                RobotStates.LEFT: 0.0,
                RobotStates.CENTER: 0.0,
                RobotStates.RIGHT: 0.0,
            },
            RobotActions.GO_RIGHT: {
                RobotStates.LEFT: 0.3,
                RobotStates.CENTER: 0.7,
                RobotStates.RIGHT: 0.0,
            },
        },
        RobotStates.CENTER: {
            RobotActions.STAY: {
                RobotStates.LEFT: 0.0,
                RobotStates.CENTER: 1.0,
                RobotStates.RIGHT: 0.0,
            },
            RobotActions.GO_LEFT: {
                RobotStates.LEFT: 0.7,
                RobotStates.CENTER: 0.3,
                RobotStates.RIGHT: 0.0,
            },
            RobotActions.GO_RIGHT: {
                RobotStates.LEFT: 0.0,
                RobotStates.CENTER: 0.3,
                RobotStates.RIGHT: 0.7,
            },
        },
        RobotStates.RIGHT: {
            RobotActions.STAY: {
                RobotStates.LEFT: 0.0,
                RobotStates.CENTER: 0.0,
                RobotStates.RIGHT: 1.0,
            },
            RobotActions.GO_LEFT: {
                RobotStates.LEFT: 0.0,
                RobotStates.CENTER: 0.7,
                RobotStates.RIGHT: 0.3,
            },
            RobotActions.GO_RIGHT: {
                RobotStates.LEFT: 0.0,
                RobotStates.CENTER: 0.0,
                RobotStates.RIGHT: 0.0,
            },
        },
    }

    def get_transitions(self, state: RobotStates, action: RobotActions) \
            -> List[Tuple[float, RobotStates, int]]:
        # [(prob, next_state, reward)]
        return [
            (prob, next_state, STATES_REWARDS[next_state] - ACTIONS_COST[action])
            for next_state, prob in self.transition[state][action].items()
        ]
