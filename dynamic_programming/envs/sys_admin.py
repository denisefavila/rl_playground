from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np

from envs.base import StochasticMDPEnv


class SysAdminStates(Enum):
    S0 = 0
    S1 = 1
    S2 = 2
    S3 = 3


class SysAdminActions(Enum):
    NOREBOOT = 0
    REBOOT1 = 1
    REBOOT2 = 2


REWARDS = {
    SysAdminStates.S0: 0,
    SysAdminStates.S1: 0,
    SysAdminStates.S2: 0,
    SysAdminStates.S3: 1,
}


@dataclass
class SysAdminEnv(StochasticMDPEnv):

    idx_state_map = {i.value: i for i in SysAdminStates}
    idx_action_map = {i.value: i for i in SysAdminActions}

    transition = {
        SysAdminStates.S0: {
            SysAdminActions.NOREBOOT: {
                SysAdminStates.S0: 0.693,
                SysAdminStates.S1: 0.007,
                SysAdminStates.S2: 0.297,
                SysAdminStates.S3: 0.003
            },
            SysAdminActions.REBOOT1: {
                SysAdminStates.S0: 0.0,
                SysAdminStates.S1: 0.0,
                SysAdminStates.S2: 0.99,
                SysAdminStates.S3: 0.01
            },
            SysAdminActions.REBOOT2: {
                SysAdminStates.S0: 0.0,
                SysAdminStates.S1: 0.7,
                SysAdminStates.S2: 0.0,
                SysAdminStates.S3: 0.3
            },
        },
        SysAdminStates.S1: {
            SysAdminActions.NOREBOOT: {
                SysAdminStates.S0: 0.35,
                SysAdminStates.S1: 0.35,
                SysAdminStates.S2: 0.15,
                SysAdminStates.S3: 0.15
            },
            SysAdminActions.REBOOT1: {
                SysAdminStates.S0: 0.0,
                SysAdminStates.S1: 0.0,
                SysAdminStates.S2: 0.5,
                SysAdminStates.S3: 0.5
            },
            SysAdminActions.REBOOT2: {
                SysAdminStates.S0: 0.0,
                SysAdminStates.S1: 0.7,
                SysAdminStates.S2: 0.0,
                SysAdminStates.S3: 0.3
            },
        },
        SysAdminStates.S2: {
            SysAdminActions.NOREBOOT: {
                SysAdminStates.S0: 0.07,
                SysAdminStates.S1: 0.03,
                SysAdminStates.S2: 0.63,
                SysAdminStates.S3: 0.27
            },
            SysAdminActions.REBOOT1: {
                SysAdminStates.S0: 0.0,
                SysAdminStates.S1: 0.0,
                SysAdminStates.S2: 0.7,
                SysAdminStates.S3: 0.3
            },
            SysAdminActions.REBOOT2: {
                SysAdminStates.S0: 0.0,
                SysAdminStates.S1: 0.1,
                SysAdminStates.S2: 0.0,
                SysAdminStates.S3: 0.9
            },
        },
        SysAdminStates.S3: {
            SysAdminActions.NOREBOOT: {
                SysAdminStates.S0: 0.01,
                SysAdminStates.S1: 0.09,
                SysAdminStates.S2: 0.09,
                SysAdminStates.S3: 0.81
            },
            SysAdminActions.REBOOT1: {
                SysAdminStates.S0: 0.0,
                SysAdminStates.S1: 0.0,
                SysAdminStates.S2: 0.1,
                SysAdminStates.S3: 0.9
            },
            SysAdminActions.REBOOT2: {
                SysAdminStates.S0: 0.0,
                SysAdminStates.S1: 0.1,
                SysAdminStates.S2: 0.0,
                SysAdminStates.S3: 0.9
            },
        },

    }

    def get_transitions(self, state: SysAdminStates, action: SysAdminActions) \
            -> List[Tuple[float, SysAdminStates, int]]:
        # [(prob, next_state, reward)]
        return [
            (prob, next_state, REWARDS[next_state])
            for next_state, prob in self.transition[state][action].items()
        ]
