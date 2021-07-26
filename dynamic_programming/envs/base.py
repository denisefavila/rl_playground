from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import TypeVar, List, Tuple, Dict

STATE = TypeVar('STATE')
ACTION = TypeVar('ACTION')


@dataclass
class StochasticMDPEnv(ABC):

    idx_state_map = Dict[int, STATE]
    idx_action_map = Dict[int, ACTION]

    @abstractmethod
    def get_transitions(self, state: STATE, action: ACTION) -> List[Tuple[float, STATE, int]]:
        """
        Return a list with transitions [(prob, next_state, reward)]
        """
        pass

