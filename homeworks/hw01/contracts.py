from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Callable

import gym


@dataclass(frozen=True)
class State:
    """Родительский класс для состояний среды"""
    pass


class Strategy(ABC):
    """Родительский класс для стратегий"""

    @abstractmethod
    def get_action(self, state: State) -> int:
        pass


class EnvironmentWrapper:
    """Оборачивает оригинальный gym.Env, чтобы возвращать состояния в виде объектов State"""

    def __init__(self,
                 gym_env: gym.Env,
                 state_getter: Callable[[Tuple, gym.Env], State]):
        self.env = gym_env
        self.state_getter = state_getter

    def reset(self) -> State:
        gym_state: Tuple = self.env.reset()
        return self.state_getter(gym_state, self.env)

    def step(self, action: int) -> Tuple[State, float, bool]:
        gym_state, reward, done, _ = self.env.step(action)
        state = self.state_getter(gym_state, self.env)
        return state, reward, done
