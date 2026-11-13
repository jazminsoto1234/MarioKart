"""Minimal agent interface template."""
from stk_agent import Action


class Agent:
    def __init__(self, config: dict):
        self.config = config
        self.memory = None

    def reset(self):
        self.memory = None

    def act(self, rgb) -> Action:
        """Return an action from the current RGB frame."""
        raise NotImplementedError

    def close(self):
        pass
