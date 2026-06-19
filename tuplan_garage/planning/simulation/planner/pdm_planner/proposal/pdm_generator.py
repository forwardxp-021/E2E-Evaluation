from __future__ import annotations


class PDMGenerator:
    def __init__(self, leading_agent_update_rate: int = 2, **kwargs) -> None:
        self.leading_agent_update_rate = int(leading_agent_update_rate)
        self.extra_kwargs = kwargs
