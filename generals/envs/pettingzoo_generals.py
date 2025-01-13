import functools
from typing import Any, TypeAlias

import numpy as np
import pettingzoo  # type: ignore
from gymnasium import spaces

from generals.core.environment import Action, Environment, Observation
from generals.core.grid import GridFactory
from generals.rewards.reward_fn import RewardFn

AgentID: TypeAlias = str


class PettingZooGenerals(pettingzoo.ParallelEnv):
    def __init__(
        self,
        agent_ids: list[str],
        grid_factory: GridFactory = None,
        truncation: int = None,
        reward_fn: RewardFn = None,
        to_render: bool = False,
        speed_multiplier: float = 1.0,
    ):
        """
        Args:
            agents: A dictionary of the agent-ids & agents.
            grid_factory: Can be used to specify the game-board i.e. grid generator.
            truncation: The maximum number of turns a game can last before it's truncated.
            reward_fn: An instance of the RewardFn abstract base class.
            speed_multiplier: Relatively increase or decrease the speed of the real-time
                game graphic. This has no effect if render_mode is None.
            to_render: If True, the game will be rendered in real-time.
        """
        assert len(agent_ids) == len(
            set(agent_ids)
        ), "Agent ids must be unique - you can pass custom ids to agent constructors."

        self.truncation = truncation
        self.agent_ids = agent_ids
        self.environment = Environment(agent_ids, grid_factory, truncation, reward_fn, to_render, speed_multiplier)

    @functools.cache
    def observation_space(self, agent: AgentID) -> spaces.Space:
        grid_dims = self.environment.grid.shape
        grid_multi_binary = spaces.MultiBinary(grid_dims)

        return spaces.Dict(
            {
                "armies": spaces.MultiDiscrete(np.ones(grid_dims, dtype=int) * Environment.max_army_size),
                "generals": grid_multi_binary,
                "cities": grid_multi_binary,
                "mountains": grid_multi_binary,
                "neutral_cells": grid_multi_binary,
                "owned_cells": grid_multi_binary,
                "opponent_cells": grid_multi_binary,
                "fog_cells": grid_multi_binary,
                "structures_in_fog": grid_multi_binary,
                "owned_land_count": spaces.Discrete(Environment.max_land_owned),
                "owned_army_count": spaces.Discrete(Environment.max_army_size),
                "opponent_land_count": spaces.Discrete(Environment.max_land_owned),
                "opponent_army_count": spaces.Discrete(Environment.max_army_size),
                "timestep": spaces.Discrete(Environment.max_timestep),
                "priority": spaces.Discrete(2),
            }
        )

    @functools.cache
    def action_space(self, agent: AgentID) -> spaces.Space:
        grid_height, grid_width = self.environment.grid.shape
        return spaces.MultiDiscrete([2, grid_height, grid_width, 4, 2])

    def render(self):
        self.environment.render()

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, Observation], dict[AgentID, dict]]:
        observations, infos = self.environment.reset(seed=seed, options=options)
        return observations, infos

    def step(
        self, actions: dict[AgentID, Action]
    ) -> tuple[
        dict[AgentID, Observation],
        dict[AgentID, float],
        bool,
        bool,
        dict[AgentID, dict[str, Any]],
    ]:
        observations, rewards, terminated, truncated, infos = self.environment.step(actions)
        return observations, rewards, terminated, truncated, infos
