import copy
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from generals.gui import GUI
from generals.gui.event_handler import ReplayCommand
from generals.gui.properties import GuiMode
from generals.rewards.reward_fn import RewardFn
from generals.rewards.win_lose_reward_fn import WinLoseRewardFn

from .action import Action
from .config import DIRECTIONS
from .grid import NEUTRAL_ID, Grid, GridFactory
from .observation import Observation


class Environment:
    """
    This class represents the Environment we're trying to optimize and accordingly manages the bulk
    of the logic for the game -- generals.io. Like any reinforcement-learning Environment, it accepts actions
    at each timestep and gives rise to new observations and rewards based on those actions.

    When implementing new Environment classes for existing RL frameworks (e.g. Gymnasium, PettingZoo,
    RLLib), this class should manage all game-related logic. And that new class should only modify
    the outputs of this class to cater to each libraries specific expectations. For examples of how
    this can be done, look to the currently available env implementations in generals/envs.
    """

    # Generals games are extremely unlikely to need values beyond these. However,
    # they may still be tweaked if desired. The environment/simulator will
    # crash if any of these values are exceeded.
    max_army_size = 100_000
    max_timestep = 100_000
    max_land_owned = 250 * 250

    # Every {increment_rate} turns, each land-tile that's owned
    # generates an army.
    increment_rate = 50

    # Default fps of the GUI. Can be modified via speed_multiplier.
    default_render_fps = 6

    def __init__(
        self,
        agent_ids: list[str],
        grid_factory: GridFactory = None,
        truncation: int = None,
        reward_fn: RewardFn = None,
        to_render: bool = False,
        speed_multiplier: float = 1.0,
        save_replays: bool = False,
        pad_to: int = None,
    ):
        self.agent_ids = agent_ids
        self.grid_factory = grid_factory if grid_factory is not None else GridFactory(agent_ids)
        self.truncation = truncation
        self.reward_fn = reward_fn if reward_fn is not None else WinLoseRewardFn()
        self.to_render = to_render
        self.speed_multiplier = speed_multiplier
        self.save_replays = save_replays
        self.pad_to = pad_to

        self.episode_num = 0
        self.reset()

    def render(self):
        if self.to_render:
            fps = int(self.default_render_fps * self.speed_multiplier)
            _ = self.gui.tick(self.get_infos(), self.num_turns, fps)

    def close(self):
        if self.to_render:
            self.gui.close()

    def reset(self, rng: np.random.Generator = None, seed: int = None, options: dict[str, Any] = None):
        """reset contains instructions common for resetting all types of envs."""

        # Observations for each agent at the prior time-step.
        self.prior_observations: dict[str, Observation] = None

        # The priority-ordering of each agent. This determines which agents' action is processed first.
        self.agents_in_order_of_prio = self.agent_ids[:]

        # The number of turns and the time displayed in game differ. In generals.io there are two turns
        # each agent may take per in-game unit of time.
        self.num_turns = 0

        # Reset the grid.
        if options is not None and "grid_layout_str" in options:
            self.grid = Grid.from_string(options["grid_layout_str"], self.agent_ids)
        else:
            self.grid = self.grid_factory.generate(rng=rng, seed=seed)

        # Reset the GUI for the upcoming game.
        if self.to_render:
            self.gui = GUI(self.grid, self.agent_ids, GuiMode.TRAIN)

        # Prepare a new replay to save the upcoming game.
        if self.save_replays:
            self.replay = Replay(self.episode_num, self.grid, self.agent_ids)
            self.replay.add_state(self.grid)

        self.episode_num += 1

        observations = {agent_id: self.agent_observation(agent_id) for agent_id in self.agent_ids}
        infos = {agent_id: self.get_infos()[agent_id] for agent_id in self.agent_ids}

        return observations, infos

    def step(
        self, actions: dict[str, Action]
    ) -> tuple[dict[str, Observation], dict[str, Any], bool, bool, dict[str, Any]]:
        """
        Perform one step of the game
        """
        done_before_actions = self.is_done()
        for agent in self.agents_in_order_of_prio:
            to_pass, si, sj, direction, to_split = actions[agent]
            if to_pass:
                # Skip if agent wants to pass the turn
                continue
            if to_split:
                # Agent wants to split the army
                army_to_move = self.grid.armies[si, sj] // 2
            else:
                # Leave just one army in the source cell
                army_to_move = self.grid.armies[si, sj] - 1

            if army_to_move < 1:
                # Skip if army size to move is less than 1
                continue

            # Cap the amount of army to move (previous moves may have lowered available army)
            army_to_move = min(army_to_move, self.grid.armies[si, sj] - 1)
            army_to_stay = self.grid.armies[si, sj] - army_to_move

            # Check if the current agent still owns the source cell and has more than 1 army
            if self.grid.owners[agent][si, sj] == 0 or army_to_move < 1:
                continue

            # Destination indices.
            di, dj = (
                si + DIRECTIONS[direction].value[0],
                sj + DIRECTIONS[direction].value[1],
            )

            # Skip if the destination cell is a mountain or out of bounds.
            if di < 0 or di >= self.grid.shape[0] or dj < 0 or dj >= self.grid.shape[1]:
                continue
            if self.grid.mountains[di, dj] == 1:
                continue

            # Figure out the target square owner and army size
            target_square_army = self.grid.armies[di, dj]
            target_square_owner_idx = np.argmax(
                [self.grid.owners[agent][di, dj] for agent in ["neutral"] + self.agent_ids]
            )
            target_square_owner = (["neutral"] + self.agent_ids)[target_square_owner_idx]
            if target_square_owner == agent:
                self.grid.armies[di, dj] += army_to_move
                self.grid.armies[si, sj] = army_to_stay
            else:
                # Calculate resulting army, winner and update grid.
                remaining_army = np.abs(target_square_army - army_to_move)
                square_winner = agent if target_square_army < army_to_move else target_square_owner
                self.grid.armies[di, dj] = remaining_army
                self.grid.armies[si, sj] = army_to_stay
                self.grid.owners[square_winner][di, dj] = True
                if square_winner != target_square_owner:
                    self.grid.owners[target_square_owner][di, dj] = False

        # Swap agent order (because priority is alternating)
        self.agents_in_order_of_prio = self.agents_in_order_of_prio[::-1]

        if not done_before_actions:
            self.num_turns += 1

        if self.is_done():
            # give all cells of loser to winner
            winner = self.agent_ids[0] if self.agent_won(self.agent_ids[0]) else self.agent_ids[1]
            loser = self.agent_ids[1] if winner == self.agent_ids[0] else self.agent_ids[0]
            self.grid.owners[winner] += self.grid.owners[loser]
            self.grid.owners[loser] = np.full(self.grid.shape, False)
        else:
            self._global_game_update()

        observations = {agent: self.agent_observation(agent) for agent in self.agent_ids}
        infos = self.get_infos()

        if self.prior_observations is None:
            # Cannot compute rewards without prior-observations. This should only happen
            # on the first time-step.
            rewards = {agent: 0.0 for agent in self.agent_ids}
        else:
            rewards = {
                agent: self.reward_fn(
                    prior_obs=self.prior_observations[agent],
                    # Technically actions are the prior-actions, since they are what will give
                    # rise to the current-observations.
                    prior_action=actions[agent],
                    obs=observations[agent],
                )
                for agent in self.agent_ids
            }

        terminated = self.is_done()
        truncated = False
        if self.truncation is not None:
            truncated = self.num_turns >= self.truncation

        if self.save_replays:
            self.replay.add_state(self.grid)

        if (terminated or truncated) and self.save_replays:
            self.replay.store()

        self.prior_observations = observations

        return observations, rewards, terminated, truncated, infos

    def _global_game_update(self) -> None:
        """
        Update game state globally.
        """

        owners = self.agent_ids

        # every `increment_rate` steps, increase army size in each cell
        if self.num_turns % self.increment_rate == 0:
            for owner in owners:
                self.grid.armies += self.grid.owners[owner]

        # Increment armies on general and city cells, but only if they are owned by player
        if self.num_turns % 2 == 0 and self.num_turns > 0:
            update_mask = self.grid.generals + self.grid.cities
            for owner in owners:
                self.grid.armies += (update_mask * self.grid.owners[owner]).astype(int)

    def is_done(self) -> bool:
        """
        Returns True if the game is over, False otherwise.
        """
        return any(self.agent_won(agent) for agent in self.agent_ids)

    def get_infos(self) -> dict[str, dict[str, Any]]:
        """
        Returns a dictionary of player statistics.
        Keys and values are as follows:
        - army: total army size
        - land: total land size
        - is_done: True if the game is over, False otherwise
        - is_winner: True if the player won, False otherwise
        """
        players_stats = {}
        for agent in self.agent_ids:
            army_size = np.sum(self.grid.armies * self.grid.owners[agent]).astype(int)
            land_size = np.sum(self.grid.owners[agent]).astype(int)
            players_stats[agent] = {
                "army": army_size,
                "land": land_size,
                "is_done": self.is_done(),
                "is_winner": self.agent_won(agent),
            }
        return players_stats

    def agent_observation(self, agent: str) -> Observation:
        """
        Returns an observation for a given agent.
        """
        scores = {}
        for _agent in self.agent_ids:
            army_size = np.sum(self.grid.armies * self.grid.owners[_agent]).astype(int)
            land_size = np.sum(self.grid.owners[_agent]).astype(int)
            scores[_agent] = {
                "army": army_size,
                "land": land_size,
            }

        is_visible = self.grid.get_visibility(agent)
        is_invisible = 1 - is_visible

        opponent_id = self.agent_ids[0] if agent == self.agent_ids[1] else self.agent_ids[1]
        structures_in_fog = is_invisible * (self.grid.mountains + self.grid.cities)

        return Observation(
            armies=(is_visible * self.grid.armies),
            generals=(is_visible * self.grid.generals),
            cities=(is_visible * self.grid.cities),
            mountains=(is_visible * self.grid.mountains),
            neutral_cells=(is_visible * self.grid.owners[NEUTRAL_ID]),
            owned_cells=(is_visible * self.grid.owners[agent]),
            opponent_cells=(is_visible * self.grid.owners[opponent_id]),
            fog_cells=(is_invisible - structures_in_fog),
            structures_in_fog=structures_in_fog,
            owned_land_count=scores[agent]["land"],
            owned_army_count=scores[agent]["army"],
            opponent_land_count=scores[opponent_id]["land"],
            opponent_army_count=scores[opponent_id]["army"],
            timestep=self.num_turns,
            priority=1 if agent == self.agents_in_order_of_prio[0] else 0,
            pad_to=self.pad_to,
        )

    def agent_won(self, agent: str) -> bool:
        """
        Returns True if the agent won the game, False otherwise.
        """

        num_generals_owned_by_agent = (self.grid.generals * self.grid.owners[agent]).sum()
        num_generals = self.grid.generals.sum()

        return num_generals == num_generals_owned_by_agent


class Replay:
    replays_dir = Path.cwd() / "replays"

    def __init__(self, episode_num: int, grid: Grid, agent_ids: list[str]):
        # Create the replays/ directory if necessary.
        self.replays_dir.mkdir(parents=True, exist_ok=True)

        self.replay_filename = self.replays_dir / f"replay_{episode_num}.pkl"
        self.grid = grid
        self.agent_ids = agent_ids

        self.game_states: list[Grid] = []

    def add_state(self, state: Grid):
        copied_state = copy.deepcopy(state)
        self.game_states.append(copied_state)

    def store(self):
        with open(self.replay_filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Replay saved to {self.replay_filename}.")

    @classmethod
    def load(cls, path):
        path = path if path.endswith(".pkl") else path + ".pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def play(self):
        env = Environment(self.agent_ids)
        # Manually set the grid.
        env.grid = copy.deepcopy(self.game_states[0])
        gui = GUI(env.grid, self.agent_ids, gui_mode=GuiMode.REPLAY)
        desired_turn_num, last_input_time, last_move_time = 0, 0, 0
        total_num_turns = len(self.game_states) - 1

        while True:
            current_time = time.time()

            # Check for any commands every 8ms.
            if current_time - last_input_time > 8e-3:
                command = gui.tick()
                last_input_time = current_time
            else:
                command = ReplayCommand()

            if command.restart:
                desired_turn_num = 0

            desired_turn_num += command.frame_change
            # Ensure 0 <= current-turn-num <= total_num_turns.
            desired_turn_num = max(min(desired_turn_num, total_num_turns), 0)

            # If the replay is paused and a specific turn-number has been requested, go to it.
            if gui.properties.is_paused and desired_turn_num != env.num_turns:
                env.grid = copy.deepcopy(self.game_states[desired_turn_num])
                env.num_turns = desired_turn_num
                last_move_time = current_time
            # Otherwise, replay through the game normally.
            elif (
                current_time - last_move_time > (1 / gui.properties.game_speed) * 0.512 and not gui.properties.is_paused
            ):
                desired_turn_num += 1

                is_replay_done = env.is_done() or desired_turn_num >= total_num_turns
                if is_replay_done:
                    gui.properties.is_paused = True

                env.grid = copy.deepcopy(self.game_states[desired_turn_num])
                env.num_turns = desired_turn_num
                last_move_time = current_time

            gui.properties.clock.tick(60)
