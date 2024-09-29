from generals.env import pz_generals
from generals.agents import RandomAgent, ExpanderAgent
from generals.map import Mapper

# Initialize agents - their names are then called for actions
randomer = RandomAgent()
expander = ExpanderAgent()

agents = {
    randomer.name: randomer,
    expander.name: expander,
}

mapper = Mapper(
    grid_dims=(4, 8), # height x width
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(0, 0), (3, 3)],
)

# Custom map that will override mapper's map for next game
map = """
A..#
.#3#
...#
##B#
"""

# Create environment
env = pz_generals(mapper, agents, render_mode=None) # Disable rendering

options = {
    "map": map,
    "replay_file": "replay",
}

observations, info = env.reset(options=options)
done = False

while not done:
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].play(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    done = any(terminated.values())