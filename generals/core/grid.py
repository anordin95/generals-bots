import dataclasses
import math

import numpy as np
import scipy

# The location of generals are specified as two tuples.
GeneralsLocs = tuple[tuple[int, int], tuple[int, int]]

# This is the initial owner/agent-id of empty land & cities.
NEUTRAL_ID = "neutral"

# These are characters used in string-representations of a grids layout.
EMPTY = "."
CITY = "c"
GENERAL = "G"
MOUNTAIN = "#"


@dataclasses.dataclass
class Grid:
    armies: np.ndarray
    generals: np.ndarray
    mountains: np.ndarray
    cities: np.ndarray
    owners: dict[str, np.ndarray]

    def get_visibility(self, agent_id: str) -> np.ndarray:
        return scipy.ndimage.maximum_filter(self.owners[agent_id], size=3).astype(bool)

    @property
    def shape(self) -> tuple[int, int]:
        return self.armies.shape

    @staticmethod
    def from_string(grid_layout_str: str, agent_ids: list[str] = ["A", "B"]) -> "Grid":
        """Create a Grid from a Grid-layout string. A layout defines the Grids initial
        or starting setup -- the locations of cities, mountains, generals & neutral/empty
        land.

        Here's an example of a Grid-layout string.
        grid_layout_str = '''
            .....
            .G##c
            ...c.
            #...c
            ...Gc
        '''
        """

        # Parse the input string.
        line_arrays = []
        for line_str in grid_layout_str.strip().split("\n"):
            line_str = line_str.strip()
            line_chars = list(line_str)
            line_arr = np.array(line_chars)
            line_arrays.append(line_arr)
        grid_layout_arr = np.array(line_arrays)

        # Parse the mountains & cities.
        mountains = grid_layout_arr == MOUNTAIN
        cities = grid_layout_arr == CITY

        # Parse the generals & generate the initial armies.
        generals_locs = np.argwhere(grid_layout_arr == GENERAL)
        generals = GridFactory.generals_locs_to_mask(generals_locs, grid_dims=mountains.shape)
        armies = GridFactory.generate_armies_mask(generals, cities, rng=np.random.default_rng())
        owners = GridFactory.generate_owners_mask(generals_locs, agent_ids, mountains)

        # Ensure there are only two generals which can reach one another.
        assert len(generals_locs) == 2, f"Received grid with {len(generals_locs)} generals, not 2, on it."
        assert _are_generals_reachable(
            is_passable=~mountains & ~cities, generals_locs=generals_locs
        ), "There is no path between the generals on the provided grid."

        return Grid(armies, generals, mountains, cities, owners)


class GridFactory:
    def __init__(
        self,
        agent_ids: list[str],
        min_grid_dims: tuple[int, int] = (15, 15),
        max_grid_dims: tuple[int, int] = (23, 23),
        p_mountain: float = 0.2,
        p_city: float = 0.05,
        generals_locs: GeneralsLocs = None,
    ):
        """
        Args:
            min_grid_dims: The minimum (inclusive) height & width of the grid.
            max_grid_dims: The maximum (inclusive) height & width of the grid.
            agent_ids: A list of the agents ids.
            p_mountain: The probability any given square will be a mountain.
            p_city: The probability any given square will be a city.
            generals_locs: A fixed (row, col) for each general for every
                generated grid. If specified, the order the agent-ids are
                provided in will correspond to the order of the generals
                locations. If unspecified, the generals, will be randomly,
                but thoughtfully (i.e. not too close, reachable, etc.),
                placed for each generated grid.
        """
        self.min_grid_dims = min_grid_dims
        self.max_grid_dims = max_grid_dims
        self.agent_ids = agent_ids
        self.p_mountain = p_mountain
        self.p_city = p_city
        self.generals_locs = generals_locs

    def _get_largest_region_mask(self, is_passable: np.ndarray) -> np.ndarray:
        
        # Pad with 0's around the border, otherwise scipy wraps boundaries, e.g.
        # the top-middle cell is connected to the bottom-middle cell.
        is_passable = np.pad(is_passable, 1, mode="constant", constant_values=0)
        connected_region_mask, num_regions = scipy.ndimage.label(is_passable)

        # Find the largest contiguous region.
        largest_region_label, largest_region_size = -1, -1
        for region_label in range(1, num_regions + 1):
            region_size = (connected_region_mask == region_label).sum()
            if region_size > largest_region_size:
                largest_region_label, largest_region_size = region_label, region_size

        # Save the largest-regions mask & remove the padding.
        largest_region_mask = connected_region_mask == largest_region_label
        largest_region_mask = largest_region_mask[1:-1, 1:-1]

        return largest_region_mask

    def generate(self, rng: np.random.Generator = None, seed: int = None) -> Grid:
        """
        Construct a new Grid.

        Args:
            rng: A numpy random number generator to use for any random operations.
            seed: A seed to initialize any ensuing random operations. If both
                rng & seed are provided, rng will be used and seed ignored.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        grid_height = rng.integers(self.min_grid_dims[0], self.max_grid_dims[0] + 1)
        grid_width = rng.integers(self.min_grid_dims[1], self.max_grid_dims[1] + 1)
        grid_dims = (grid_height, grid_width)

        # Ensure we don't generate a split-map scenario: one where there are two large regions entirely
        # seperated by mountains and cities. And check if the generals locations were manually specified
        # that they are both within the largest connected region.
        largest_region_proportion = 0.0
        are_generals_disconnected = True
        while largest_region_proportion <= 0.70 or are_generals_disconnected:
            p_empty = 1 - self.p_mountain - self.p_city
            mountain_or_city = rng.choice([0, "m", "c"], size=grid_dims, p=[p_empty, self.p_mountain, self.p_city])
            mountains = mountain_or_city == "m"
            cities = mountain_or_city == "c"
            is_passable = ~mountains & ~cities

            # Compute the proportion of the non-mountain tiles encompassed by the largest region.
            largest_region_mask = self._get_largest_region_mask(is_passable)
            largest_region_size = largest_region_mask.sum()
            largest_region_proportion = largest_region_size.sum() / is_passable.sum()

            
            if self.generals_locs is not None:
                # If the generals locations were manually specified, check that they are both within 
                # that largest region.
                is_general_one_in_region = largest_region_mask[self.generals_locs[0]]
                is_general_two_in_region = largest_region_mask[self.generals_locs[1]]
                are_generals_disconnected = not (is_general_one_in_region and is_general_two_in_region)
            else:
                # The generals location generation logic ensures they will be connected, so no need to 
                # re-do the grid layout.
                are_generals_disconnected = False
        
        generals_locs = self.generals_locs
        if generals_locs is None:
            generals_locs = self.generate_generals_locs(is_passable=(~mountains & ~cities), rng=rng)
        else:
            # The manually specified generals-locations may conflict with the randomly placed mountains & cities.
            for general_loc in generals_locs:
                mountains[tuple(general_loc)] = False
                cities[tuple(general_loc)] = False

        generals = self.generals_locs_to_mask(generals_locs, grid_dims)
        armies = self.generate_armies_mask(generals, cities, rng)
        owners = self.generate_owners_mask(generals_locs, self.agent_ids, mountains)

        grid = Grid(armies, generals, mountains, cities, owners)
        return grid

    @staticmethod
    def generals_locs_to_mask(generals_locs: GeneralsLocs, grid_dims: tuple[int, int]) -> np.ndarray:
        general_one_loc, general_two_loc = generals_locs
        generals_mask = np.full(grid_dims, False, dtype=bool)
        generals_mask[tuple(general_one_loc)] = True
        generals_mask[tuple(general_two_loc)] = True

        return generals_mask

    @staticmethod
    def generate_armies_mask(
        generals_mask: np.ndarray, cities_mask: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        armies_mask = np.zeros(generals_mask.shape, dtype=int)
        armies_mask[generals_mask] = 1

        num_cities = cities_mask.sum()
        # Cities always spawn with anywhere from 41 through 49 neutral armies on them.
        city_army_values = rng.integers(low=41, high=49, size=(num_cities,))
        armies_mask[cities_mask] = city_army_values

        return armies_mask

    @staticmethod
    def generate_owners_mask(
        generals_locs: GeneralsLocs, agent_ids: list[str], mountains: np.ndarray
    ) -> dict[str, np.ndarray]:
        grid_dims = mountains.shape
        owners_masks = {}

        # Generate ownership mask for neutral.
        neutral_mask = np.full(grid_dims, True)
        neutral_mask[mountains] = False
        for general_loc in generals_locs:
            neutral_mask[tuple(general_loc)] = False
            owners_masks[NEUTRAL_ID] = neutral_mask

        # Generate ownership masks for each agent.
        for idx, agent_id in enumerate(agent_ids):
            agent_owner_mask = np.full(grid_dims, False)
            agent_general_loc = tuple(generals_locs[idx])
            agent_owner_mask[agent_general_loc] = True
            owners_masks[agent_id] = agent_owner_mask

        return owners_masks

    @staticmethod
    def generate_generals_locs(is_passable: np.ndarray, rng: np.random.Generator) -> GeneralsLocs:
        grid_height, grid_width = is_passable.shape
        # Pad with 0's around the border, otherwise scipy wraps boundaries, e.g.
        # the top-middle cell is connected to the bottom-middle cell.
        is_passable = np.pad(is_passable, 1, mode="constant", constant_values=0)
        connected_region_mask, num_regions = scipy.ndimage.label(is_passable)

        # Find the largest contiguous region.
        largest_region_label, largest_region_size = -1, -1
        for region_label in range(1, num_regions + 1):
            region_size = (connected_region_mask == region_label).sum()
            if region_size > largest_region_size:
                largest_region_label, largest_region_size = region_label, region_size

        # Save the largest-regions mask & remove the padding.
        largest_region_mask = connected_region_mask == largest_region_label
        largest_region_mask = largest_region_mask[1:-1, 1:-1]

        # Do not place the first general roughly within or around the center of the map.
        center = (grid_height / 2, grid_width / 2)
        is_not_in_innermost_rect = np.full((grid_height, grid_width), True)
        innermost_rect_height, innermost_rect_width = grid_height // 3, grid_width // 3
        innermost_rect_top_left = (
            math.floor(center[0] - innermost_rect_height / 2),
            math.floor(center[1] - innermost_rect_width / 2),
        )
        innermost_rect_bottom_right = (
            math.ceil(center[0] + innermost_rect_height / 2),
            math.ceil(center[1] + innermost_rect_width / 2),
        )
        is_not_in_innermost_rect[
            innermost_rect_top_left[0] : innermost_rect_bottom_right[0],
            innermost_rect_top_left[1] : innermost_rect_bottom_right[1],
        ] = False

        # Place the first general.
        valid_general_one_locs = np.argwhere(largest_region_mask & is_not_in_innermost_rect)
        random_idx = rng.integers(0, len(valid_general_one_locs))
        general_one_loc = valid_general_one_locs[random_idx]

        # Place the second general.
        grid_indices = np.stack(np.meshgrid(range(grid_height), range(grid_width), indexing="ij"), axis=-1)
        dist_from_general_one_mask = np.sqrt(((grid_indices - general_one_loc) ** 2).sum(axis=-1))
        # Choose from any square in the largest region mask whose distance is in the 85th
        # percentile of valid distances or above.
        distances_from_general_one = dist_from_general_one_mask[largest_region_mask]
        min_distance_from_general_one = np.percentile(distances_from_general_one, 85)
        valid_general_two_locs = np.argwhere(dist_from_general_one_mask > min_distance_from_general_one)
        random_idx = rng.integers(0, len(valid_general_two_locs))
        general_two_loc = valid_general_two_locs[random_idx]

        return (general_one_loc, general_two_loc)


def _are_generals_reachable(is_passable: np.ndarray, generals_locs: GeneralsLocs) -> bool:
    """Determine if generals placed in the given location can reach one another without
    having to cross through mountains or cities.

    Args:
        is_passable: a mask denoting which cells are neither a city nor a mountain.
        general_locs: the location of each general.
    """

    # Pad with 0's around the border, otherwise scipy wraps boundaries, e.g.
    # the top-middle cell is connected to the bottom-middle cell.
    is_passable = np.pad(is_passable, 1, mode="constant", constant_values=0)
    connected_region_mask, _ = scipy.ndimage.label(is_passable)
    # Undo the padding.
    connected_region_mask = connected_region_mask[1:-1, 1:-1]
    general_one_region_label = connected_region_mask[tuple(generals_locs[0])]
    general_two_region_label = connected_region_mask[tuple(generals_locs[1])]

    return general_one_region_label == general_two_region_label
