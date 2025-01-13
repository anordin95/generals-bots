import numpy as np
import pytest
from generals.core.grid import Grid, GridFactory
import random
import math

def test_create_from_string():
    grid_layout_str = """
        .....
        .G##c
        ...c.
        ..cc.
        ...G.
    """
    Grid.from_string(grid_layout_str)

def test_create_valid_grids_from_string():
    grid_layout_str = """
        .....
        .G##c
        ...c.
        ..cc.
        ...G.
    """
    Grid.from_string(grid_layout_str)

    grid_layout_str = """
        .....
        GGc#c
        ##.c.
        ..c##
        .....
    """
    Grid.from_string(grid_layout_str)

def test_create_invalid_grids_from_string():

    grid_layout_str = """
        .....
        .G##c
        ##.c.
        ..###
        ...G.
    """
    with pytest.raises(Exception):
        Grid.from_string(grid_layout_str)

    grid_layout_str = """
        ...#.
        #Gc#c
        ##.#G
        ..c##
        .....
    """
    with pytest.raises(Exception):
        Grid.from_string(grid_layout_str)

    grid_layout_str = """
        ...#.
        G#c#c
        ##.#G
        ..c#.
        .....
    """
    with pytest.raises(Exception):
        Grid.from_string(grid_layout_str)

def _compute_generals_distance(grid: Grid) -> float:
    generals_indices = np.argwhere(grid.generals)
    generals_distance = math.sqrt(((generals_indices[0] - generals_indices[1]) ** 2).sum())
    return generals_distance


def test_grid_factory_makes_valid_grids():

    grid_factory = GridFactory(agent_ids=["A", "B"], min_grid_dims=(15, 15), max_grid_dims=(23, 23))
    rng = np.random.default_rng()

    for _ in range(10):
        grid = grid_factory.generate(rng=rng)
        assert grid.generals.sum() == 2
        min_generals_dist = ((grid.shape[0] + grid.shape[1]) // 4)
        assert _compute_generals_distance(grid) >= min_generals_dist

    for _ in range(10):
        seed = random.randint(1, 100)
        grid = grid_factory.generate(seed=seed)
        assert grid.generals.sum() == 2
        min_generals_dist = ((grid.shape[0] + grid.shape[1]) // 4)
        assert _compute_generals_distance(grid) >= min_generals_dist

def test_grid_factory_determinism():
    grid_factory = GridFactory(agent_ids=["A", "B"])
    fixed_seed = 42

    for _ in range(10):
        grid1 = grid_factory.generate(rng=np.random.default_rng(fixed_seed))
        grid2 = grid_factory.generate(rng=np.random.default_rng(fixed_seed))

        assert np.array_equal(grid1.armies, grid2.armies)
        assert np.array_equal(grid1.generals, grid2.generals)
        assert np.array_equal(grid1.cities, grid2.cities)
        assert np.array_equal(grid1.mountains, grid2.mountains)

    for _ in range(10):
        grid1 = grid_factory.generate(seed=fixed_seed)
        grid2 = grid_factory.generate(seed=fixed_seed)

        assert np.array_equal(grid1.armies, grid2.armies)
        assert np.array_equal(grid1.generals, grid2.generals)
        assert np.array_equal(grid1.cities, grid2.cities)
        assert np.array_equal(grid1.mountains, grid2.mountains)
