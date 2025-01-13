from enum import Enum

from pygame.time import Clock

from generals.core.config import Dimension
from generals.core.grid import Grid


class GuiMode(Enum):
    TRAIN = "train"
    GAME = "game"
    REPLAY = "replay"


class Properties:
    def __init__(
        self,
        grid: Grid,
        agent_ids: list[str],
        gui_mode: GuiMode,
        game_speed: float = 1.0,
        clock: Clock = Clock(),
        font_size: int = 18,
    ):
        # Given fields.
        self.grid = grid
        self.agent_ids = agent_ids
        self.grid_height, self.grid_width = self.grid.shape
        self.gui_mode = gui_mode
        self.game_speed = game_speed
        self.clock = clock
        self.font_size = font_size

        colors = [(255, 107, 108), (0, 130, 255)]
        self.agent_id_to_color = {agent_ids[idx]: colors[idx] for idx in range(0, 2)}

        # Derived fields.
        self.display_grid_height = Dimension.SQUARE_SIZE.value * self.grid_height
        self.display_grid_width = Dimension.SQUARE_SIZE.value * self.grid_width
        self.right_panel_width = 4 * Dimension.GUI_CELL_WIDTH.value
        self.is_paused = False
        self.agent_fov = {agent_id: True for agent_id in agent_ids}

    def update_speed(self, multiplier: float):
        self.game_speed *= multiplier
