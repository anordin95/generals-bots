from typing import Any, TypeAlias

import numpy as np
import pygame

from generals.core.config import Dimension, Path
from generals.core.grid import NEUTRAL_ID
from generals.gui.properties import GuiMode, Properties

Color: TypeAlias = tuple[int, int, int]
FOG_OF_WAR: Color = (70, 73, 76)
NEUTRAL_CASTLE: Color = (128, 128, 128)
VISIBLE_PATH: Color = (200, 200, 200)
VISIBLE_MOUNTAIN: Color = (187, 187, 187)
BLACK: Color = (0, 0, 0)
WHITE: Color = (230, 230, 230)


class Renderer:
    def __init__(self, properties: Properties):
        """
        Initialize the pygame GUI
        """
        pygame.init()
        pygame.display.set_caption("Generals")
        pygame.key.set_repeat(500, 64)

        self.properties = properties

        ############
        # Surfaces #
        ############
        window_width = self.properties.display_grid_width + self.properties.right_panel_width
        window_height = self.properties.display_grid_height + 1

        width = Dimension.GUI_CELL_WIDTH.value
        height = Dimension.GUI_CELL_HEIGHT.value

        # Main window
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # Scoreboard
        self.right_panel = pygame.Surface((self.properties.right_panel_width, window_height))
        self.score_cols = {}
        for col in ["Player", "Army", "Land"]:
            size = (width, height)
            if col == "Player":
                size = (2 * width, height)
            self.score_cols[col] = [pygame.Surface(size) for _ in range(3)]

        self.info_panel = {
            "time": pygame.Surface((self.properties.right_panel_width / 2, height)),
            "speed": pygame.Surface((self.properties.right_panel_width / 2, height)),
        }
        # Game area and tiles
        self.game_area = pygame.Surface((self.properties.display_grid_width, self.properties.display_grid_height))
        self.tiles = [
            [
                pygame.Surface((Dimension.SQUARE_SIZE.value, Dimension.SQUARE_SIZE.value))
                for _ in range(self.properties.grid_width)
            ]
            for _ in range(self.properties.grid_height)
        ]

        self._mountain_img = pygame.image.load(str(Path.MOUNTAIN_PATH), "png").convert_alpha()
        self._general_img = pygame.image.load(str(Path.GENERAL_PATH), "png").convert_alpha()
        self._city_img = pygame.image.load(Path.CITY_PATH, "png").convert_alpha()

        self._font = pygame.font.Font(Path.FONT_PATH, self.properties.font_size)

    def render(self, agent_id_to_infos: dict[str, Any], current_time: int, fps=None):
        self.render_grid()
        self.render_stats(agent_id_to_infos, current_time)
        pygame.display.flip()
        if fps:
            self.properties.clock.tick(fps)

    def render_cell_text(
        self,
        cell: pygame.Surface,
        text: str,
        fg_color: Color = BLACK,
        bg_color: Color = WHITE,
    ):
        """
        Draw a text in the middle of the cell with given foreground and background colors

        Args:
            cell: cell to draw
            text: text to write on the cell
            fg_color: foreground color of the text
            bg_color: background color of the cell
        """
        center = (cell.get_width() // 2, cell.get_height() // 2)

        text_surface = self._font.render(text, True, fg_color)
        if bg_color:
            cell.fill(bg_color)
        cell.blit(text_surface, text_surface.get_rect(center=center))

    def render_stats(self, agent_id_to_infos: dict[str, Any], current_time: int):
        """
        Draw player stats and additional info on the right panel
        """
        names = self.properties.agent_ids
        player_stats = agent_id_to_infos
        gui_cell_height = Dimension.GUI_CELL_HEIGHT.value
        gui_cell_width = Dimension.GUI_CELL_WIDTH.value

        # Write names
        for i, name in enumerate(["Player"] + names):
            color: Color = self.properties.agent_id_to_color.get(name, WHITE)
            # add opacity to the color, where color is a Color(r,g,b)
            if name in self.properties.agent_fov and not self.properties.agent_fov[name]:
                color = tuple([int(0.5 * rgb) for rgb in color])  # type: ignore
            self.render_cell_text(self.score_cols["Player"][i], name, bg_color=color)

        # Write other columns
        for i, col in enumerate(["Army", "Land"]):
            self.render_cell_text(self.score_cols[col][0], col)
            for j, name in enumerate(names):
                if name in self.properties.agent_fov and not self.properties.agent_fov[name]:
                    color = (128, 128, 128)
                self.render_cell_text(
                    self.score_cols[col][j + 1],
                    str(player_stats[name][col.lower()]),
                    bg_color=WHITE,
                )

        # Blit each right_panel cell to the right_panel surface
        for i, col in enumerate(["Player", "Army", "Land"]):
            for j, cell in enumerate(self.score_cols[col]):
                rect_dim = (0, 0, cell.get_width(), cell.get_height())
                pygame.draw.rect(cell, BLACK, rect_dim, 1)

                position = ((i + 1) * gui_cell_width, j * gui_cell_height)
                if col == "Player":
                    position = (0, j * gui_cell_height)
                self.right_panel.blit(cell, position)

        info_text = {
            "time": f"Time: {str(current_time // 2) + ('.' if current_time % 2 == 1 else '')}",
            "speed": "Paused"
            if self.properties.gui_mode == GuiMode.REPLAY and self.properties.is_paused
            else f"Speed: {str(self.properties.game_speed)}x",
        }

        # Write additional info
        for i, key in enumerate(["time", "speed"]):
            self.render_cell_text(self.info_panel[key], info_text[key])

            rect_dim = (
                0,
                0,
                self.info_panel[key].get_width(),
                self.info_panel[key].get_height(),
            )
            pygame.draw.rect(self.info_panel[key], BLACK, rect_dim, 1)

            self.right_panel.blit(self.info_panel[key], (i * 2 * gui_cell_width, 3 * gui_cell_height))
        # Render right_panel on the screen
        self.screen.blit(self.right_panel, (self.properties.display_grid_width, 0))

    def render_grid(self):
        """
        Render the game grid
        """
        agents = self.properties.agent_ids
        # Maps of all owned and visible cells
        owned_map = np.zeros((self.properties.grid_height, self.properties.grid_width), dtype=bool)
        visible_map = np.zeros((self.properties.grid_height, self.properties.grid_width), dtype=bool)
        for agent in agents:
            ownership = self.properties.grid.owners[agent]
            owned_map = np.logical_or(owned_map, ownership)
            if self.properties.agent_fov[agent]:
                visibility = self.properties.grid.get_visibility(agent)
                visible_map = np.logical_or(visible_map, visibility)

        # Helper maps for not owned and invisible cells
        not_owned_map = np.logical_not(owned_map)
        invisible_map = np.logical_not(visible_map)

        # Draw background of visible owned squares
        for agent in agents:
            ownership = self.properties.grid.owners[agent]
            visible_ownership = np.logical_and(ownership, visible_map)
            self.draw_channel(visible_ownership, self.properties.agent_id_to_color[agent])

        # Draw visible generals
        visible_generals = np.logical_and(self.properties.grid.generals, visible_map)
        self.draw_images(visible_generals, self._general_img)

        # Draw background of visible but not owned squares
        visible_not_owned = np.logical_and(visible_map, not_owned_map)
        self.draw_channel(visible_not_owned, WHITE)

        # Draw background of squares in fog of war
        self.draw_channel(invisible_map, FOG_OF_WAR)

        # Draw background of visible mountains
        visible_mountain = np.logical_and(self.properties.grid.mountains, visible_map)
        self.draw_channel(visible_mountain, VISIBLE_MOUNTAIN)

        # Draw mountains (even if they are not visible)
        self.draw_images(self.properties.grid.mountains, self._mountain_img)

        # Draw background of visible neutral cities
        visible_cities = np.logical_and(self.properties.grid.cities, visible_map)
        visible_cities_neutral = np.logical_and(visible_cities, self.properties.grid.owners[NEUTRAL_ID])
        self.draw_channel(visible_cities_neutral, NEUTRAL_CASTLE)

        # Draw invisible cities as mountains
        invisible_cities = np.logical_and(self.properties.grid.cities, invisible_map)
        self.draw_images(invisible_cities, self._mountain_img)

        # Draw visible cities
        self.draw_images(visible_cities, self._city_img)

        # Draw nonzero army counts on visible squares
        visible_army = self.properties.grid.armies * visible_map
        visible_army_indices = self.channel_to_indices(visible_army)
        for i, j in visible_army_indices:
            self.render_cell_text(
                self.tiles[i][j],
                str(int(visible_army[i, j])),
                fg_color=WHITE,
                bg_color=None,  # Transparent background
            )

        # Blit tiles to the self.game_area
        square_size = Dimension.SQUARE_SIZE.value
        for i, j in np.ndindex(self.properties.grid_height, self.properties.grid_width):
            self.game_area.blit(self.tiles[i][j], (j * square_size, i * square_size))
        self.screen.blit(self.game_area, (0, 0))

    def channel_to_indices(self, channel: np.ndarray) -> np.ndarray:
        """
        Returns a list of indices of cells with non-zero values from specified a channel.
        """
        return np.argwhere(channel != 0)

    def draw_channel(self, channel: np.ndarray, color: Color):
        """
        Draw background and borders (left and top) for grid tiles of a given channel
        """
        square_size = Dimension.SQUARE_SIZE.value
        for i, j in self.channel_to_indices(channel):
            self.tiles[i][j].fill(color)
            pygame.draw.line(self.tiles[i][j], BLACK, (0, 0), (0, square_size), 1)
            pygame.draw.line(self.tiles[i][j], BLACK, (0, 0), (square_size, 0), 1)

    def draw_images(self, channel: np.ndarray, image: pygame.Surface):
        """
        Draw images on grid tiles of a given channel
        """
        for i, j in self.channel_to_indices(channel):
            self.tiles[i][j].blit(image, (3, 2))
