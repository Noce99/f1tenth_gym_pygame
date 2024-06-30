"""
Rendering engine for F1Tenth gym env based on pygame
Author: Enrico Mannocci
"""
# pygame
import pygame

# other
import numpy as np
import yaml
import cv2

# zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31


class EnvRenderer:
    """
    A window class inherited from pyglet.window.Window, handles the camera/projection interaction,
    resizing window, and rendering the environment
    """

    def __init__(self, width, height):
        """
        Class constructor

        Args:
            width (int): width of the window
            height (int): height of the window

        Returns:
            None
        """

        # initialize camera values
        self.left = -width / 2
        self.right = width / 2
        self.bottom = -height / 2
        self.top = height / 2
        self.zoom_level = 1.2
        self.zoomed_width = width
        self.zoomed_height = height

        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('NutMade')
        self.half_width = width // 2
        self.half_height = height // 2
        self.font = pygame.font.Font(None, 74)

        # current env agent poses, (num_agents, 3), columns are (x, y, theta)
        self.poses = None

        # current env agent vertices, (num_agents, 4, 2), 2nd and 3rd dimensions are the 4 corners in 2D
        self.vertices = None

        # background image
        self.track_image = None
        self.image_config = {
            "resolution": None,  # m/px
            "origin": [None, None, None],
            "negate": None,
            "occupied_thresh": None,
            "free_thresh": None
        }
        self.m_focus = (None, None)
        self.track_surf = None
        self.track_surf_rect = None

        # controls
        self.wasd_tick_move = 5.0  # m

    def m_to_pixel_image(self, coordinates):
        x, y = coordinates
        return (int((x - self.image_config["origin"][0]) / self.image_config["resolution"]),
                int((y - self.image_config["origin"][1]) / self.image_config["resolution"]))

    def pixel_image_to_m(self, coordinates):
        x, y = coordinates
        return (x * self.image_config["resolution"] + self.image_config["origin"][0],
                y * self.image_config["resolution"] + self.image_config["origin"][1])

    def m_to_pixel_window(self, coordinates):
        x, y = coordinates
        return (int((x - self.m_focus[0]) / self.image_config["resolution"]) + self.half_width,
                int((y - self.m_focus[1]) / self.image_config["resolution"]) + self.half_height)

    """
    def pixel_window_to_m(self, coordinates):
        x, y = coordinates
        return (x ,
                y )
    """

    def update_map(self, map_path, map_ext):
        """
        Update the map being drawn by the renderer.
        Converts image to a list of 3D points representing each obstacle pixel in the map.

        Args:
            map_path (str): absolute path to the map without extensions
            map_ext (str): extension for the map image file

        Returns:
            None
        """
        image_path = f"{map_path}{map_ext}"
        self.track_image = cv2.imread(image_path)
        self.track_image = np.rot90(self.track_image, k=3)
        print(f"Track Image Shape: {self.track_image.shape}")
        yaml_path = f"{map_path}.yaml"
        with open(yaml_path, "r") as file:
            yaml_content = yaml.safe_load(file)
            for key in yaml_content:
                self.image_config[key] = yaml_content[key]
        self.m_focus = (0.0, 0.0)
        self.get_map_given_the_center_in_m(self.m_focus)

    def get_map_given_the_center_in_m(self, center):
        focus_in_px = self.m_to_pixel_image((center[0], center[1]))

        left = focus_in_px[0] - self.half_width
        left = max(left, 0)
        add_on_the_left = (self.half_width - (focus_in_px[0] - left)) / 2

        right = focus_in_px[0] + self.half_width
        right = min(right, self.track_image.shape[0])
        add_on_the_right = (self.half_width - (right - focus_in_px[0])) / 2

        top = focus_in_px[1] - self.half_height
        top = max(top, 0)
        add_on_the_top = (self.half_height - (focus_in_px[1] - top)) / 2

        bottom = focus_in_px[1] + self.half_height
        bottom = min(bottom, self.track_image.shape[1])
        add_on_the_bottom = (self.half_height - (bottom - focus_in_px[1])) / 2

        self.track_surf = pygame.surfarray.make_surface(self.track_image[left:right, top:bottom])
        self.track_surf_rect = self.track_surf.get_rect()
        self.track_surf_rect.center = (int(self.half_width + add_on_the_left - add_on_the_right),
                                       int(self.half_height + add_on_the_top - add_on_the_bottom))

    def render_text_about_laps_and_time(self, obs):
        text_surface = self.font.render(f"Lap Time:  {obs['lap_times'][0]:.2f},"
                                        f" Ego Lap Count: {obs['lap_counts'][obs['ego_idx']]:.0f}",
                                        True, (255, 155, 0))
        text_rect = text_surface.get_rect()
        text_rect.center = (self.half_width * 2 - text_rect.width / 2,
                            self.half_height * 2 - text_rect.height / 2)
        self.screen.blit(text_surface, text_rect)

    def update_obs(self, obs, render_callbacks):
        """
        Updates the renderer with the latest observation from the gym environment, including the agent poses,
        and the information text.

        Args:
            render_callbacks:
            obs (dict): observation dict from the gym env

        Returns:
            None
        """
        ego_idx = obs['ego_idx']
        poses_x = obs['poses_x']
        poses_y = obs['poses_y']
        poses_theta = obs['poses_theta']
        num_agents = len(poses_x)

        # THERE WE CLEAN THE SCREEN
        self.screen.fill((0, 0, 0))

        # THERE WE RENDER THE MAP
        self.screen.blit(self.track_surf, self.track_surf_rect)

        # THERE WE RENDER THE OPTIONAL STUFF
        for render_callback in render_callbacks:
            render_callback(self)

        # THERE WE RENDER THE VEHICLES
        for agent_i in range(num_agents):
            if agent_i == ego_idx:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            pygame.draw.circle(self.screen,
                               color,
                               self.m_to_pixel_window((poses_x[0], poses_y[0])),
                               5)

        # LET'S PUT A BLUE CIRCLE AT THE ORIGIN
        pygame.draw.circle(self.screen,
                           (0, 0, 255),
                           (self.half_width, self.half_height),
                           5)

        # LET'S RENDER THE TEXT GIVING INFORMATION ABOUT THE LAP TIME
        self.render_text_about_laps_and_time(obs)

        # FINALLY WE RENDER THE SCENE
        pygame.display.update()

    def check_keys(self):
        should_quit = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.m_focus = (self.m_focus[0], self.m_focus[1] - self.wasd_tick_move)
                    self.get_map_given_the_center_in_m(self.m_focus)
                elif event.key == pygame.K_a:
                    self.m_focus = (self.m_focus[0] - self.wasd_tick_move, self.m_focus[1])
                    self.get_map_given_the_center_in_m(self.m_focus)
                elif event.key == pygame.K_s:
                    self.m_focus = (self.m_focus[0], self.m_focus[1] + self.wasd_tick_move)
                    self.get_map_given_the_center_in_m(self.m_focus)
                elif event.key == pygame.K_d:
                    self.m_focus = (self.m_focus[0] + self.wasd_tick_move, self.m_focus[1])
                    self.get_map_given_the_center_in_m(self.m_focus)
        return should_quit
