# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Rendering engine for f1tenth gym env based on pyglet and OpenGL
Author: Hongrui Zheng
"""

# opengl stuff
import pyglet
from pyglet.gl import *

# pygame
import pygame

# other
import numpy as np
from PIL import Image
import yaml
import cv2

# helpers
from f110_gym.envs.collision_models import get_vertices

# zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31


class EnvRenderer():
    """
    A window class inherited from pyglet.window.Window, handles the camera/projection interaction, resizing window, and rendering the environment
    """

    def __init__(self, width, height, *args, **kwargs):
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

        # Inityalize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('NutMade')
        self.half_width = width // 2
        self.half_height = height // 2


        # current env agent poses, (num_agents, 3), columns are (x, y, theta)
        self.poses = None

        # current env agent vertices, (num_agents, 4, 2), 2nd and 3rd dimensions are the 4 corners in 2D
        self.vertices = None

        # current score label
        """
        self.score_label = pyglet.text.Label(
            'Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}'.format(
                laptime=0.0, count=0.0),
            font_size=36,
            x=0,
            y=-800,
            anchor_x='center',
            anchor_y='center',
            # width=0.01,
            # height=0.01,
            color=(255, 255, 255, 255),
            batch=self.batch)
        """

        # stuff for retrocompatibylity
        self.batch = []

        # background image
        self.track_image = None
        self.image_rect = None
        self.image_config = {
            "resolution": None,
            "origin": [None, None, None],
            "negate": None,
            "occupied_thresh": None,
            "free_thresh": None
        }
        self.m_focus = (None, None)

    def m_to_pixel_image(self, coords):
        x, y = coords
        return (int((x - self.image_config["origin"][0]) / self.image_config["resolution"]),
                int((y - self.image_config["origin"][1]) / self.image_config["resolution"]))

    def pixel_image_to_m(self, coords):
        x, y = coords
        return (x * self.image_config["resolution"] + self.image_config["origin"][0],
                y * self.image_config["resolution"] + self.image_config["origin"][1])

    def update_map(self, map_path, map_ext):
        """
        Update the map being drawn by the renderer. Converts image to a list of 3D points representing each obstacle pixel in the map.

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

    def get_map_given_the_center_in_m(self, center):
        focus_in_px = self.m_to_pixel_image((poses_x[0], poses_y[0]))
        left = focus_in_px[0]-self.half_width
        left = min(left, 0)
        right = focus_in_px[0]+self.half_width
        right = max(right, self.track_image.shape[0])
        top = focus_in_px[1]-self.half_height
        top = min(top, 0)
        bottom = focus_in_px[1]+self.half_height
        bottom = max(bottom, self.track_image.shape[1])

        surf = pygame.surfarray.make_surface(self.track_image[
                         focus_in_px[0]-self.half_width:focus_in_px[0]+self.half_width,
                         focus_in_px[1]-self.half_height:focus_in_px[1]+self.half_height])


    def update_obs(self, obs):
        """
        Updates the renderer with the latest observation from the gym environment, including the agent poses, and the information text.

        Args:
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
        focus_in_px = self.m_to_pixel_image((poses_x[0], poses_y[0]))# self.m_focus)
        surf = pygame.surfarray.make_surface(self.track_image[
                         focus_in_px[0]-self.half_width:focus_in_px[0]+self.half_width,
                         focus_in_px[1]-self.half_height:focus_in_px[1]+self.half_height])
        self.screen.blit(surf,
                         (0, 0))

        # THERE WE RENDER THE VEHICLES
        for agent_i in range(num_agents):
            pygame.draw.circle(self.screen,
                               (255, 0, 0),
                               self.m_to_pixel_image((poses_x[0], poses_y[0])),
                               5)

        # LET'S PUT A BLUE CIRCLE AT THE ORIGIN
        pygame.draw.circle(self.screen,
                           (0, 0, 255),
                           (self.half_width, self.half_height),
                           5)

        # FINALLY WE RENDER THE SCENE
        pygame.display.update()


        # TEST
        print(self.m_to_pixel_image((0, 0)))

    def check_keys(self):
        quit = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
        return quit