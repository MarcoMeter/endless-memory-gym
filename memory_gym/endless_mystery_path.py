import gymnasium as gym
import math
import numpy as np
import os
import pygame

from argparse import ArgumentParser
from gymnasium import spaces
from memory_gym.environment import CustomEnv
from memory_gym.character_controller import CharacterController
from memory_gym.pygame_assets import EndlessMysteryPath, draw_column_tile_surface
from pygame._sdl2 import Window, Texture, Renderer

SCALE = 0.25

class EndlessMysteryPathEnv(CustomEnv):
    metadata = {
        "render_modes": ["rgb_array", "debug_rgb_array"],
        "render_fps": 25,
    }

    default_reset_parameters = {
                "max_steps": 1024,
                "agent_scale": 1.0 * SCALE,
                "agent_speed": 12.0 * SCALE,
                "show_origin": False,
                "show_past_path": True,
                "show_background": False,
                "show_stamina": False,
                "visual_feedback": True,
                "camera_offset_scale": 5.0, # must be between 0.0 and 5.5
                "stamina_level": 20,
                "reward_fall_off": 0.0,
                "reward_path_progress": 0.1,
                "reward_step": 0.0
            }

    def process_reset_params(reset_params):
        """Compares the provided reset parameters to the default ones. It asserts whether false reset parameters were provided.
        Missing reset parameters are filled with the default ones.

        Arguments:
            reset_params {dict} -- Provided reset parameters that are to be validated and completed

        Returns:
            dict -- Returns a complete and valid dictionary comprising the to be used reset parameters.
        """
        cloned_params = EndlessMysteryPathEnv.default_reset_parameters.copy()
        if reset_params is not None:
            for k, v in reset_params.items():
                assert k in cloned_params.keys(), "Provided reset parameter (" + str(k) + ") is not valid. Check spelling."
                cloned_params[k] = v
        return cloned_params

    def __init__(self, render_mode = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        if render_mode is None:
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            pygame.display.set_caption("Environment")

        # Init PyGame screen
        pygame.init()
        self.screen_dim = int(336 * SCALE)
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim), pygame.NOFRAME)
        self.clock = pygame.time.Clock()
        if render_mode is None:
            pygame.event.set_allowed(None)

        # Init debug window
        self.debug_window = None

        # Setup observation and action space
        self.action_space = spaces.Discrete(4)
        self.observation_space= spaces.Box(
                    low = 0.0,
                    high = 1.0,
                    shape = [self.screen_dim, self.screen_dim, 3],
                    dtype = np.float32)
        
        # Environment members
        self.rotated_agent_surface, self.rotated_agent_rect = None, None
        self.grid_dim = 7
        self.tile_dim = self.screen_dim // self.grid_dim

        # Scrolling background members
        self.tile_coloumn_surface = draw_column_tile_surface(self.tile_dim, self.grid_dim)
        self.num_coloums = math.ceil(self.screen_dim / self.tile_dim) + 2
        
        # Stamina bar surface
        self.stamina_surface = pygame.Surface((16 * SCALE, self.screen_dim))

    def _draw_past_path(self):
        # Get normalized agent x position
        x = self.normalized_agent_position[0] - 1
        if x < 0:
            return
        depth = int(self.reset_params["camera_offset_scale"])
        past_x = max(0, x - depth)
        current_node = self.current_node.previous_node
        # Iterate over previous nodes
        while x >= past_x and x >= 0:
            if current_node is None:
                break
            x, y = current_node.x, current_node.y
            draw_x = (x) * self.tile_dim - self.camera_x
            # Draw previous node
            pygame.draw.rect(self.screen, (255, 255, 255), (draw_x, y * self.tile_dim, self.tile_dim, self.tile_dim))
            # Draw border
            pygame.draw.rect(self.screen, (210, 210, 210), (draw_x, y * self.tile_dim, self.tile_dim, self.tile_dim), 1)
            if x == past_x:
                break
            current_node = current_node.previous_node

    def _draw_surfaces(self, surfaces):
        # Draw scrolling background
        if self.reset_params["show_background"]:
            for i in range(self.num_coloums):
                self.screen.blit(self.tile_coloumn_surface, (i * self.tile_dim + self.bg_scroll - self.tile_dim, 0))
        else:
            self.screen.fill((0, 0, 0))
        # Draw past path
        if self.reset_params["show_past_path"]:
            self._draw_past_path()
        # Draw origin
        if self.reset_params["show_origin"] and self.normalized_agent_position[0] < 5 and not self.reset_params["show_origin"]:
            pygame.draw.rect(self.screen, (0, 255, 0), (self.start.x * self.tile_dim - self.camera_x, self.start.y * self.tile_dim, self.tile_dim, self.tile_dim))
            pygame.draw.rect(self.screen, (210, 210, 210), (self.start.x * self.tile_dim - self.camera_x, self.start.y * self.tile_dim, self.tile_dim, self.tile_dim), int(4 * SCALE))
        # Draw remaining surfaces
        for surface in surfaces:
            if surface[0] is not None:
                self.screen.blit(surface[0], surface[1])
        # Draw visual feedback
        if self.reset_params["visual_feedback"]:
            self.screen.blit(self.fall_off_surface, self.fall_off_rect)
        pygame.display.flip()

    def _build_debug_surface(self):
        surface = pygame.Surface((336 * SCALE, 336 * SCALE))
        surface.fill(0)
        # Draw scrolling background
        if self.reset_params["show_background"]:
            for i in range(self.num_coloums):
                surface.blit(self.tile_coloumn_surface, (i * self.tile_dim + self.bg_scroll - self.tile_dim, 0))
        surface.blit(self.endless_path.surface, (-self.camera_x, 0))
        if self.rotated_agent_surface:
            surface.blit(self.rotated_agent_surface, (self.agent_draw_x, self.rotated_agent_rect.y))
        else:
            surface.blit(self.agent.surface, self.agent.rect)
        surface.blit(self.fall_off_surface, self.fall_off_rect)
        # Draw stamina bar
        surface.blit(self.stamina_surface, self.stamina_position)
        return pygame.transform.scale(surface, (336, 336))

    def _normalize_agent_position(self, agent_position):
            return (agent_position[0] // self.tile_dim, agent_position[1] // self.tile_dim)

    def reset(self, seed = None, return_info = True, options = None):
        super().reset(seed=seed)
        self.current_seed = seed
        self.t = 0

        # Check reset parameters for completeness and errors
        self.reset_params = EndlessMysteryPathEnv.process_reset_params(options)
        self.max_episode_steps = self.reset_params["max_steps"]

        # Track all rewards during one episode
        self.episode_rewards = []

        # Setup initial path segments
        self.endless_path = EndlessMysteryPath(self.grid_dim, self.grid_dim, self.tile_dim, self.np_random, num_initial_segments=3)
        self.start = self.endless_path.path[0][0]

        # Fall off surface to indicate that the agent lost the path
        dim = 40 * SCALE
        self.fall_off_surface = pygame.Surface((dim, dim))
        self.fall_off_rect = self.fall_off_surface.get_rect()
        self.fall_off_surface.fill(0)
        self.fall_off_surface.set_colorkey(0)
        pygame.draw.line(self.fall_off_surface, (255, 0, 0), (0, 0), (dim - 1, dim - 1), int(12 * SCALE))
        pygame.draw.line(self.fall_off_surface, (255, 0, 0), (dim - 1, 0), (0, dim - 1), int(12 * SCALE))
        self.fall_off_surface.set_alpha(0)

        # Init camera x position for the scrolling effect
        camera_offset_scale = max(0, min(5.5, self.reset_params["camera_offset_scale"]))
        self.camera_offset = -self.tile_dim * camera_offset_scale
        self.camera_x = self.camera_offset
        self.bg_scroll = 0

        # Setup the agent
        self.agent = CharacterController(self.reset_params["agent_speed"], self.reset_params["agent_scale"], 270)
        self.rotated_agent_surface, self.rotated_agent_rect = self.agent.get_rotated_sprite(270)
        # Place the agent on the path's starting position
        self.agent.rect.center = (self.start.x * self.tile_dim + self.agent.radius, self.start.y * self.tile_dim + self.agent.radius)
        self.agent_draw_x = self.agent.rect.topleft[0] - self.camera_offset
        self.normalized_agent_position = self._normalize_agent_position(self.agent.rect.center)
        self.current_node = self.endless_path.path[0][0]
        self.is_off_path = False
        self.current_segment = 0
        self.num_fails = 0
        self.fall_off_locations = []
        self.stamina = self.reset_params["stamina_level"]
        self.max_x_reached = 0
        self.tiles_visited = 0

        # Reset stamina bar
        self.max_visible_stamina = self.reset_params["stamina_level"]
        self.stamina_surface.fill((0, 255, 0))
        self.stamina_position = (self.screen_dim - 16 * SCALE, 0)
        height = int(self.screen_dim * (1 - (max(self.stamina, self.max_visible_stamina) / self.max_visible_stamina)))
        pygame.draw.rect(self.stamina_surface, (255, 0, 0),(0, 0, 16 * SCALE, height))

        # Draw
        if self.reset_params["show_stamina"]:
            self._draw_surfaces([(self.rotated_agent_surface, (self.agent_draw_x, self.rotated_agent_rect.y)), (self.stamina_surface, self.stamina_position)])
        else:
            self._draw_surfaces([(self.rotated_agent_surface, (self.agent_draw_x, self.rotated_agent_rect.y))])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, {}

    def step(self, action):
        # Map single discrete action to multi-discrete action
        if action == 0:
            action = [0, 0]
        elif action == 1:
            action = [2, 0]
        elif action == 2:
            action = [0, 1]
        elif action == 3:
            action = [0, 2]

        reward = 0
        done = False
        previous_max_x = self.max_x_reached

        # Move the agent's controlled character and scroll the camera and the background
        if not self.is_off_path:
            self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step(action)
            # Update camera position based on agent x velocity
            self.camera_x += self.agent.velocity.x
            # Update background scroll based on agent x velocity
            self.bg_scroll -= self.agent.velocity.x
            if abs(self.bg_scroll) >= self.tile_dim:
                # bg scroll cannot always be reset to 0 due to the different velocity of moving diagonally
                remainder = abs(self.bg_scroll) % abs(self.agent.velocity.x)
                sign = self.bg_scroll / abs(self.bg_scroll)
                self.bg_scroll = remainder * sign
        else:
            self.agent.rect.center = (self.start.x * self.tile_dim + self.agent.radius, self.start.y * self.tile_dim + self.agent.radius)
            self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step([0, 0])
            # Reset camera x position
            self.camera_x = self.camera_offset
            # Reset background scroll
            self.bg_scroll = 0

        # Normalize the agent's position to run the path logic
        self.normalized_agent_position = self._normalize_agent_position(self.agent.rect.center)

        # Determine the current segment of the path based on the the agent's normalized x position
        self.current_segment = int(self.normalized_agent_position[0]) // self.endless_path.segment_length
        nodes = self.endless_path.path[self.current_segment]

        # Generate new path segment
        if self.current_segment > self.endless_path.num_segments - 2:
            self.endless_path.add_path_segment()
            self.endless_path.gen_surface(self.tile_dim)

        # Check whether the agent fell off the path
        on_path = False
        for node in nodes:
            if self.normalized_agent_position == (node.x, node.y):
                on_path = True
                self.current_node = node
                if not node.reward_visited and not (node.x, node.y) == self.start:
                    # Reward the agent for reaching a tile that it has not visisted before
                    reward += self.reset_params["reward_path_progress"]
                    self.tiles_visited += 1
                    node.reward_visited = True
                if not node.stamina_visited and not (node.x, node.y) == self.start:
                    # Reset the agent's stamina for reaching a tile that it has not visited before
                    self.stamina = self.reset_params["stamina_level"]
                    node.stamina_visited = True
                break

        if not on_path:
            reward += self.reset_params["reward_fall_off"]
            self.num_fails += 1
            if self.reset_params["visual_feedback"]:
                self.fall_off_surface.set_alpha(255)
            self.is_off_path = True
            # Terminate the episode if the agent fell off the path too soon
            if self.normalized_agent_position[0] < self.max_x_reached:
                done = True
            # Terminate the episode if the agent fell off the path at the location where it fell off before
            else:
                fall_off_pos_found = False
                for fall_off_pos in self.fall_off_locations:
                    if self.normalized_agent_position == fall_off_pos:
                        done = True
                        fall_off_pos_found = True
                        break
                if not fall_off_pos_found:
                    self.fall_off_locations.append(self.normalized_agent_position)

            # Reset all visited tiles so that the agent can gain stamina again
            for segment in self.endless_path.path:
                for node in segment:
                    node.stamina_visited = False
                    self.stamina = self.reset_params["stamina_level"]
        else:
            self.fall_off_surface.set_alpha(0)
            self.is_off_path = False
        self.fall_off_rect.center = (self.agent.rect.center[0] - self.camera_x, self.agent.rect.center[1])

        # Reset fall off locations if the agent reached a new maximum x position
        if self.max_x_reached > previous_max_x:
            self.fall_off_locations = []

        # Emit a reward signal for every single step
        reward += self.reset_params["reward_step"]

        # Time limit (agent ran out of stamina)
        self.stamina -= 1
        if self.stamina == 0:
            done = True
        # Upper time limit
        # Time limit
        self.t += 1
        if self.t == self.max_episode_steps:
            done = True
        
        # Determine the maximum normalized x position reached by the agent
        if self.normalized_agent_position[0] > self.max_x_reached and on_path:
            self.max_x_reached = self.normalized_agent_position[0]

        # Track all rewards
        self.episode_rewards.append(reward)

        if done:
            info = {
                "reward": sum(self.episode_rewards),
                "length": len(self.episode_rewards),
                "num_fails": self.num_fails,
                "max_x": self.max_x_reached,
                "tiles_visited": self.tiles_visited
            }
        else:
            info = {}

        # Update the stamina bar
        self.stamina_surface.fill((0, 255, 0))
        height = int(self.screen_dim * (1 - (min(self.stamina, self.max_visible_stamina) / self.max_visible_stamina)))
        pygame.draw.rect(self.stamina_surface, (255, 0, 0),(0, 0, 16 * SCALE, height))
        # Draw
        if self.reset_params["show_stamina"]:
            self._draw_surfaces([(self.rotated_agent_surface, (self.agent_draw_x, self.rotated_agent_rect.y)), (self.stamina_surface, self.stamina_position)])
        else:
            self._draw_surfaces([(self.rotated_agent_surface, (self.agent_draw_x, self.rotated_agent_rect.y))])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, reward, done, False, info

    def render(self):
        if self.render_mode is not None:
            if self.render_mode == "rgb_array":
                self.clock.tick(EndlessMysteryPathEnv.metadata["render_fps"])
                return np.fliplr(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8), 3)) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)
            elif self.render_mode == "debug_rgb_array":
                # Create debug window if it doesn't exist yet
                if self.debug_window is None:
                    self.debug_window = Window(size = (336, 336))
                    self.debug_window.show()
                    self.renderer = Renderer(self.debug_window)
                
                self.debug_window.title = "seed " + str(self.current_seed)
                self.clock.tick(EndlessMysteryPathEnv.metadata["render_fps"])

                debug_surface = self._build_debug_surface()
                texture = Texture.from_surface(self.renderer, debug_surface)
                texture.draw(dstrect=(0, 0))
                self.renderer.present()
                return np.fliplr(np.rot90(pygame.surfarray.array3d(self.renderer.to_surface()).astype(np.uint8), 3))

    def close(self):
        if self.debug_window is not None:
            self.debug_window.destroy()
        pygame.quit()

def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, help="The to be used seed for the environment's random number generator.", default=0)
    options = parser.parse_args()

    env = EndlessMysteryPathEnv(render_mode = "debug_rgb_array")
    reset_params = {}
    seed = options.seed
    vis_obs, reset_info = env.reset(seed = seed, options = reset_params)
    img = env.render()
    done = False

    while not done:
        action = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = 2
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = 3
        if keys[pygame.K_PAGEDOWN] or keys[pygame.K_PAGEUP]:
            if keys[pygame.K_PAGEUP]:
                seed += 1
            if keys[pygame.K_PAGEDOWN]:
                if not seed <= 0:
                    seed -= 1
            vis_obs, reset_info = env.reset(seed = seed, options = reset_params)
            img = env.render()
        vis_obs, reward, done, truncation, info = env.step(action)
        img = env.render()

        # Process event-loop
        for event in pygame.event.get():
        # Quit
            if event.type == pygame.QUIT:
                done = True

    print("episode reward: " + str(info["reward"]))
    print("episode length: " + str(info["length"]))
    print("num fails: " + str(info["num_fails"]))
    print("max x: " + str(info["max_x"]))
    print("tiles visited: " + str(info["tiles_visited"]))

    env.close()
    exit()

if __name__ == "__main__":
    main()