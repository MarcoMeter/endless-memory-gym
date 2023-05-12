import gymnasium as gym
import math
import numpy as np
import os
import pygame

from argparse import ArgumentParser
from gymnasium import  spaces
from memory_gym.character_controller import CharacterController
from memory_gym.pygame_assets import Coin, Exit, GridPositionSampler, Spotlight, get_tiled_background_surface
from pygame._sdl2 import Window, Texture, Renderer

SCALE = 0.25

class SearingSpotlightsEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "debug_rgb_array"],
        "render_fps": 25,
    }

    default_reset_parameters = {
                # Spotlight parameters
                "max_steps": 256,
                "initial_spawns": 4,
                "num_spawns": 30,
                "initial_spawn_interval": 30,
                "spawn_interval_threshold": 10,
                "spawn_interval_decay": 0.95,
                "spot_min_radius": 30.0 * SCALE,
                "spot_max_radius": 55.0 * SCALE,
                "spot_min_speed": 0.0025,
                "spot_max_speed": 0.0075,
                "spot_damage": 1.0,
                "visual_feedback": True,
                "black_background": False,
                "hide_chessboard": False,
                # Light Parameters
                "light_dim_off_duration": 0,
                "light_threshold": 255,
                # Coin Parameters
                "num_coins": [1],
                "coin_scale": 1.5 * SCALE,
                "coins_visible": False,
                # Exit Parameters
                "use_exit": True,
                "exit_scale": 2.0 * SCALE,
                "exit_visible": False,
                # Agent Parameters
                "agent_speed": 12.0 * SCALE,
                "agent_health": 5,
                "agent_scale": 1.0 * SCALE,
                "agent_visible": False,
                "sample_agent_position": True,
                # Reward Function
                "reward_inside_spotlight": 0.0,
                "reward_outside_spotlight": 0.0,
                "reward_death": 0.0,
                "reward_exit": 1.0,
                "reward_max_steps": 0.0,
                "reward_coin": 0.25,
            }

    def process_reset_params(reset_params):
        """Compares the provided reset parameters to the default ones. It asserts whether false reset parameters were provided.
        Missing reset parameters are filled with the default ones.

        Arguments:
            reset_params {dict} -- Provided reset parameters that are to be validated and completed

        Returns:
            dict -- Returns a complete and valid dictionary comprising the to be used reset parameters.
        """
        cloned_params = SearingSpotlightsEnv.default_reset_parameters.copy()
        if reset_params is not None:
            for k, v in reset_params.items():
                assert k in cloned_params.keys(), "Provided reset parameter (" + str(k) + ") is not valid. Check spelling."
                cloned_params[k] = v
        return cloned_params

    def __init__(self, render_mode = None) -> None:
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
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space= spaces.Box(
                    low = 0.0,
                    high = 1.0,
                    shape = [self.screen_dim, self.screen_dim, 3],
                    dtype = np.float32)

        # Environment members
        # Tiled background surface
        self.blue_background_surface = get_tiled_background_surface(self.screen, self.screen_dim, (0, 0, 255), SCALE)
        self.red_background_surface = get_tiled_background_surface(self.screen, self.screen_dim, (255, 0, 0), SCALE)

        # Spotlight surface
        self.spotlight_surface = pygame.Surface((self.screen_dim, self.screen_dim))
        self.spotlight_surface.fill(0)
        self.spotlight_surface.set_colorkey((255, 0, 0))

        # Agent boundaries
        self.walkable_rect = pygame.Rect(0, 16 * SCALE, self.screen_dim, self.screen_dim - 16 * SCALE)

        # Init grid spawner
        self.grid_sampler = GridPositionSampler(self.screen_dim)
        # self.grid_sampler = GridPositionSampler(self.screen_dim - 16 * SCALE, self.screen_dim // 24)

        self.rotated_agent_surface, self.rotated_agent_rect = None, None

    def _compute_spawn_intervals(self, reset_params) -> list:
        intervals = []
        initial = reset_params["initial_spawn_interval"]
        for i in range(reset_params["num_spawns"]):
            intervals.append(int(initial + reset_params["spawn_interval_threshold"]))
            initial = initial * math.pow(reset_params["spawn_interval_decay"], 1)
        return intervals

    def _draw_surfaces(self, surfaces):
        # Draw all surfaces
        for surface in surfaces:
            if surface[0] is not None:
                self.screen.blit(surface[0], surface[1])
        pygame.display.flip()

    def _build_debug_surface(self):
        surface = pygame.Surface((336 * SCALE, 336 * SCALE))
        # Create coin surface
        coin_surface = pygame.Surface((self.screen_dim, self.screen_dim))
        coin_surface.fill(255)
        coin_surface.set_colorkey(255)
        for coin in self.coins:
            coin.draw(coin_surface)

        # Gather surfaces
        surfs = [(self.bg, (0, 0)), (self.spotlight_surface, (0, 0)), (self.exit.surface, self.exit.rect),
                (coin_surface, (0, 0))]
        # Retrieve the rotated agent surface or the original one
        if self.rotated_agent_surface is not None:
            surfs.append((self.rotated_agent_surface, self.rotated_agent_rect))
        else:
            surfs.append(self.agent.get_rotated_sprite(0))
        surfs.append((self.top_bar_surface, (0, 0)))
        # Blit all surfaces
        for surf, rect in surfs:
            if surf is not None:
                surface.blit(surf, rect)

        return pygame.transform.scale(surface, (336, 336))

    def _step_spotlight_task(self):
        reward = 0.0
        done = False
        # Spawn spotlights
        self.spawn_timer += 1
        if self.spawn_intervals:
            if self.spawn_timer >= self.spawn_intervals[0]:
                self.spotlights.append(Spotlight(self.screen_dim, self.np_random.integers(self.reset_params["spot_min_radius"], self.reset_params["spot_max_radius"] + 1),
                                                            self.np_random.uniform(self.reset_params["spot_min_speed"], self.reset_params["spot_max_speed"]), self.np_random, 
                                                            self.reset_params["black_background"]))
                self.spawn_intervals.pop()
                self.spawn_timer = 0

        # Draw spotlights and check whether the agent is visible or not
        self.spotlight_surface.fill(0)
        spotlight_hit = 0
        for spot in self.spotlights:        
            # Remove spotlights that finished traversal
            if spot.done:
                self.spotlights.remove(spot)
            # Draw spotlight and measure distance to agent
            else:
                spot.draw(self.spotlight_surface)
                if spot.is_agent_inside(self.agent):
                    spotlight_hit += 1

        if spotlight_hit > 0:
            self.current_agent_health -= self.reset_params["spot_damage"]
            reward += self.reset_params["reward_inside_spotlight"]
            width = int(self.screen_dim // 2 * (1 - self.current_agent_health / self.agent_health))
            pygame.draw.rect(self.top_bar_surface, (255, 0, 0), (0, 0, width, 16 * SCALE))
            # Render the background tiles in red if visual feedback is desired
            if self.reset_params["visual_feedback"]:
                bg = self.red_background_surface
            else:
                bg = self.blue_background_surface
        else:
            bg = self.blue_background_surface
            reward += self.reset_params["reward_outside_spotlight"]

        if self.reset_params["black_background"]:
            bg.fill(0)

        # Determine done
        if self.current_agent_health <= 0:
            done = True
            reward += self.reset_params["reward_death"]

        return reward, done, bg

    def _process_spawn_pos(self, spawn_pos, offset = 30):
        x = spawn_pos[0]
        y = spawn_pos[1]
        offset = int(offset * SCALE)
        if x < offset:
            x = offset
        elif x > self.screen_dim - offset:
            x = self.screen_dim - offset
        if y < offset:
            y = offset
        elif y > self.screen_dim - offset:
            y = self.screen_dim - offset
        return (x, y)

    def _spawn_coins(self):
        self.coin_surface = pygame.Surface((self.screen_dim, self.screen_dim))
        self.coin_surface.fill(255)
        self.coin_surface.set_colorkey(255)
        for _ in range(self.num_coins):
            spawn_pos = self.grid_sampler.sample(21)
            spawn_pos = (spawn_pos[0] + self.np_random.integers(2, 4), spawn_pos[1] + self.np_random.integers(2, 4))
            spawn_pos = self._process_spawn_pos(spawn_pos)
            coin = Coin(self.reset_params["coin_scale"], spawn_pos)
            coin.draw(self.coin_surface)
            self.coins.append(coin)

    def _spawn_exit(self):
        spawn_pos = self.grid_sampler.sample(21)
        spawn_pos = (spawn_pos[0] + self.np_random.integers(2, 4), spawn_pos[1] + self.np_random.integers(2, 4))
        spawn_pos = self._process_spawn_pos(spawn_pos)
        self.exit = Exit(spawn_pos, self.reset_params["exit_scale"])
        self.exit.draw(open=False)

    def _step_coin_task(self):
        reward = 0.0
        done = False
        # Check whether the agent collected a coin and redraw the coin surface
        update_coin_surface = False
        for coin in self.coins:
            if coin.is_agent_inside(self.agent):
                self.coins.remove(coin)
                reward += self.reset_params["reward_coin"]
                update_coin_surface = True
                self.coins_collected += 1
        # Redraw coins if at least one was collected
        if update_coin_surface:
            self.coin_surface.fill(255)
            for coin in self.coins:
                coin.draw(self.coin_surface)
        if not self.coins:
            done = True
        return reward, done

    def _step_exit_task(self, coins_done):
        reward = 0.0
        done = False
        if coins_done:
            self.exit.draw(open = True)
            if self.exit.is_agent_inside(self.agent):
                done = True
                reward = self.reset_params["reward_exit"]
        return reward, done

    def reset(self, seed = None, return_info = True, options = None):
        super().reset(seed=seed)
        self.current_seed = seed
        self.reset_params = SearingSpotlightsEnv.process_reset_params(options)
        self.max_episode_steps = self.reset_params["max_steps"]
        self.t = 0

        if self.reset_params["hide_chessboard"]:
            self.blue_background_surface.fill((255, 255, 255))
            self.red_background_surface.fill((255, 255, 255))

        # Reset spawner
        self.grid_sampler.reset(self.np_random)

        # Track all rewards during one episode
        self.episode_rewards = []

        # Setup agent
        self.last_action = [0, 0]   # The agent shall sense its last action to potentially infer its postion from its past actions
        rotation = self.np_random.choice([0, 45, 90, 135, 180, 225, 270, 315])
        self.agent = CharacterController(self.reset_params["agent_speed"], self.reset_params["agent_scale"], rotation)
        if self.reset_params["sample_agent_position"]:
            spawn_pos = self.grid_sampler.sample(28)
            spawn_pos = (spawn_pos[0] + self.np_random.integers(2, 4), spawn_pos[1] + self.np_random.integers(2, 4))
        else:
            spawn_pos = (self.screen_dim // 2, self.screen_dim // 2)
            self.grid_sampler.block_spawn_position(spawn_pos)
        self.agent.rect.center = spawn_pos
        self.agent_health = self.reset_params["agent_health"]
        self.current_agent_health = self.agent_health
        # The agent's interface, which is a bar on the top of the screen
        # Render the agent's health as a green bar on the first half of the screen width
        self.quarter_width = int(self.screen_dim // 4)
        self.top_bar_surface = pygame.Surface((self.screen_dim, 16 * SCALE))
        pygame.draw.rect(self.top_bar_surface, (0, 255, 0), (0, 0, self.quarter_width * 2, 16 * SCALE))
        # Render the last action of the agent
        self.action_colors = [(120, 120, 120), (116, 1, 113), (255, 255, 20)]
        self.act_rect_0 =  (self.quarter_width * 2, 0, self.quarter_width * 3, 16 * SCALE)
        self.act_rect_1 =  (self.quarter_width * 3, 0, self.quarter_width * 4, 16 * SCALE)
        pygame.draw.rect(self.top_bar_surface, self.action_colors[0], self.act_rect_0)
        pygame.draw.rect(self.top_bar_surface, self.action_colors[0], self.act_rect_1)

        # Setup spotlights
        self.spawn_intervals = self._compute_spawn_intervals(self.reset_params)
        if self.reset_params["light_dim_off_duration"] > 0:
            self.spotlight_surface.set_alpha(0)
        else:
            self.spotlight_surface.set_alpha(self.reset_params["light_threshold"])
        self.spotlights = []
        self.spawn_timer = 0
        for _ in range(self.reset_params["initial_spawns"]):
            self.spotlights.append(Spotlight(self.screen_dim, self.np_random.integers(self.reset_params["spot_min_radius"], self.reset_params["spot_max_radius"] + 1),
                                                            self.np_random.uniform(self.reset_params["spot_min_speed"], self.reset_params["spot_max_speed"]), self.np_random,
                                                            self.reset_params["black_background"]))

        # Spawn coin and exit entities if applicable
        self.coins_collected = 0
        self.coins = []
        self.num_coins = self.np_random.choice(self.reset_params["num_coins"]) if len(self.reset_params["num_coins"]) > 0 else 0
        if self.num_coins > 0:
            self._spawn_coins()
        else:
            self.coin_surface = None
        if self.reset_params["use_exit"]:
            self._spawn_exit()
        else:
            self.exit_surface = None

        # Draw initially all surfaces
        self.bg = self.blue_background_surface
        if self.reset_params["black_background"]:
            self.bg.fill(0)
        surfaces = [(self.bg, (0, 0)), (self.spotlight_surface, (0, 0)), (self.top_bar_surface, (0, 0))]
        spot_surface_id = 1
        if self.reset_params["coins_visible"]:
            surfaces.insert(spot_surface_id + 1, (self.coin_surface, (0, 0)))
        else:
            surfaces.insert(spot_surface_id, (self.coin_surface, (0, 0)))
            spot_surface_id += 1
        if self.reset_params["exit_visible"]:
            surfaces.insert(spot_surface_id + 2, (self.exit.surface, self.exit.rect))
        else:
            surfaces.insert(spot_surface_id, (self.exit.surface, self.exit.rect))
            spot_surface_id += 1
        if self.reset_params["agent_visible"]:
            surfaces.insert(spot_surface_id + 3, (self.rotated_agent_surface, self.rotated_agent_rect))
        else:
            surfaces.insert(spot_surface_id, (self.rotated_agent_surface, self.rotated_agent_rect))
            spot_surface_id += 1
        self._draw_surfaces(surfaces)

        # Show spawn mask for debugging purposes
        # import matplotlib.pyplot as plt
        # plt.imshow(np.flip(np.rot90(self.grid_sampler.spawn_mask), axis=0))
        # plt.show()

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, {}

    def step(self, action):
        # Move the agent's controlled character
        self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step(action, self.walkable_rect)

        # Render the last action of the agent
        pygame.draw.rect(self.top_bar_surface, self.action_colors[self.last_action[0]], self.act_rect_0)
        pygame.draw.rect(self.top_bar_surface, self.action_colors[self.last_action[1]], self.act_rect_1)
        self.last_action = action

        # Dim light untill off
        if self.spotlight_surface.get_alpha() <= self.reset_params["light_threshold"]:
            if self.reset_params["light_dim_off_duration"] > 0:
                self.spotlight_surface.set_alpha(self.spotlight_surface.get_alpha() + int(255 / self.reset_params["light_dim_off_duration"]))
            else:
                self.spotlight_surface.set_alpha(self.reset_params["light_threshold"])

        # Process tasks
        # Spotlight task
        reward = 0.0
        r, spotlights_done, self.bg = self._step_spotlight_task()
        reward += r
        # Coin collection task
        if self.num_coins > 0:
            r, coins_done = self._step_coin_task()
            reward += r
        else:
            coins_done = True
        # Exit task
        success = 0
        if self.reset_params["use_exit"]:
            r, exit_done = self._step_exit_task(coins_done)
            reward += r
        else:
            exit_done = False

        # Determine done
        done = False
        if spotlights_done:
            done = True
        elif coins_done:
            if self.reset_params["use_exit"]:
                if exit_done:
                    done = True
                    success = 1
                else:
                    done = False
            else:
                if self.num_coins > 0:
                    done = True
                    success = 1
                    
        # Time limit
        self.t += 1
        if self.t == self.max_episode_steps:
            done = True

        # Draw all surfaces
        surfaces = [(self.bg, (0, 0)), (self.spotlight_surface, (0, 0)), (self.top_bar_surface, (0, 0))]
        spot_surface_id = 1
        if self.reset_params["coins_visible"]:
            surfaces.insert(spot_surface_id + 1, (self.coin_surface, (0, 0)))
        else:
            surfaces.insert(spot_surface_id, (self.coin_surface, (0, 0)))
            spot_surface_id += 1
        if self.reset_params["exit_visible"]:
            surfaces.insert(spot_surface_id + 2, (self.exit.surface, self.exit.rect))
        else:
            surfaces.insert(spot_surface_id, (self.exit.surface, self.exit.rect))
            spot_surface_id += 1
        if self.reset_params["agent_visible"]:
            surfaces.insert(spot_surface_id + 3, (self.rotated_agent_surface, self.rotated_agent_rect))
        else:
            surfaces.insert(spot_surface_id, (self.rotated_agent_surface, self.rotated_agent_rect))
            spot_surface_id += 1
        self._draw_surfaces(surfaces)

        # Track all rewards
        self.episode_rewards.append(reward)

        if done:
            info = {
                "reward": sum(self.episode_rewards),
                "length": len(self.episode_rewards),
                "agent_health": self.current_agent_health / self.agent_health,
                "coins_collected": self.coins_collected / self.num_coins,
                "success": success,
            }
        else:
            info = {}

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, reward, done, False, info

    def close(self):
        if self.debug_window is not None:
            self.debug_window.destroy()
        pygame.quit()

    def render(self):
        if self.render_mode is not None:
            if self.render_mode == "rgb_array":
                self.clock.tick(SearingSpotlightsEnv.metadata["render_fps"])
                return np.fliplr(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8), 3)) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)
            elif self.render_mode == "debug_rgb_array":
                # Create debug window if it doesn't exist yet
                if self.debug_window is None:
                    self.debug_window = Window(size = (336, 336))
                    self.debug_window.show()
                    self.renderer = Renderer(self.debug_window)

                self.debug_window.title = "seed " + str(self.current_seed)
                self.clock.tick(SearingSpotlightsEnv.metadata["render_fps"])

                debug_surface = self._build_debug_surface()
                texture = Texture.from_surface(self.renderer, debug_surface)
                texture.draw(dstrect=(0, 0))
                self.renderer.present()
                return np.fliplr(np.rot90(pygame.surfarray.array3d(self.renderer.to_surface()).astype(np.uint8), 3))

def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, help="The to be used seed for the environment's random number generator.", default=0)
    options = parser.parse_args()

    env = SearingSpotlightsEnv(render_mode = "debug_rgb_array")
    reset_params = {}
    seed = options.seed
    vis_obs, reset_info = env.reset(seed = options.seed, options = reset_params)
    img = env.render()
    done = False

    while not done:
        actions = [0, 0]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            actions[1] = 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            actions[0] = 2
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            actions[1] = 2
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            actions[0] = 1
        if keys[pygame.K_PAGEDOWN] or keys[pygame.K_PAGEUP]:
            if keys[pygame.K_PAGEUP]:
                seed += 1
            if keys[pygame.K_PAGEDOWN]:
                if not seed <= 0:
                    seed -= 1
            vis_obs, reset_info = env.reset(seed = seed, options = reset_params)
            img = env.render()
        vis_obs, reward, done, truncation, info = env.step(actions)
        img = env.render()

        # Process event-loop
        for event in pygame.event.get():
        # Quit
            if event.type == pygame.QUIT:
                done = True

    print("episode reward: " + str(info["reward"]))
    print("episode length: " + str(info["length"]))
    print("agent health: " + str(info["agent_health"]))
    print("coins collected: " + str(info["coins_collected"]))
    print("success: " + str(bool(info["success"])))

    env.close()
    exit()

if __name__ == "__main__":
    main()