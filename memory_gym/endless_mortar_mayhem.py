import gymnasium as gym
import numpy as np
import os
import pygame

from argparse import ArgumentParser
from gymnasium import spaces
from memory_gym.environment import CustomEnv
from memory_gym.character_controller import ScreenWrapCharacterController
from memory_gym.pygame_assets import Command, MortarArena
from pygame._sdl2 import Window, Texture, Renderer

SCALE = 0.25

class EndlessMortarMayhemEnv(CustomEnv):
    metadata = {
        "render_modes": ["rgb_array", "debug_rgb_array"],
        "render_fps": 25,
    }

    default_reset_parameters = {
                "max_steps": -1,
                "agent_scale": 1.0 * SCALE,
                "agent_speed": 12.0 * SCALE,
                "allowed_commands": 9,
                "initial_command_count": 1,
                "command_show_duration": [3],
                "command_show_delay": [1],
                "explosion_duration": [6],
                "explosion_delay": [18],
                "visual_feedback": True,
                "reward_command_failure": 0.0,
                "reward_command_success": 0.1,
            }

    def process_reset_params(reset_params):
        """Compares the provided reset parameters to the default ones. It asserts whether false reset parameters were provided.
        Missing reset parameters are filled with the default ones.

        Arguments:
            reset_params {dict} -- Provided reset parameters that are to be validated and completed

        Returns:
            dict -- Returns a complete and valid dictionary comprising the to be used reset parameters.
        """
        cloned_params = EndlessMortarMayhemEnv.default_reset_parameters.copy()
        if reset_params is not None:
            for k, v in reset_params.items():
                assert k in cloned_params.keys(), "Provided reset parameter (" + str(k) + ") is not valid. Check spelling."
                cloned_params[k] = v
        assert cloned_params["allowed_commands"] >= 4 and cloned_params["allowed_commands"] <= 9
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
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space= spaces.Box(
                    low = 0.0,
                    high = 1.0,
                    shape = [self.screen_dim, self.screen_dim, 3],
                    dtype = np.float32)

        # Optional information that is part of the returned info dictionary during reset and step
        # The absolute position (ground truth) of the agent is distributed using the info dictionary.
        self.has_ground_truth_info = True
        self.ground_truth_space = spaces.Box(
                    low = np.zeros((2), dtype=np.float32),
                    high = np.ones((2), dtype=np.float32),
                    shape = (2, ),
                    dtype = np.float32)

        # Environment members
        self.rotated_agent_surface, self.rotated_agent_rect = None, None
        self.arena_size = 6

    def _draw_surfaces(self, surfaces):
        # Draw all surfaces
        for surface in surfaces:
            if surface[0] is not None:
                self.screen.blit(surface[0], surface[1])
        pygame.display.flip()

    def _build_debug_surface(self):
        surface = pygame.Surface((336 * SCALE, 336 * SCALE))

        # Gather surfaces
        surfs = [(self.arena.surface, self.arena.rect)]
        # Retrieve the rotated agent surface or the original one
        if self.rotated_agent_surface is not None:
            surfs.append((self.rotated_agent_surface, self.rotated_agent_rect))
        else:
            surfs.append(self.agent.get_rotated_sprite(0))

        # Draw command visualization
        if self._command_visualization:
            command = Command(self._command_visualization_clone.pop(0), SCALE)
            surfs.append((command.surface, (((self.screen_dim // 2) - command.rect_dim // 2, (self.screen_dim // 2) - command.rect_dim // 2))))
       
        # Blit all surfaces
        for surf, rect in surfs:
            surface.blit(surf, rect)

        # Mark the target tile to visualize the ground truth
        target_tile = self.arena.tiles[int(self._target_pos[0])][int(self._target_pos[1])]
        translation = self.arena.rect.center[0] - self.arena.local_center[0] + self.arena.tile_dim // 2
        pos = (target_tile.global_position[0] + translation, target_tile.global_position[1] + translation)
        pygame.draw.circle(surface, (0, 255, 0), pos, self.arena.tile_dim // 2, int(8 * SCALE))

        return pygame.transform.scale(surface, (336, 336))

    def _normalize_agent_position(self, agent_position):
        return ((agent_position[0] - self.arena.rect[0]) // self.arena.tile_dim,
                (agent_position[1] - self.arena.rect[1]) // self.arena.tile_dim)

    def _generate_commands(self, num_commands):
        commands = list(Command.COMMANDS.keys())
        samples = self.np_random.integers(0, self.reset_params["allowed_commands"], num_commands)
        commands = np.take(commands, samples).tolist()
        return commands

    def _generate_command_visualization(self, commands, duration=1, delay=0):
        """Generates a list that states on which step to show which command. Each element corresponds to one step.

        Args:
            commands {list} -- Sampled commands
            duration {int} -- How many steps to show one command (default: {1})
            delay {int} -- How many steps until the next command is shown (default: {0})

        Returns:
            {list} -- list that states on which step to show which command
        """
        command_vis = []
        for i in range(len(commands)):
            # Duplicate the command related to the duration
            for j in range(duration):
                command_vis.append(commands[i])
            # For each step delay, add None instead of the command
            for k in range(delay):
                command_vis.append("")
        return command_vis

    def reset(self, seed = None, return_info = True, options = None):
        super().reset(seed=seed)
        self.current_seed = seed
        self.t = 0

        # Check reset parameters for completeness and errors
        self.reset_params = EndlessMortarMayhemEnv.process_reset_params(options)
        self.max_episode_steps = self.reset_params["max_steps"]

        # Track all rewards during one episode
        self.episode_rewards = []

        # Setup the arena and place it on the center of the screen
        self.bg = pygame.Surface((self.screen_dim, self.screen_dim))
        self.arena = MortarArena(SCALE, self.arena_size)
        self.arena.rect.center = (self.screen_dim // 2, self.screen_dim // 2)

        # Setup the agent and sample its position
        self.agent = ScreenWrapCharacterController(self.reset_params["agent_speed"], self.reset_params["agent_scale"])
        spawn_pos = self.arena.get_tile_global_position(self.np_random.integers(0, self.arena_size ** 2))
        offset = self.np_random.integers(-8 * SCALE, 8 * SCALE, 2)
        translate_x = self.arena.rect.center[0] - self.arena.local_center[0] + self.arena.tile_dim // 2 + offset[0]
        translate_y = self.arena.rect.center[1] - self.arena.local_center[1] + self.arena.tile_dim // 2 + offset[1]
        self.agent.rect.center = spawn_pos[0] + translate_x, spawn_pos[1] + translate_y
        self.normalized_agent_position = self._normalize_agent_position(self.agent.rect.center)

        # Sample the entire command sequence
        self.num_commands = self.reset_params["initial_command_count"]
        self._commands = self._generate_commands(self.num_commands)
        self.show_duration = self.np_random.choice(self.reset_params["command_show_duration"])
        self.show_delay = self.np_random.choice(self.reset_params["command_show_delay"])
        # Prepare list which prepares all steps (i.e. frames) for the visualization
        self._command_visualization = self._generate_command_visualization(self._commands, self.show_duration, self.show_delay)
        self._command_visualization_clone = self._command_visualization.copy() # the clone is needed for render()
        # Retrieve the first command frame
        command = Command(self._command_visualization.pop(0), SCALE)

        # Init episode members
        self._target_pos = ((self.normalized_agent_position[0] + Command.COMMANDS[self._commands[0]][0]) % self.arena_size,
                            (self.normalized_agent_position[1] + Command.COMMANDS[self._commands[0]][1]) % self.arena_size)
        self._current_command = 0       # the current to be executed command
        self._command_steps = 0         # the current step while executing a command (i.e. death tiles off)
        self._command_verify_step = 0   # the current step while the command is being evaluated (i.e. death tiles on)
        self._total_commands_completed = 0
        # Sample execution delay and duration
        self._explosion_duration = self.np_random.choice(self.reset_params["explosion_duration"])
        self._explosion_delay = self.np_random.choice(self.reset_params["explosion_delay"])

        # Draw
        self._draw_surfaces([(self.bg, (0, 0)), (self.arena.surface, self.arena.rect), self.agent.get_rotated_sprite(0),
                            (command.surface, (((self.screen_dim // 2) - command.rect_dim // 2, (self.screen_dim // 2) - command.rect_dim // 2)))])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, {"ground_truth": np.asarray([*self._target_pos]) / 5.0}

    def step(self, action):
        reward = 0
        done = False
        command = None

        # Show each command one by one, while the agent cannot move
        if self._command_visualization:
            command = Command(self._command_visualization.pop(0), SCALE)
            if self.rotated_agent_surface is None and self.rotated_agent_rect is None:
                self.rotated_agent_surface, self.rotated_agent_rect = self.agent.get_rotated_sprite(0)
        # All commands were shown, the agent can move now, while the command execution logic is running
        else:
            # Move the agent's controlled character
            self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step(action, self.arena.rect)
            self.normalized_agent_position = self._normalize_agent_position(self.rotated_agent_rect.center)

            # Process the command execution logic
            # One command is alive for explosion delay steps
            verify = self._command_steps % self._explosion_delay == 0 and self._command_steps > 0

            # Run the verification logic on whether the agent succeeded on moving to the target tile
            if verify and not self.arena.tiles_on:
                if self._current_command < self.num_commands:
                    self._current_command += 1

                    # Turn on the death tiles
                    self.arena.toggle_tiles(self._target_pos, self.reset_params["visual_feedback"])

                    # Check if the agent is on the target position
                    if self.normalized_agent_position == self._target_pos:
                        # Success!
                        reward += self.reset_params["reward_command_success"]
                        self._total_commands_completed += 1
                    # If the agent is not on the target position, terminate the episode
                    else:
                        # Failure!
                        done = True
                        reward += self.reset_params["reward_command_failure"]
                # Finish the episode once all commands are completed
                if self._current_command >= self.num_commands:
                    # All commands completed!
                    # Append another command, reset command logic members, and reset the command visualization
                    new_command = self._generate_commands(1)
                    self._commands.append(new_command[0])
                    self.num_commands = len(self._commands)
                    self._current_command = 0       # the current to be executed command
                    self._command_steps = 0         # the current step while executing a command (i.e. death tiles off)
                    self._command_verify_step = 0   # the current step while the command is being evaluated (i.e. death tiles on)
                    self._command_visualization = self._generate_command_visualization(new_command, self.show_duration, self.show_delay)
                    self._command_visualization_clone = self._command_visualization.copy() # the clone is needed for render()
                self._command_steps = 1

            # Keep the death tiles on for as long as the explosion duration
            if self.arena.tiles_on:
                if self._command_verify_step % self._explosion_duration == 0 and self._command_verify_step > 0:
                    # Turn death tiles off
                    self.arena.toggle_tiles(None, self.reset_params["visual_feedback"])
                    self._command_verify_step = 0
                    if self._current_command < self.num_commands:
                        # Update target position
                        self._target_pos = ((self._target_pos[0] + Command.COMMANDS[self._commands[self._current_command]][0]) % self.arena_size,
                                            (self._target_pos[1] + Command.COMMANDS[self._commands[self._current_command]][1]) % self.arena_size)
                else:
                    # The agent dies upon walking on a death tile
                    if not self.normalized_agent_position == self._target_pos:
                        # Failure!
                        done = True
                        reward = self.reset_params["reward_command_failure"]
                    self._command_verify_step += 1
            else:
                self._command_steps +=1

        # Upper time limit
        self.t += 1
        if self.t == self.max_episode_steps:
            done = True

        # Track all rewards
        self.episode_rewards.append(reward)

        if done:
            info = {
                "reward": sum(self.episode_rewards),
                "length": len(self.episode_rewards),
                "commands_completed": self._total_commands_completed,
                "max_command_sequence": len(self._commands) - 1 if len(self._commands) > 1 else 0,
                "ground_truth": np.asarray([*self._target_pos]) / 5.0
            }
        else:
            # The info dict is used to track the ground truth position of the target
            info = {"ground_truth": np.asarray([*self._target_pos]) / 5.0}
        
        # Draw
        surfaces = [(self.bg, (0, 0)), (self.arena.surface, self.arena.rect), (self.rotated_agent_surface, self.rotated_agent_rect)]
        if command is not None:
            surfaces.append((command.surface, ((self.screen_dim // 2) - command.rect_dim // 2, (self.screen_dim // 2) - command.rect_dim // 2)))
        self._draw_surfaces(surfaces)

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, reward, done, False, info

    def render(self):
        if self.render_mode is not None:
            if self._command_visualization:
                    fps = 3
            else:
                fps = EndlessMortarMayhemEnv.metadata["render_fps"]
            
            if self.render_mode == "rgb_array":
                self.clock.tick(fps)
                return np.fliplr(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8), 3)) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)
            elif self.render_mode == "debug_rgb_array":
                # Create debug window if it doesn't exist yet
                if self.debug_window is None:
                    self.debug_window = Window(size = (336, 336))
                    self.debug_window.show()
                    self.renderer = Renderer(self.debug_window)
                
                self.debug_window.title = "seed " + str(self.current_seed)
                self.clock.tick(fps)

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

    env = EndlessMortarMayhemEnv(render_mode="debug_rgb_array")
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
    print("commands completed: " + str(info["commands_completed"]))
    print("max command sequence completed: " + str(info["max_command_sequence"]))

    env.close()
    exit()

if __name__ == "__main__":
    main()