import gym
import math
import numpy as np
import os
import pygame

from argparse import ArgumentParser
from gym import  spaces
from memory_gym.pygame_assets import CharacterController, Command, MortarArena
from pygame._sdl2 import Window, Texture, Renderer

SCALE = 1.0

class MortarMayhemEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "debug_rgb_array"],
        "render_fps": 25,
    }

    default_reset_parameters = {
                "agent_scale": 1.0 * SCALE,
                "agent_speed": 10.0 * SCALE,
                "arena_size": 5,
                "allowed_commands": 4,
                "command_count": 5,
                "command_show_duration": 3,
                "command_duration": 12,
                "command_delay": 1,
                "use_command_alternative": False,
                "explosion_duration": 4,
                "explosion_delay": 4,
                "reward_command_failure": -0.1,
                "reward_command_success": 0.1,
                "reward_episode_success": 0.0
            }

    def process_reset_params(reset_params):
        cloned_params = MortarMayhemEnv.default_reset_parameters.copy()
        if reset_params is not None:
            for k, v in reset_params.items():
                assert k in cloned_params.keys(), "Provided reset parameter (" + str(k) + ") is not valid. Check spelling."
                cloned_params[k] = v
        assert cloned_params["allowed_commands"] >= 4 and cloned_params["allowed_commands"] <= 9
        assert cloned_params["arena_size"] >= 2 and cloned_params["arena_size"] <= 6
        return cloned_params

    def __init__(self, headless = True) -> None:
        if headless:
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            pygame.display.set_caption("Environment")

        self.screen_dim = int(336 * SCALE)

        # Setup observation and action space
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space= spaces.Box(
                    low = 0.0,
                    high = 1.0,
                    shape = [self.screen_dim, self.screen_dim, 3],
                    dtype = np.float32)

        # Init PyGame screen
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim), pygame.NOFRAME)

        # Init debug window
        self.debug_window = None
        
        if headless:
            pygame.event.set_allowed(None)
        self.clock = pygame.time.Clock()

        self.rotated_agent_surface, self.rotated_agent_rect = None, None

    def _draw_surfaces(self, surfaces):
        # Draw all surfaces
        for surface in surfaces:
            if surface[0] is not None:
                self.screen.blit(surface[0], surface[1])
        pygame.display.flip()

    def _normalize_agent_position(self, agent_position):
        return ((agent_position[0] - self.arena.rect[0]) // self.arena.tile_dim,
                (agent_position[1] - self.arena.rect[1]) // self.arena.tile_dim)

    def _get_valid_commands(self, pos):
        # Check whether each command can be executed or not
        valid_commands = []
        keys = list(Command.COMMANDS.keys())[:self.reset_params["allowed_commands"]]
        available_commands = {key: Command.COMMANDS[key] for key in keys}
        for key, value in available_commands.items():
            test_pos = (pos[0] + value[0], pos[1] + value[1])
            if test_pos[0] >= 0 and test_pos[0] < self.reset_params["arena_size"]:
                if test_pos[1] >= 0 and test_pos[1] < self.reset_params["arena_size"]:
                    valid_commands.append(key)
        # Return the commands that can be executed
        return valid_commands

    def _generate_commands(self, start_pos):
        simulated_pos = start_pos
        commands = []
        for i in range(self.reset_params["command_count"]):
            # Retrieve valid commands (we cannot walk on to a wall)
            valid_commands = self._get_valid_commands(simulated_pos)            
            # Sample one command from the available ones
            sample = valid_commands[self.np_random.integers(0, len(valid_commands))]
            commands.append(sample)
            # Update the simulated position
            simulated_pos = (simulated_pos[0] + Command.COMMANDS[sample][0], simulated_pos[1] + Command.COMMANDS[sample][1])
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
        self.reset_params = MortarMayhemEnv.process_reset_params(options)

        # Track all rewards during one episode
        self.episode_rewards = []

        # Setup arena and place on the center of the screen
        self.bg = pygame.Surface((self.screen_dim, self.screen_dim))
        self.bg.fill(0)
        self.arena = MortarArena(SCALE, self.reset_params["arena_size"])
        self.arena.rect.center = (self.screen_dim // 2, self.screen_dim // 2)

        # Setup the agent and sample its position
        self.agent = CharacterController(self.screen_dim, self.reset_params["agent_speed"], self.reset_params["agent_scale"])
        spawn_pos = self.arena.get_tile_global_position(self.np_random.integers(0, self.reset_params["arena_size"] ** 2))
        spawn_pos = (spawn_pos[0] + self.np_random.integers(-12 * SCALE, 12 * SCALE), spawn_pos[1] + self.np_random.integers(-12 * SCALE, 12 * SCALE))
        self.agent.rect.center = spawn_pos
        self.normalized_agent_position = self._normalize_agent_position(self.agent.rect.center)

        # Sample n commands
        self._commands = self._generate_commands(self.normalized_agent_position)
        self._command_visualization = self._generate_command_visualization(self._commands, self.reset_params["command_show_duration"], self.reset_params["command_delay"])
        # Show first command frame
        command = Command(self._command_visualization.pop(0), SCALE)

        # Init episode members
        self._target_pos = (self.normalized_agent_position[0] + Command.COMMANDS[self._commands[0]][0],
                            self.normalized_agent_position[1] + Command.COMMANDS[self._commands[0]][1])
        self._current_command = 0
        self._step_count_commands = 0

        # Draw
        self._draw_surfaces([(self.bg, (0, 0)), (self.arena.surface, self.arena.rect), (self.agent.surface, self.agent.rect),
                            (command.surface, (((self.screen_dim // 2) - command.rect_dim // 2, (self.screen_dim // 2) - command.rect_dim // 2)))])

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs

    def step(self, action):
        reward = 0
        done = False

        # Show each command one by one, while the agent cannot move
        if self._command_visualization:
            command = Command(self._command_visualization.pop(0), SCALE)
            self.rotated_agent_surface, self.rotated_agent_rect = self.agent.surface, self.agent.rect
        else:
            command = None
            # Move the agent's controlled character
            self.rotated_agent_surface, self.rotated_agent_rect = self.agent.step(action, self.arena.rect)
            self.normalized_agent_position = self._normalize_agent_position(self.rotated_agent_rect.center)

            # Process the command execution logic
            # One command is alive for command_duration steps
            if (self._step_count_commands) % (self.reset_params["command_duration"]) == 0 and self._step_count_commands > 0:
                 # Check if to be executed commands are still remaining
                if self._current_command < self.reset_params["command_count"]:
                    self._current_command += 1

                    # Check if the agent is on the target position
                    if self.normalized_agent_position == self._target_pos:
                        # Success!
                        if self._current_command < self.reset_params["command_count"]:
                            # Update target position
                            self._target_pos = (self._target_pos[0] + Command.COMMANDS[self._commands[self._current_command]][0],
                                                self._target_pos[1] + Command.COMMANDS[self._commands[self._current_command]][1])
                        reward += self.reset_params["reward_command_success"]
                        print("SUCCESS")
                    # If the agent is not on the target position, terminate the episode
                    else:
                        # Failure!
                        done = True
                        reward += self.reset_params["reward_command_failure"]
                        print("FAILED")
                # Finish the episode once all commands are completed
                if self._current_command >= self.reset_params["command_count"]:
                    # All commands completed!
                    done = True
                    reward += self.reset_params["reward_episode_success"]
                    print("SUCCESS and DONE")

            # We cannot make use of the global step count due to potentially using the single command visualization
            self._step_count_commands +=1

        # Track all rewards
        self.episode_rewards.append(reward)

        if done:
            info = {
                "reward": sum(self.episode_rewards),
                "length": len(self.episode_rewards)
            }
        else:
            info = {}
        
        # Draw
        surfaces = [(self.bg, (0, 0)), (self.arena.surface, self.arena.rect), (self.rotated_agent_surface, self.rotated_agent_rect)]
        if command is not None:
            surfaces.append((command.surface, ((self.screen_dim // 2) - command.rect_dim // 2, (self.screen_dim // 2) - command.rect_dim // 2)))
        self._draw_surfaces(surfaces)

        # Retrieve the rendered image of the environment
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

        return vis_obs, reward, done, info

    def render(self, mode = "rgb_array"):
        if mode == "rgb_array":
            if self._command_visualization:
                self.clock.tick(4)
            else:
                self.clock.tick(MortarMayhemEnv.metadata["render_fps"])
            return pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8) # pygame.surfarray.pixels3d(pygame.display.get_surface()).astype(np.uint8)

    def close(self):
            if self.debug_window is not None:
                self.debug_window.destroy()
            pygame.quit()

def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, help="The to be used seed for the environment's random number generator.", default=0)
    options = parser.parse_args()

    env = MortarMayhemEnv(headless = False)
    reset_params = {}
    vis_obs = env.reset(seed = options.seed, options = reset_params)
    img = env.render(mode = "rgb_array")
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
        vis_obs, reward, done, info = env.step(actions)
        img = env.render(mode = "rgb_array")

        # Process event-loop
        for event in pygame.event.get():
        # Quit
            if event.type == pygame.QUIT:
                done = True

    print("episode reward: " + str(info["reward"]))
    print("episode length: " + str(info["length"]))

    env.close()
    exit()

if __name__ == "__main__":
    main()