import math
import numpy as np
import pygame
import random

from pygame.math import Vector2

class GridPositionSampler():
    def __init__(self, world_dim, cell_dim) -> None:
        assert cell_dim < world_dim
        assert world_dim % cell_dim == 0
        self.cell_dim = cell_dim
        self.grid_dim = int(world_dim // cell_dim)
        self.area = np.arange(0, self.grid_dim ** 2, dtype=np.int32).reshape((self.grid_dim, self.grid_dim))

    def reset(self, rng):
        self.spawn_mask = np.zeros((self.grid_dim, self.grid_dim), dtype=np.bool_)
        self.rng = rng

    def sample(self, block_radius = 2):
        # Retrieve valid indices
        spawn_area = np.ma.array(self.area, mask=self.spawn_mask).compressed()
        # Sample index
        cell_id = self.rng.integers(0, len(spawn_area))
        # Convert to 2D indices
        y = int(spawn_area[cell_id] % self.grid_dim)
        x = int(spawn_area[cell_id] / self.grid_dim)
        # Update spawn mask
        self.spawn_mask[max(0, x - block_radius) : x + block_radius,
                        max(0, y - block_radius) : y + block_radius] = True
        return (x * self.cell_dim, y * self.cell_dim)

class CharacterController():
    def __init__(self, screen_dim, speed, scale) -> None:
        self.screen_dim = screen_dim
        self.speed = speed
        self.rotation = 180
        # Body
        self.radius = int(25 * scale)
        body_color = (250, 204, 153)
        # Hands
        hand_radius = int(10 * scale)
        hands_x_distance = int(18 * scale)
        hand_y_offset = int(12 * scale)
        hand_color = (250, 250, 250)
        hand_outline_color = (50, 50, 50)
        hand_outline_size = int(3 * scale)
        # rect dims
        rect_dim = self.radius * 2 + hand_radius
        self.surface = pygame.Surface((rect_dim, rect_dim))
        self.surface.fill(255)
        self.surface.set_colorkey(255)
        self.rect = self.surface.get_rect()
        self.rect.center = (0, 0)
        self.rotation = 0
        # Draw body
        pygame.draw.circle(self.surface, body_color, (self.radius + hand_radius // 2, self.radius + hand_radius // 2), self.radius)
        # Draw hands
        pygame.draw.circle(self.surface, hand_color, (rect_dim // 2 - hands_x_distance, hand_y_offset), hand_radius) # left
        pygame.draw.circle(self.surface, hand_color, (rect_dim // 2 + hands_x_distance, hand_y_offset), hand_radius) # right
        # Draw hand outline
        pygame.draw.circle(self.surface, hand_outline_color, (rect_dim // 2 - hands_x_distance, hand_y_offset), hand_radius, hand_outline_size) # left
        pygame.draw.circle(self.surface, hand_outline_color, (rect_dim // 2 + hands_x_distance, hand_y_offset), hand_radius, hand_outline_size) # right
        # Draw rect boundaries for debugging
        # pygame.draw.rect(self.surface, (0, 0, 0), (0, 0, rect_dim, rect_dim), 1)

    def rotate(self, angle):
        new_surface = pygame.transform.rotate(self.surface, angle)
        rect = new_surface.get_rect(center = self.rect.center)
        return new_surface, rect

    def step(self, action):
        # Determine agent velocity and rotation
        velocity = Vector2()
        if action[0] == 1:
            self.rotation = 90
            velocity.x = -1
        if action[0] == 2:
            self.rotation = 270
            velocity.x = 1
        if action[1] == 1:
            self.rotation = 0
            velocity.y = -1
        if action[1] == 2:
            self.rotation = 180
            velocity.y = 1

        if velocity.x < 0 and velocity.y < 0:
            self.rotation = 45 # -,-
        if velocity.x < 0 and velocity.y > 0:
            self.rotation = 135 # -,+
        if velocity.x > 0 and velocity.y < 0:
            self.rotation = 305 # +,-
        if velocity.x > 0 and velocity.y > 0:
            self.rotation = 215 # +,+

        # Normalize velocity
        if velocity.length() != 0.0:
            velocity = velocity.normalize() * self.speed
        
        # Update the agent's position
        self.rect.center = Vector2(self.rect.center[0],self.rect.center[1]) + velocity

        # Limit the agent to the screen's boundary
        x = self.rect.center[0]
        y = self.rect.center[1]
        if self.rect.center[0] > self.screen_dim - self.radius:
            x = self.screen_dim - self.radius
        if self.rect.center[0] < self.radius:
            x = 25
        if self.rect.center[1] > self.screen_dim - self.radius:
            y = self.screen_dim - self.radius
        if self.rect.center[1] < self.radius:
            y = self.radius
        self.rect.center = (x, y)

        return self.rotate(self.rotation)

class Spotlight():
    def __init__(self, dim, radius, speed) -> None:
        self.speed = speed
        self.t = 0
        self.done = False
        # Center of the screen
        center = (dim // 2, dim // 2)
        # Length of the diagonal of the screen
        diagonal = math.sqrt(math.pow(dim, 2) + math.pow(dim, 2))
        # Determine final spawn radius to ensure that spotlights are not visible upon spawning
        self.spawn_radius = diagonal / 2 + radius
        self.radius = radius

        # Sample angles for start, end and offset position
        start_angle = random.randint(0, 360) # TODO: Sample all angles during reset
        inverted_angle = start_angle + 180
        target_angle = inverted_angle + random.randint(-45, 45) # TODO: Sample all angles during reset
        offset_angle = target_angle + random.randint(-135, 135) # TODO: Sample all angles during reset

        # Calculate the start position by the sampled angle
        # Code variant A
        # x = spawn_radius * math.cos(math.radians(angle)) + 336 // 2
        # y = spawn_radius * math.sin(math.radians(angle)) + 336 // 2
        # self.start_position = (int(x), int(y))
        # Code variant B
        self.spawn_location = center + Vector2(self.spawn_radius, 0).rotate(start_angle)
        self.current_location = self.spawn_location
        # Calculate target location
        self.target_location = center + Vector2(self.spawn_radius, 0).rotate(target_angle)
        # Calculate offset location
        self.offset_location = center + Vector2(self.spawn_radius, 0).rotate(offset_angle)

    def draw(self, surface):
        lerp_target = self.target_location.lerp(self.offset_location, self.t)
        self.current_location = self.spawn_location.lerp(lerp_target, self.t)
        pygame.draw.circle(surface, (255, 255, 255), (int(self.current_location.x), int(self.current_location.y)), self.radius)
        self.t += self.speed
        if self.t >= 1.0:
            self.t = 1.0
            self.done = True

    def is_agent_inside(self, agent) -> bool:
        distance = self.current_location.distance_to(agent.rect.center)
        if distance <= self.radius + agent.radius:
            return True
        return False

class Coin():
    def __init__(self, scale, location) -> None:
        self.scale = scale
        self.radius = int(10 * scale)
        self.location = location

    def draw(self, surface):
        pygame.draw.circle(surface, (255, 255, 0), self.location, self.radius)
        pygame.draw.circle(surface, (255, 165, 0), self.location, self.radius, int(2 * self.scale))

    def is_agent_inside(self, agent) -> bool:
        location = Vector2(self.location)
        distance = location.distance_to(agent.rect.center)
        if distance <= self.radius + agent.radius:
            return True
        return False

class Exit():
    def __init__(self, location , scale) -> None:
        rect_dim = 20 * scale
        self.surface = pygame.Surface((rect_dim, rect_dim))
        self.surface.fill(255)
        self.surface.set_colorkey(255)
        self.rect = self.surface.get_rect()
        self.origin = self.rect.copy()
        self.scale = scale
        self.radius = 20 / 2 * scale
        self.color_open = (48, 141, 70)
        self.color_closed = (55, 55, 55)
        self.location = location
        self.open = True

    def draw(self, open = False):
        if open != self.open:
            self.open = open
            self.rect.center = self.origin.center
            if open:
                pygame.draw.rect(self.surface, self.color_open, self.rect, 0, 0, 10, 10, 0, 0)
            else:
                pygame.draw.rect(self.surface, self.color_closed, self.rect, 0, 0, 10, 10, 0, 0)
            pygame.draw.rect(self.surface, 0, self.rect, int(2 * self.scale), 0, 10, 10, 0, 0)
            self.rect.center = self.location

    def is_agent_inside(self, agent) -> bool:
            location = Vector2(self.location)
            distance = location.distance_to(agent.rect.center)
            if distance <= self.radius + agent.radius:
                return True
            return False

def get_tiled_background_surface(screen, screen_dim, tile_color):
    background_surface = pygame.Surface((screen_dim, screen_dim))
    ts, w, h, c1, c2 = 50, *screen.get_size(), (255, 255, 255), tile_color
    tiles = [((x*ts, y*ts, ts, ts), c1 if (x+y) % 2 == 0 else c2) for x in range((w+ts-1)//ts) for y in range((h+ts-1)//ts)]
    for rect, color in tiles:
        pygame.draw.rect(background_surface, color, rect)
    return background_surface