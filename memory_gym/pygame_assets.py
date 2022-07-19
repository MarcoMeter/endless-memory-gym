import math
import numpy as np
import pygame

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

    def step(self, action, boundary_rect = None):
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

        # Limit the agent to the specified rect's boundary
        if boundary_rect is not None:
            x = self.rect.center[0]
            y = self.rect.center[1]
            if x > boundary_rect.bottomright[0] - self.radius:
                x = boundary_rect.bottomright[0] - self.radius
            if x < boundary_rect.topleft[0] + self.radius:
                x = boundary_rect.topleft[0] + self.radius
            if y > boundary_rect.bottomright[1] - self.radius:
                y = boundary_rect.bottomright[1] - self.radius
            if y < boundary_rect.topleft[1] + self.radius:
                y = boundary_rect.topleft[1] + self.radius
            self.rect.center = (x, y)

        return self.rotate(self.rotation)


class Spotlight():
    def __init__(self, dim, radius, speed, rng) -> None:
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
        start_angle = rng.integers(0, 360)
        inverted_angle = start_angle + 180
        target_angle = inverted_angle + rng.integers(-45, 45)
        offset_angle = target_angle + rng.integers(-135, 135)

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
        self.rounded_corners = (int(10 * scale), int(10 * scale), 0, 0)
        self.location = location
        self.open = True

    def draw(self, open = False):
        if open != self.open:
            self.open = open
            self.rect.center = self.origin.center
            if open:
                pygame.draw.rect(self.surface, self.color_open, self.rect, 0, 0, *self.rounded_corners)
            else:
                pygame.draw.rect(self.surface, self.color_closed, self.rect, 0, 0, *self.rounded_corners)
            pygame.draw.rect(self.surface, 0, self.rect, int(2 * self.scale), 0, *self.rounded_corners)
            self.rect.center = self.location

    def is_agent_inside(self, agent) -> bool:
            location = Vector2(self.location)
            distance = location.distance_to(agent.rect.center)
            if distance <= self.radius + agent.radius:
                return True
            return False

def get_tiled_background_surface(screen, screen_dim, tile_color, scale):
    background_surface = pygame.Surface((screen_dim, screen_dim))
    ts, w, h, c1, c2 = int(50 * scale), *screen.get_size(), (255, 255, 255), tile_color
    tiles = [((x*ts, y*ts, ts, ts), c1 if (x+y) % 2 == 0 else c2) for x in range((w+ts-1)//ts) for y in range((h+ts-1)//ts)]
    for rect, color in tiles:
        pygame.draw.rect(background_surface, color, rect)
    return background_surface

class Command():
    COMMANDS = {
        "right"     : (1, 0),
        "down"      : (0, 1),
        "left"      : (-1, 0),
        "up"        : (0, -1),
        "stay"      : (0, 0),
        "right_down": (1, 1),
        "right_up"  : (1, -1),
        "left_down" : (-1, 1),
        "left_up"   : (-1, -1),
    }

    def __init__(self, command_type, scale) -> None:
        assert command_type in Command.COMMANDS or command_type == ""
        self.scale = scale
        self.rect_dim = 88 * scale
        self.surface = pygame.Surface((self.rect_dim, self.rect_dim))
        self.surface.fill(0)
        self.surface.set_colorkey(0)
        self.rect = self.surface.get_rect()

        # Draw command symbol
        line_width = int(8 * scale)
        if command_type == "stay":
            radius = self.rect_dim // 2 - 4 * scale
            x = self.rect_dim - 12 * scale
            y = self.rect_dim // 2 - 8 * scale
            pygame.draw.circle(self.surface, (255, 255, 255), (radius, radius), radius=radius, width=line_width)
            pygame.draw.line(self.surface, (255, 255, 255), (0, y), (x, y), width=line_width)
        elif len(command_type) > 0:
            # Draw arrow that points right
            x1 = 2 * scale
            x2 = 80 * scale
            y1 = 40 * scale
            y2 = 0
            pygame.draw.line(self.surface, (255, 255, 255), (x1, y1), (x2, y1), line_width)
            pygame.draw.line(self.surface, (255, 255, 255), (x2, y1), (y1, y2), line_width)
            pygame.draw.line(self.surface, (255, 255, 255), (x2, y1), (y1, x2), line_width)
            # Determine rotation
            angle = 0
            if command_type == "left":
                angle = 180
            elif command_type == "down":
                angle = 270
            elif command_type == "up":
                angle = 90
            elif command_type == "right_down":
                angle = 315
            elif command_type == "right_up":
                angle = 45
            elif command_type == "left_down":
                angle = 225
            elif command_type == "left_up":
                angle = 135
            # self.surface, self.rect = self.rotate(angle)
            self.surface = pygame.transform.rotate(self.surface, angle)
            self.rect = self.surface.get_rect(center = self.rect.center)

class MortarTile():
    def __init__(self, dim, scale, global_position, surface_rect) -> None:
        self.dim = dim
        self.scale = scale
        self.surface = pygame.Surface((dim, dim))
        self.rect = self.surface.get_rect()
        self.global_position = global_position
        self.local_position = (global_position[0] - surface_rect[0], global_position[1] - surface_rect[1])
        self.normalized_pos = (self.local_position[0] // self.dim, self.local_position[1] // self.dim)
        self.blue = (21, 43, 77)
        self.light_blue = (29, 60, 107)
        self.red = (81, 18, 26)
        self.light_red = (112, 24, 36)
        self.is_blue = True
        pygame.draw.rect(self.surface, self.blue, ((0, 0, dim, dim)))
        pygame.draw.rect(self.surface, self.light_blue, ((0, 0, dim, dim)), width=int(4 * scale))

    def toggle_color(self, on):
        self.is_blue = not self.is_blue
        c1 = self.blue if not on else self.red
        c2 = self.light_blue if not on else self.light_red
        pygame.draw.rect(self.surface, c1, ((0, 0, self.dim, self.dim)))
        pygame.draw.rect(self.surface, c2, ((0, 0, self.dim, self.dim)), width=int(4 * self.scale))

class MortarArena():
    def __init__(self, scale, arena_size) -> None:
        self.scale = scale
        self.arena_size = arena_size
        self.tile_dim = 56 * scale
        self.rect_dim = self.tile_dim * arena_size
        self.surface = pygame.Surface((self.rect_dim, self.rect_dim))
        self.rect = self.surface.get_rect()
        self.tiles = [[] for _ in range(arena_size)]
        self.tiles_on = False
        for i in range(self.arena_size):
            x = self.tile_dim * i
            for j in range(self.arena_size):
                y = self.tile_dim * j
                tile = MortarTile(self.tile_dim, scale, (x, y), self.rect)
                self.tiles[i].append(tile)
                self.surface.blit(tile.surface, tile.global_position)

    def get_tile_global_position(self, flat_tile_id):
        x = flat_tile_id // self.arena_size
        y = flat_tile_id % self.arena_size
        tile = self.tiles[x][y]
        pos = tile.global_position
        pos = (pos[0] + self.tile_dim, pos[1] + self.tile_dim)
        return pos

    def toggle_tiles(self, target_tile = None):
        self.tiles_on = target_tile is not None
        for i in range(self.arena_size):
            for j in range(self.arena_size):
                if self.tiles_on:
                    if not target_tile == (i, j):
                        tile = self.tiles[i][j]
                        tile.toggle_color(on = self.tiles_on)
                        self.surface.blit(tile.surface, tile.global_position)
                else:
                    tile = self.tiles[i][j]
                    tile.toggle_color(on = self.tiles_on)
                    self.surface.blit(tile.surface, tile.global_position)
