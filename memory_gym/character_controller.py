import pygame

from enum import Enum
from pygame.math import Vector2

class CharacterController():
    """CharacterController establishes a character that is rendered and can be moved using the step function.
    This character can move vertically, horizontally, and diagonally at a certain speed. The character's orientation solely
    depends on its velocity.
    """
    def __init__(self, speed, scale, rotation = 0) -> None:
        self.speed = speed
        self.rotation = rotation
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

class CardinalDirection(Enum):
    NORTH = 0
    WEST = 1
    SOUTH = 2
    EAST = 3

class GridCharacterController(CharacterController):
    """The GridCharacterController establishes a character that has a grid-like locomotion. Using the step function, the 
    character can move forward, rotate left, or rotate right.
    """
    def __init__(self, scale, grid_index_position, grid, rotation = 0) -> None:
        super().__init__(0, scale, rotation)
        self.grid = grid
        self.grid_position = grid_index_position
        self.face_direction = CardinalDirection((rotation % 360) // 90)
        grid_position = self.grid[int(self.grid_position[0])][int(self.grid_position[1])]
        self.rect.center = (grid_position.x ,grid_position.y)

    def step(self, action):
        if action[0] == 1:  # rotate left
            self.rotation = (self.rotation + 90) % 360
        if action[0] == 2:  # rotate right
            self.rotation = (self.rotation - 90) % 360
        self.face_direction = CardinalDirection(self.rotation // 90)
        if action[0] == 3:  # move forward
            x = self.grid_position[0]
            y = self.grid_position[1]
            if self.face_direction == CardinalDirection.NORTH:
                if y > 0:
                    y -= 1
            elif self.face_direction == CardinalDirection.EAST:
                if x < len(self.grid) - 1:
                    x += 1
            elif self.face_direction == CardinalDirection.SOUTH:
                if y < len(self.grid) - 1:
                    y += 1
            elif self.face_direction == CardinalDirection.WEST:
                if x > 0:
                    x -= 1
            self.grid_position = (x, y)
            grid_position = self.grid[int(x)][int(y)]
            self.rect.center = (grid_position.x, grid_position.y)

        return self.rotate(self.rotation)

    def reset_position(self, grid_index_position):
        grid_position = self.grid[int(grid_index_position[0])][int(grid_index_position[1])]
        self.grid_position = grid_index_position
        self.rect.center = (grid_position.x, grid_position.y)