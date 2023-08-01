import pygame

from enum import Enum
from pygame.math import Vector2

class CharacterController():
    """CharacterController establishes a character that is rendered and can be moved using the step function.
    This character can move vertically, horizontally, and diagonally at a certain speed. The character's orientation solely
    depends on its velocity. The character's sprite is a circle with two hands that are rotated based on the character's
    center. Optionally, the character's position is limited to the specified boundary rect.
    """
    def __init__(self, speed, scale, rotation = 0) -> None:
        """Initializes the character controller.

        Arguments:
            speed {int} -- Speed of the character in pixels per frame.
            scale {float} -- Scale of the character.
            rotation {int} -- Initial rotation of the character in degrees (default: {0})
        """
        self.speed = speed
        self.scale = scale
        self.rotation = rotation
        self.radius = int(25 * scale)
        self.character_surfaces = self.create_character_sprites()
        self.rect = self.character_surfaces[0].get_rect()
        self.rect.center = (0, 0)
        self.velocity = Vector2(0, 0)

    def create_character_sprites(self):
        """Create eight sprites for the character, one for each 45 degree rotation.

        Returns:
            {list} -- List of surfaces
        """
        sprite_surfaces = []
        # Colors
        body_color = (250, 204, 153)
        hand_color = (250, 250, 250)
        hand_outline_color = (50, 50, 50)
        # Dimensions and offsets
        hands_x_distance = int(18 * self.scale)
        hand_y_offset = int(12 * self.scale)
        hand_radius = int(10 * self.scale)
        hand_outline_size = int(3 * self.scale)
        extension = 14
        rect_dim = self.radius * 2 + hand_radius + extension
        center = Vector2(rect_dim // 2, rect_dim // 2)
        # Initial hand positions
        left_hand_pos = Vector2(rect_dim // 2 - hands_x_distance, hand_y_offset + extension // 2)
        right_hand_pos = Vector2(rect_dim // 2 + hands_x_distance, hand_y_offset + extension // 2)
        # Create sprites
        for i in reversed(range(45, 405, 45)):
            # Setup surface
            surface = pygame.Surface((rect_dim, rect_dim))
            surface.fill(255)
            surface.set_colorkey(255)
            # Draw body
            pygame.draw.circle(surface, body_color, center, self.radius)
            # Rotate hand positions around the surface's center
            left_hand_rot_pos = left_hand_pos - center
            right_hand_rot_pos = right_hand_pos - center
            left_hand_rot_pos.rotate_ip(i)
            right_hand_rot_pos.rotate_ip(i)
            left_hand_rot_pos = left_hand_rot_pos + center
            right_hand_rot_pos = right_hand_rot_pos + center
            # Draw hands
            pygame.draw.circle(surface, hand_color, left_hand_rot_pos, hand_radius)
            pygame.draw.circle(surface, hand_color, right_hand_rot_pos, hand_radius)
            pygame.draw.circle(surface, hand_outline_color, left_hand_rot_pos, hand_radius, hand_outline_size)
            pygame.draw.circle(surface, hand_outline_color, right_hand_rot_pos, hand_radius, hand_outline_size)
            # Draw rect boundaries for debugging
            # pygame.draw.rect(surface, (0, 0, 0), (0, 0, rect_dim, rect_dim), 1)
            # Add surface to list
            sprite_surfaces.append(surface)
        return sprite_surfaces

    def get_rotated_sprite(self, angle):
        """Returns the character sprite for the specified angle.

        Arguments:
            angle {int} -- Must be a multiple of 45 degrees.

        Returns:
            {tuple} -- Surface and rect
        """
        self.character_surfaces[angle // 45]
        return self.character_surfaces[angle // 45], self.rect

    def step(self, action, boundary_rect = None):
        """Updates the character's position and rotation based on the specified action.

        Arguments:
            action {list} -- List of ints, where each int represents a direction [horizontal, vertical].
            boundary_rect {rect} -- Rect that the character's position is limited to (default: {None})

        Returns:
            {tuple} -- Character sprite surface and rect
        """
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
            self.rotation = 315 # +,-
        if velocity.x > 0 and velocity.y > 0:
            self.rotation = 225 # +,+

        # Normalize velocity
        if velocity.length() != 0.0:
            velocity = velocity.normalize() * self.speed
            velocity = Vector2(int(velocity.x), int(velocity.y))
        
        # Update the agent's position
        self.rect.center = Vector2(self.rect.center[0],self.rect.center[1]) + velocity
        self.velocity = velocity

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

        return self.get_rotated_sprite(self.rotation)

class CardinalDirection(Enum):
    """Enumeration representing the cardinal directions (North, West, South, and East)."""
    NORTH = 0
    WEST = 1
    SOUTH = 2
    EAST = 3

class GridCharacterController(CharacterController):
    """The GridCharacterController establishes a character that has a grid-like locomotion. Using the step function, the 
    character can move forward, rotate left, or rotate right. The character's position is determined by the grid and the
    grid index position. The character's rotation is determined by the rotation parameter. The character's rotation is
    always a multiple of 90 degrees.
    """
    def __init__(self, scale, grid_index_position, grid, rotation = 0) -> None:
        """Initializes the GridCharacterController.

        Arguments:
            scale {int} -- Scale of the character.
            grid_index_position {tuple} -- Grid index position of the character.
            grid {list} -- Grid that the character is on. 
            rotation {int} -- Initial rotation. (default: {0})
        """
        super().__init__(0, scale, rotation)
        self.grid = grid
        self.grid_position = grid_index_position
        self.face_direction = CardinalDirection((rotation % 360) // 90)
        grid_position = self.grid[int(self.grid_position[0])][int(self.grid_position[1])]
        self.rect.center = (grid_position.x ,grid_position.y)

    def step(self, action):
        """Updates the character's position and rotation based on the specified action.

        Arguments:
            action {int} -- Action to take. 0 = do nothing, 1 = rotate left, 2 = rotate right, 3 = move forward.

        Returns:
            {tuple} -- Character sprite surface and rect
        """
        if action == 1:  # rotate left
            self.rotation = (self.rotation + 90) % 360
        if action == 2:  # rotate right
            self.rotation = (self.rotation - 90) % 360
        self.face_direction = CardinalDirection(self.rotation // 90)
        if action == 3:  # move forward
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

        return self.get_rotated_sprite(self.rotation)

    def reset_position(self, grid_index_position):
        """Resets the character's position to the specified grid index position."""
        grid_position = self.grid[int(grid_index_position[0])][int(grid_index_position[1])]
        self.grid_position = grid_index_position
        self.rect.center = (grid_position.x, grid_position.y)

class ScreenWrapCharacterController(CharacterController):
    """ScreenWrapCharacterController establishes a character that is rendered and can be moved using the step function.
    This character can move vertically, horizontally, and diagonally at a certain speed. The character's orientation solely
    depends on its velocity. If the agent leaves the screen, it will reappear on the opposite side of the screen.
    """
    def __init__(self, speed, scale, rotation = 0) -> None:
        super().__init__(speed, scale, rotation)

    def step(self, action, boundary_rect = None):
        """Updates the character's position and rotation based on the specified action. If the character leaves the screen,
        it will reappear on the opposite side of the screen.

        Arguments:
            action {list} -- List of ints, where each int represents a direction [horizontal, vertical].
            boundary_rect {rect} -- Rect that the character's position is limited to (default: {None})

        Returns:
            {tuple} -- Character sprite surface and rect
        """
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
            self.rotation = 315 # +,-
        if velocity.x > 0 and velocity.y > 0:
            self.rotation = 225 # +,+

        # Normalize velocity
        if velocity.length() != 0.0:
            velocity = velocity.normalize() * self.speed

        # Update the agent's position
        self.rect.center = Vector2(self.rect.center[0],self.rect.center[1]) + velocity
        
        # Wrap the agent to the opposite side of the screen (i.e. boundary_rect)
        offset = 0.5
        if boundary_rect is not None:
            x = self.rect.center[0]
            y = self.rect.center[1]
            if x > boundary_rect.bottomright[0] + self.radius * offset:
                x = boundary_rect.topleft[0] - self.radius * offset
            if x < boundary_rect.topleft[0] - self.radius * offset:
                x = boundary_rect.bottomright[0] + self.radius * offset
            if y > boundary_rect.bottomright[1] + self.radius * offset:
                y = boundary_rect.topleft[1] - self.radius * offset
            if y < boundary_rect.topleft[1] - self.radius * offset:
                y = boundary_rect.bottomright[1] + self.radius * offset
            self.rect.center = (x, y)

        return self.get_rotated_sprite(self.rotation)