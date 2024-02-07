<<<<<<< HEAD
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import random
from collections import deque

# Colors for Snake
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)

DIS_WIDTH = 600
DIS_HEIGHT = 400

SNAKE_BLOCK = 10
SNAKE_SPEED = 50

SNAKE_LEN_GOAL =  50

RENDER = False


# Draws the body of the Snake
def our_snake(snake_block, snake_list, dis):
    for x in snake_list:
        pygame.draw.rect(dis, BLACK, [x[0], x[1], snake_block, snake_block])
 

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # PyGame Rendering
        pygame.init()
        self.dis = pygame.display.set_mode((DIS_WIDTH, DIS_HEIGHT))
        pygame.display.set_caption('Snake Game by Edureka')
        self.font_style = pygame.font.SysFont("bahnschrift", 25)
        self.score_font = pygame.font.SysFont("comicsansms", 35)
        self.clock = pygame.time.Clock()

        # Init variables for tracking with callbacks
        self.attempts = 0
        self.total_food_count = 0

        # Define action
        self.action_space = spaces.Discrete(4)

        # Init and fill deque for observation space
        prev_actions_space_low = deque(maxlen = SNAKE_LEN_GOAL)
        prev_actions_space_high = deque(maxlen = SNAKE_LEN_GOAL)

        for _ in range(SNAKE_LEN_GOAL):
            prev_actions_space_low.append(-1)
            prev_actions_space_high.append(3)

        prev_actions_space_low.append(-1)
        
        # Define observation
        self.observation_space = spaces.Box(low=np.array([0, 0, -590, -390, 0] + list(prev_actions_space_low)), high=np.array([590, 390, 590, 390, SNAKE_LEN_GOAL] + list(prev_actions_space_high)),
                                            shape=(5+SNAKE_LEN_GOAL,), dtype=np.float64)

    def step(self, action):
        # Handle actions ie. appending previous actions, and prep for updating coords of snake head
        # 0 = left 1 = right 2 = up 3 = down
        if action == 0:
            if self.prev_actions[-1] != 1:
                self.prev_actions.append(action)
                self.x1_change = -SNAKE_BLOCK
                self.y1_change = 0
        elif action == 1:
            if self.prev_actions[-1] != 0:
                self.prev_actions.append(action)
                self.x1_change = SNAKE_BLOCK
                self.y1_change = 0
        elif action == 2:
            if self.prev_actions[-1] != 3:
                self.prev_actions.append(action)
                self.y1_change = -SNAKE_BLOCK
                self.x1_change = 0
        elif action == 3:
            if self.prev_actions[-1] != 2:
                self.prev_actions.append(action)
                self.y1_change = SNAKE_BLOCK
                self.x1_change = 0

        # If snake goes off screen set game over to true
        if self.x1 >= DIS_WIDTH or self.x1 < 0 or self.y1 >= DIS_HEIGHT or self.y1 < 0:
            print("OFF SCREEN | FOOD COUNT: " , self.food_count)
            self.game_over = True

        # Update coords of snake head
        self.x1 += self.x1_change
        self.y1 += self.y1_change

        #Fill the screen blue to remove past snake bodies and draw food on screen
        if(RENDER):
            self.dis.fill(BLUE)
            pygame.draw.rect(self.dis, GREEN, [self.foodx, self.foody, SNAKE_BLOCK, SNAKE_BLOCK])

        # Update x and y coord of snake head and append previous snake head to list of body parts
        self.snake_Head = []
        self.snake_Head.append(self.x1)
        self.snake_Head.append(self.y1)
        self.snake_List.append(self.snake_Head)

        # Remove the coords for the end of snake after moving if we didn't grow this step
        if len(self.snake_List) > self.Length_of_snake:
            del self.snake_List[0]

        # If snake head coords = any coords of the snake body set game over to true
        for x in self.snake_List[:-1]:
            if x == self.snake_Head:
                print("SELF EAT | FOOD COUNT: ", self.food_count)
                self.game_over = True

        # Render the new location of the snake after moveing
        if(RENDER):
            our_snake(SNAKE_BLOCK, self.snake_List, self.dis)

        # Update score 
        self.score = self.Length_of_snake - 1

        # Update game window with all changes made
        if(RENDER):
            pygame.display.update()

        # Check if we ate food. If so randomly assign new food coords, increase the length of snake,
        # and give large reward for eating food
        food_reward = 0
        if self.x1 == self.foodx and self.y1 == self.foody:
            self.foodx = round(random.randrange(0, DIS_WIDTH - SNAKE_BLOCK) / 10.0) * 10.0
            self.foody = round(random.randrange(0, DIS_HEIGHT - SNAKE_BLOCK) / 10.0) * 10.0
            self.Length_of_snake += 1
            self.food_count += 1
            food_reward = 10000

        # Tick speed
        self.clock.tick(SNAKE_SPEED)

        # Calculate the euclidean distance from the head of the snake to the food
        euclidean_dist_to_food = np.linalg.norm(np.array([self.x1, self.y1]) - np.array([self.foodx, self.foody]))

        # Calculate reward
        self.total_reward = ((250 - euclidean_dist_to_food) + food_reward) / 100

        # If snake died update the logger variables and punish model for dieing
        if self.game_over:
            self.attempts += 1
            self.total_food_count += self.food_count
            self.reward = -10

        # Calculate distance to food for observation
        food_delta_x = self.foodx - self.x1
        food_delta_y = self.foody - self.y1

        # Build observation
        observation = [self.x1, self.y1, food_delta_x, food_delta_y, self.Length_of_snake] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.total_reward, self.game_over, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game_over = False
        self.score = 0
        self.food_count = 0

        self.x1 = DIS_WIDTH / 2
        self.y1 = DIS_HEIGHT / 2
 
        self.x1_change = 0
        self.y1_change = 0
 
        self.snake_List = []
        self.Length_of_snake = 1
 
        self.foodx = round(random.randrange(0, DIS_WIDTH - SNAKE_BLOCK) / 10.0) * 10.0
        self.foody = round(random.randrange(0, DIS_HEIGHT - SNAKE_BLOCK) / 10.0) * 10.0

        self.prev_actions = deque(maxlen = SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        food_delta_x = self.foodx - self.x1
        food_delta_y = self.foody - self.y1

        observation = [self.x1, self.y1, food_delta_x, food_delta_y, self.Length_of_snake] + list(self.prev_actions)
        observation = np.array(observation)
        info = {}
        return observation, info


=======
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import random
from collections import deque

# Colors for Snake
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)

DIS_WIDTH = 600
DIS_HEIGHT = 400

SNAKE_BLOCK = 10
SNAKE_SPEED = 50

SNAKE_LEN_GOAL =  50

RENDER = False


# Draws the body of the Snake
def our_snake(snake_block, snake_list, dis):
    for x in snake_list:
        pygame.draw.rect(dis, BLACK, [x[0], x[1], snake_block, snake_block])
 

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # PyGame Rendering
        pygame.init()
        self.dis = pygame.display.set_mode((DIS_WIDTH, DIS_HEIGHT))
        pygame.display.set_caption('Snake Game by Edureka')
        self.font_style = pygame.font.SysFont("bahnschrift", 25)
        self.score_font = pygame.font.SysFont("comicsansms", 35)
        self.clock = pygame.time.Clock()

        # Init variables for tracking with callbacks
        self.attempts = 0
        self.total_food_count = 0

        # Define action
        self.action_space = spaces.Discrete(4)

        # Init and fill deque for observation space
        prev_actions_space_low = deque(maxlen = SNAKE_LEN_GOAL)
        prev_actions_space_high = deque(maxlen = SNAKE_LEN_GOAL)

        for _ in range(SNAKE_LEN_GOAL):
            prev_actions_space_low.append(-1)
            prev_actions_space_high.append(3)

        prev_actions_space_low.append(-1)
        
        # Define observation
        self.observation_space = spaces.Box(low=np.array([0, 0, -590, -390, 0] + list(prev_actions_space_low)), high=np.array([590, 390, 590, 390, SNAKE_LEN_GOAL] + list(prev_actions_space_high)),
                                            shape=(5+SNAKE_LEN_GOAL,), dtype=np.float64)

    def step(self, action):
        # Handle actions ie. appending previous actions, and prep for updating coords of snake head
        # 0 = left 1 = right 2 = up 3 = down
        if action == 0:
            if self.prev_actions[-1] != 1:
                self.prev_actions.append(action)
                self.x1_change = -SNAKE_BLOCK
                self.y1_change = 0
        elif action == 1:
            if self.prev_actions[-1] != 0:
                self.prev_actions.append(action)
                self.x1_change = SNAKE_BLOCK
                self.y1_change = 0
        elif action == 2:
            if self.prev_actions[-1] != 3:
                self.prev_actions.append(action)
                self.y1_change = -SNAKE_BLOCK
                self.x1_change = 0
        elif action == 3:
            if self.prev_actions[-1] != 2:
                self.prev_actions.append(action)
                self.y1_change = SNAKE_BLOCK
                self.x1_change = 0

        # If snake goes off screen set game over to true
        if self.x1 >= DIS_WIDTH or self.x1 < 0 or self.y1 >= DIS_HEIGHT or self.y1 < 0:
            print("OFF SCREEN | FOOD COUNT: " , self.food_count)
            self.game_over = True

        # Update coords of snake head
        self.x1 += self.x1_change
        self.y1 += self.y1_change

        #Fill the screen blue to remove past snake bodies and draw food on screen
        if(RENDER):
            self.dis.fill(BLUE)
            pygame.draw.rect(self.dis, GREEN, [self.foodx, self.foody, SNAKE_BLOCK, SNAKE_BLOCK])

        # Update x and y coord of snake head and append previous snake head to list of body parts
        self.snake_Head = []
        self.snake_Head.append(self.x1)
        self.snake_Head.append(self.y1)
        self.snake_List.append(self.snake_Head)

        # Remove the coords for the end of snake after moving if we didn't grow this step
        if len(self.snake_List) > self.Length_of_snake:
            del self.snake_List[0]

        # If snake head coords = any coords of the snake body set game over to true
        for x in self.snake_List[:-1]:
            if x == self.snake_Head:
                print("SELF EAT | FOOD COUNT: ", self.food_count)
                self.game_over = True

        # Render the new location of the snake after moveing
        if(RENDER):
            our_snake(SNAKE_BLOCK, self.snake_List, self.dis)

        # Update score 
        self.score = self.Length_of_snake - 1

        # Update game window with all changes made
        if(RENDER):
            pygame.display.update()

        # Check if we ate food. If so randomly assign new food coords, increase the length of snake,
        # and give large reward for eating food
        food_reward = 0
        if self.x1 == self.foodx and self.y1 == self.foody:
            self.foodx = round(random.randrange(0, DIS_WIDTH - SNAKE_BLOCK) / 10.0) * 10.0
            self.foody = round(random.randrange(0, DIS_HEIGHT - SNAKE_BLOCK) / 10.0) * 10.0
            self.Length_of_snake += 1
            self.food_count += 1
            food_reward = 10000

        # Tick speed
        self.clock.tick(SNAKE_SPEED)

        # Calculate the euclidean distance from the head of the snake to the food
        euclidean_dist_to_food = np.linalg.norm(np.array([self.x1, self.y1]) - np.array([self.foodx, self.foody]))

        # Calculate reward
        self.total_reward = ((250 - euclidean_dist_to_food) + food_reward) / 100

        # If snake died update the logger variables and punish model for dieing
        if self.game_over:
            self.attempts += 1
            self.total_food_count += self.food_count
            self.reward = -10

        # Calculate distance to food for observation
        food_delta_x = self.foodx - self.x1
        food_delta_y = self.foody - self.y1

        # Build observation
        observation = [self.x1, self.y1, food_delta_x, food_delta_y, self.Length_of_snake] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.total_reward, self.game_over, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game_over = False
        self.score = 0
        self.food_count = 0

        self.x1 = DIS_WIDTH / 2
        self.y1 = DIS_HEIGHT / 2
 
        self.x1_change = 0
        self.y1_change = 0
 
        self.snake_List = []
        self.Length_of_snake = 1
 
        self.foodx = round(random.randrange(0, DIS_WIDTH - SNAKE_BLOCK) / 10.0) * 10.0
        self.foody = round(random.randrange(0, DIS_HEIGHT - SNAKE_BLOCK) / 10.0) * 10.0

        self.prev_actions = deque(maxlen = SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        food_delta_x = self.foodx - self.x1
        food_delta_y = self.foody - self.y1

        observation = [self.x1, self.y1, food_delta_x, food_delta_y, self.Length_of_snake] + list(self.prev_actions)
        observation = np.array(observation)
        info = {}
        return observation, info


>>>>>>> 83517401dc00beacb07bef29d54f504856276924
