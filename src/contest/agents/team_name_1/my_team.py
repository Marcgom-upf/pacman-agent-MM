# baseline_team.py
# check
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    MAX_TRAPPED_TURNS = 10
    DEFEND_TURNS_LEFT = 50
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.trapped = False
        self.trapped_turns = 0

    def choose_action(self, game_state):
        self.trapped = self.is_trapped(game_state)
        return super().choose_action(game_state)

    def is_trapped(self, game_state):
        my_state = game_state.get_agent_state(self.index)

        if not my_state.is_pacman:
            self.trapped_turns = 0
            return False
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [nearest_point(e.get_position()) for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0] # nearest_point because we'll use the position for walls and we need int
        
        if not ghosts:
            self.trapped_turns = 0
            return False

        dangerous = set()
        walls = game_state.get_walls()
        for gx, gy in ghosts:
            dangerous.add((gx, gy))
            for near_x, near_y in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
                if not walls[near_x][near_y]:
                    dangerous.add((near_x, near_y))

        mid_x = (walls.width // 2) - 1 if self.red else (walls.width // 2)
        home_positions = set([(mid_x, y) for y in range(walls.height) if not walls[mid_x][y]]) #frontier with enemy
        capsules = set(self.get_capsules(game_state)) 
        food = set(self.get_food(game_state).as_list())

        #BFS to find if there is an exit
        start = nearest_point(my_state.get_position())
        queue = [start]
        visited = {start}
        reachable_food = 0

        while queue:
            cell = queue.pop(0)
            if cell in home_positions or cell in capsules:
                self.trapped_turns = 0
                return False
            if cell in food:
                reachable_food += 1
            
            x, y = cell
            for near_cell in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if near_cell not in visited and not walls[near_cell[0]][near_cell[1]] and near_cell not in dangerous:
                    visited.add(near_cell)
                    queue.append(near_cell)

        self.trapped_turns += 1

        if reachable_food == 0:
            return True
        turns_left = game_state.data.timeleft // 4
        if self.get_score(game_state) > 0 and turns_left < self.DEFEND_TURNS_LEFT: #If we're winning go defend
            return True

        if self.trapped_turns > self.MAX_TRAPPED_TURNS:
            return True
        return False

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        
        if self.trapped:
            new_pos = successor.get_agent_state(self.index).get_position()
            if new_pos == self.start:
                features['suicide'] = 0
            else:
                enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
                ghosts = [e.get_position() for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
                if ghosts:
                    features['suicide'] = min(self.get_maze_distance(new_pos, g) for g in ghosts)
            if action == Directions.STOP:
                features['stop'] = 1                    
            return features
        
        food_list = self.get_food(successor).as_list()
        my_state = game_state.get_agent_state(self.index)
        new = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        new_pos = new.get_position()
        
        #when in the current state, I carry some pelets, and in the new one I have less, it is becouse I deposited
        deposit = 1 if new.num_carrying < my_state.num_carrying else 0
        # we were eaten, not depositing
        if new_pos == self.start and my_pos != self.start:
            deposit = 0
        features['deposit'] = deposit

        walls = game_state.get_walls()
        mid_x = (walls.width // 2) - 1 if self.red else (walls.width // 2)
        home_positions = [(mid_x, y) for y in range(walls.height) if not walls[mid_x][y]] #frontier with enemy
        dist_to_home = min([self.get_maze_distance(new_pos, position) for position in home_positions])
    
        features['ghost_scared_grade'] = 0
        features['dangerous_ghost_dist'] = 0

        # we look through all the ghosts and add their properties to grade
        ghost_blocks_home = False
        for i in self.get_opponents(game_state): #enemies
            g = successor.get_agent_state(i) #enemy state after my action
            now = game_state.get_agent_state(i) #enemy state now

            if g.is_pacman or g.get_position() is None:
                continue

            start_pos_g = game_state.get_initial_agent_position(i) 
            if (now.scared_timer > 0 and now.get_position() is not None and now.get_position() != start_pos_g and g.get_position() == start_pos_g):
                features['eat_scared_ghost'] = 1
                continue

            dist = self.get_maze_distance(new_pos, g.get_position())
            # if we can eat the nearest ghost we reward going to it
            if g.scared_timer > 0:
                features['ghost_scared_grade'] += dist
            #check if the ghost is blocking the path
            elif g.scared_timer == 0:
                ghost_d_home = min(self.get_maze_distance(g.get_position(), p) for p in home_positions)
                if ghost_d_home < dist_to_home:
                    ghost_blocks_home = True
                #if the ghost is agressive and close we discourage approaching it
                if dist <= 3: 
                    features['dangerous_ghost_dist'] += (3 - dist)
                #if the ghost is far and harmful, we ignore it

        #Capsule feature
        active_enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_ghosts = [e.get_position() for e in active_enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
        capsules = self.get_capsules(game_state)

        target_capsule = None
        def dist_to_capsule(capsule):
            return self.get_maze_distance(my_pos, capsule)
        if active_ghosts and capsules:
            dist_to_near_ghost = min(self.get_maze_distance(my_pos, ag) for ag in active_ghosts)
            if dist_to_near_ghost <= 5: 
                reachable_c = [c for c in capsules if dist_to_capsule(c) < min(self.get_maze_distance(ag, c) for ag in active_ghosts)]
                if reachable_c:
                    target_capsule = min(reachable_c, key=dist_to_capsule)
        if target_capsule is not None:
            features['capsule_distance'] = self.get_maze_distance(new_pos, target_capsule)
        if len(self.get_capsules(successor)) < len(capsules):
            features['eat_capsule'] = 1

        
        if (my_state.num_carrying >= 3 and not ghost_blocks_home) or deposit:
            #if we have 5 food, return home to deposit
            features['home_duty'] = dist_to_home
        else:
            features['successor_score'] = -len(food_list)  # self.get_score(successor)
            if not food_list: #if there is no more food
                features['home_duty'] = dist_to_home
            else:
                #search for the nearest food palet
                dists = [self.get_maze_distance(new_pos, food) for food in food_list] 
                features['distance_to_food'] = min(dists) 

        if action == Directions.STOP:
            features['stop'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'home_duty': -10,
                'deposit': 200,
                'successor_score': 110,
                'distance_to_food': -1,
                'ghost_scared_grade': -2,
                'dangerous_ghost_dist': -90,
                'eat_scared_ghost': 300,
                'suicide': -100,
                'stop': -1000,
                'capsule_distance': -60,
                'eat_capsule': 300}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    PATROL_RADIUS = 4
    PATROL_POINTS = 3
    SCARED_DISTANCE = 2

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.patrol_points = []
        self.patrol_idx = 0
        self.patrol_target = None
        self.last_eaten_food = None

    def choose_action(self, game_state):
        self.update_patrol(game_state)
        self.update_eaten_food(game_state)
        return super().choose_action(game_state)

    def update_patrol(self, game_state):

        walls = game_state.get_walls()
        mid_x = (walls.width // 2) - 1 if self.red else (walls.width // 2)
        frontier = [(mid_x, y) for y in range(walls.height) if not walls[mid_x][y]]
        base_food = (game_state.get_red_food() if self.red else game_state.get_blue_food()).as_list()
        
        #Calculate what area has more food density
        def food_concentration(point):
            food_sum = [1 for f in base_food if self.get_maze_distance(point, f) <= self.PATROL_RADIUS]
            return sum(food_sum)

        points_ranked = sorted(frontier, key=food_concentration, reverse=True)
        # We assign only the first 3 points with most food density (PATROL_POINTS = 3)
        self.patrol_points = points_ranked[:self.PATROL_POINTS] 
        if not self.patrol_points:
            self.patrol_target = None

        self.patrol_idx = self.patrol_idx % len(self.patrol_points)
        my_pos = game_state.get_agent_state(self.index).get_position()
        #if it already checked the point, move to another one
        if self.get_maze_distance(my_pos, self.patrol_points[self.patrol_idx]) <= 1:
            self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_points)
        self.patrol_target = self.patrol_points[self.patrol_idx]

    def update_eaten_food(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders_somewhere = [e for e in enemies if e.is_pacman]

        if not invaders_somewhere:
            self.last_eaten_food = None
            return
        
        if self.last_eaten_food is not None and self.get_maze_distance(my_pos, self.last_eaten_food) <= 1:
            self.last_eaten_food = None

        previous = self.get_previous_observation()
        if previous is None:
            return

        food_before = set(self.get_food_you_are_defending(previous).as_list()) 
        food_now = set(self.get_food_you_are_defending(game_state).as_list())
        eaten = food_before - food_now
        def distance_to_eaten(food):
            return self.get_maze_distance(my_pos, food)
        if eaten:
            self.last_eaten_food = min(eaten, key=distance_to_eaten)
        
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders_somewhere = [e for e in enemies if e.is_pacman]
        invaders = [i for i in invaders_somewhere if i.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        my_capsules = self.get_capsules_you_are_defending(game_state)
        scared = True if (game_state.get_agent_state(self.index).scared_timer > 0) else False
        if invaders:
            dists = [self.get_maze_distance(my_pos, i.get_position()) for i in invaders]
            if scared:
                if min(dists) <= 1:
                    features['scared_too_close'] = 1
                features['scared_distance'] = abs(min(dists) - self.SCARED_DISTANCE)
            else:
                features['invader_distance'] = min(dists)
        
        elif invaders_somewhere and self.last_eaten_food is not None:
            features['eaten_food_distance'] = self.get_maze_distance(my_pos, self.last_eaten_food)
        elif invaders_somewhere and my_capsules:
            dists = [self.get_maze_distance(my_pos, c) for c in my_capsules]
            features['capsule_distance'] = min(dists)
        # if there are no invaders and you have not selected a target, choose one
        elif self.patrol_target is not None:
            features['patrol_distance'] = self.get_maze_distance(my_pos, self.patrol_target)

        if scared and my_pos == self.start and not game_state.get_agent_position(self.index) == self.start:
            features['death_distance'] = 1

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 
                'on_defense': 100, 
                'invader_distance': -10, 
                'stop': -100, 
                'reverse': -2,
                'patrol_distance': -10,
                'scared_distance': -10,
                'death_distance': -1000,
                'scared_too_close': -500,
                'capsule_distance': -10,
                'eaten_food_distance': -10}
