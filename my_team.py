# baseline_team.py
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
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


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
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.previous_positions = []  # Keep track of previous visited positions

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
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

        # Track previous positions
        my_pos = game_state.get_agent_state(self.index).get_position()
        self.previous_positions.append(my_pos)
        if len(self.previous_positions) > 10:  # Keep last 10 positions
            self.previous_positions.pop(0)

        return random.choice(best_actions)

    def get_features(self, game_state, action):
        """
        This method returns a dictionary-like object where the keys are feature names and values are the feature values.
        """
        features = util.Counter()

        # Base variables
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        my_state = successor.get_agent_state(self.index)

        # === GAME AWARENESS ===
        # Opmerking: achteraf gezien blijkt deze feature niet nuttig te zijn voor de agent. De score teruggeven moedigt de agent niet aan om een specifieke actie te ondernemen. Onze gedachte was om een baseline te geven aan de gewichten zoals we gedaan hadden in een WPO uit het vorig semester.
        score = self.get_score(successor)
        features['score'] = score

        # === FOOD ===
        food_list = self.get_food(successor).as_list()
        current_food_list = self.get_food(game_state).as_list()
        if len(current_food_list) > len(food_list):
            features['distance_to_food'] = 0  # to stop going back and forth in front of food
            features['eat_food'] = 10  # motivate agent to eat food

        # Minimize distance to food pellets.
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # === ENEMIES ===
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        defenders = [enemy for enemy in enemies if enemy.get_position() is not None and not enemy.is_pacman]
        scared_defenders = [defender for defender in defenders if defender.scared_timer > 0]
        normal_defenders = [defender for defender in defenders if defender.scared_timer == 0]
        invaders = [enemy for enemy in enemies if enemy.get_position() is not None and enemy.is_pacman]

        # Minimize distance to invaders if agent is on his side.
        invader_distances = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
        if invaders and not my_state.is_pacman:
            features['invader_distance'] = min(invader_distances)

        # Minimize distance to defenders who are scared
        scared_defender_distances = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in scared_defenders]
        if scared_defenders:
            features['scared_defender_distance'] = min(scared_defender_distances)

        # Maximize distance to defenders who are not scared
        normal_defender_distances = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in normal_defenders]
        if normal_defenders:
            features['normal_defender_distance'] = min(normal_defender_distances)

        # The more food you carry, the more you should stay away from non-scared ghosts
        if my_state.num_carrying > 0:
            features['normal_defender_distance'] *= my_state.num_carrying * 1.2

        # === CAPSULES ===
        # Minimize distance to capsules if we can see non-scared ghosts
        capsules = self.get_capsules(successor)
        if len(capsules) > 0:
            min_capsule_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])
            if normal_defenders: # only interested in capsules if there are non-ghost enemies I can see
                features['distance_to_capsule'] = min_capsule_distance

        # === MOVEMENT ===
        # Discourage stopping and reversing
        features['stop'] = 1 if action == Directions.STOP else 0
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        features['reverse'] = 1 if action == reverse else 0

        # Discourage revisting previous positions
        if my_pos in self.previous_positions[-5:]:
            features['revisit'] = 1

        # Discourage choosing actions that have dead ends
        escape_actions = len([a for a in successor.get_legal_actions(self.index) if a != Directions.STOP])
        features['limited_escape'] = 1 if escape_actions <= 2 else 0

        # === RETURN STRATEGY ===
        # Cross the boundary if carrying food
        if my_state.is_pacman and my_state.num_carrying > 0:
            boundary_x = (game_state.data.layout.width // 2) - 1 if self.red else (game_state.data.layout.width // 2)
            boundary_positions = []

            for y in range(game_state.data.layout.height):
                if not game_state.has_wall(boundary_x, y):
                    boundary_positions.append((boundary_x,y))

            boundary_distances = [self.get_maze_distance(my_pos, pos) for pos in boundary_positions]
            min_boundary_distance = min(boundary_distances)

            return_factor = my_state.num_carrying

            if my_state.num_carrying >= 3 or (normal_defenders and min(normal_defender_distances) < 4):
                score = self.get_score(game_state)
                if score < 0:
                    # More aggressive return factor when losing
                    return_factor = my_state.num_carrying * 3
                else:  # Winning or tied
                    return_factor = my_state.num_carrying * 1.5

            features['return_with_food'] = min_boundary_distance * return_factor

        # === DEFENSE STRATEGY ===
        # Play defensive if winning significantly, there are invaders you can see
        # This code is a copy from the defensive agent
        features['defensive'] = 0
        if score >= 5 and invaders and my_state.scared_timer > 0 and my_state.is_pacman == False:
            features['defensive'] = 1
            features['scared_of_invader'] = 1

            if min(invader_distances) < 3:
                features['distance_to_invader'] = 3
            else:
                features['invader_chase'] = 1 / (1 + min(invader_distances))

        elif score >= 5 and invaders and my_state.scared_timer == 0 and my_state.is_pacman == False:
            features['defensive'] = 1
            features['invader_chase'] = 1 / (1 + min(invader_distances))

        return features

    def get_weights(self, game_state, action):
        """
        This method returns a dictionary where the keys are feature names and values are the feature weights.
        """
        features = self.get_features(game_state, action)

        if features['defensive'] == 0:
            return {
                # 'score': 50,
                'normal_defender_distance': 5,
                'scared_defender_distance': -3,
                'distance_to_food': -3,
                'eat_food': 20,
                'stop': -15,
                'reverse': -5,
                'revisit': -13,
                'invader_distance': -2,
                'distance_to_capsule': -1,
                'limited_escape': -3,
                'return_with_food': -2,
            }
        else:
            return {
                'distance_to_invader': -100,
                'invader_chase': 500,
                'scared_of_invader': 300,
            }


class DefensiveReflexAgent(ReflexCaptureAgent):
    def register_initial_state(self, game_state):

        super().register_initial_state(game_state)
        self.patrol_points = self.get_dynamic_patrol_points(game_state)
        self.patrol_target = random.choice(self.patrol_points)
    def get_dynamic_patrol_points(self, game_state):
        """Genereert flexibele patrouillepunten in het midden van de kaart."""
        mid_x = (game_state.data.layout.width // 2) - 1 if self.red else (game_state.data.layout.width // 2)
        patrol_positions = []

        for y in range(1, game_state.data.layout.height - 1, 4):
            if not game_state.has_wall(mid_x, y):
                patrol_positions.append((mid_x, y))

        return patrol_positions

    def get_features(self, game_state, action):

        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Basis verdediging
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        # Vind invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if invaders:
            # Zoek de dichtstbijzijnde invader
            distances = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
            min_distance = min(distances)
            features['invader_distance'] = min_distance

            # Stop met patrouilleren zodra een invader wordt gezien
            self.patrol_target = None

            # Controleer de scared status van de agent
            if my_state.scared_timer > 0:
                features['scared_of_invader'] = 1
                # Zorg ervoor dat de agent op minstens 3 stappen afstand blijft van de invader
                if min_distance < 3:
                    features['invader_distance'] = 3  # Houd altijd 3 stappen afstand van de invader
                else:
                    features['invader_chase'] = 1 / (1 + min_distance)
            else:
                # Als de agent niet bang is, kan hij de invader achtervolgen
                features['invader_chase'] = 1 / (1 + min_distance)
        else:
            # Geen invaders? Ga verder met patrouilleren
            if self.patrol_target is None or my_pos == self.patrol_target:
                self.patrol_target = random.choice(self.patrol_points)
            features['patrol_distance'] = self.get_maze_distance(my_pos, self.patrol_target)

        # Voorkom stilstand en nutteloze bewegingen
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """Geeft de weging van de kenmerken terug."""
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -100,
            'invader_chase': 500,
            'patrol_distance': -10,
            'scared_of_invader': 300,  # Beloning voor vermijden als tegenstander een capsule heeft
            'stop': -100,
            'reverse': -50
        }
