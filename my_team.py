# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu)
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
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.

  Main Strategy:
  - Wisselwerking tussen attacker en defender, als attacker wordt gegeten, wissel de rollen

  Attacker Strategy:
  - maximize food eaten (through food length)
  - minimize risk of getting eaten
  - Use capsules strategically

  Specific strategies
  - if two ghosts in a certain radius, immediately return to home base, but still avoid the ghosts
  - take food gradually, first secure one single food pellet, then two, etc.
  - if time is almost up and winning significantly, go to your half and defend
  - if time is almost up and losing significantly, make both agents aggressors
  - consider tracking distance to nearest food cluster (built on top of closest food)
  - a longer path where more food is collected is preferrable to a shorter path with less food
  - change strategies based on how much food is eaten on your side
  - don't choose paths where un-scared ghosts travel (in relation to nearest food pellet), can be implemented with A*
  - on_defense like in DefensiveReflexAgent to check if you are on your side or not
  - as you are eating more food, the weight to return to your side increases
  - as you are deeper into enemy territory, the weight to return to your side increases
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)
        my_pos = successor.get_agent_state(self.index).get_position()
        my_state = successor.get_agent_state(self.index)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        scared_enemies = [enemy for enemy in enemies if
                          enemy.scared_timer > 0 and enemy.get_position() is not None]
        active_enemies = [enemy for enemy in enemies if
                          enemy.scared_timer == 0 and enemy.get_position() is not None]



        # If scared enemies are nearby, go for them
        if scared_enemies:
            min_dist = min([self.get_maze_distance(my_pos, enemy.get_position()) for enemy in scared_enemies])
            features['distance_to_scared_ghost'] = min_dist
            features['eat_scared_ghost'] = 1 if min_dist == 0 else 0
        else:
            features['distance_to_scared_ghost'] = 0
            features['eat_scared_ghost'] = 0

        # If defenders are nearby, avoid them
        defenders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if len(defenders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
            features['defender_distance'] = min(dists)

        # If active enemies are nearby, avoid them
        if active_enemies:
            min_distance = min([self.get_maze_distance(my_pos, enemy.get_position()) for enemy in active_enemies])
            if min_distance <= 4:
                features['distance_to_active_ghost'] = 1.0 / min_distance  # Higher if closer
            else:
                features['distance_to_active_ghost'] = 0
        else:
            features['distance_to_active_ghost'] = 0

        # If we are pacman and carrying 4 food, prioritize returning to our side
        if my_state.is_pacman and my_state.num_carrying == 4:
            mid_x = game_state.data.layout.width // 2 - (0 if self.red else 1) # midline
            depth_into_enemy = abs(my_pos[0] - mid_x)
            features['distance_to_midline'] = depth_into_enemy * (1.7 * my_state.num_carrying)

        # stolen from defender
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # If there are capsules and active enemies, prioritize eating them
        capsule_count = len(self.get_capsules(game_state))
        if capsule_count > 0 and active_enemies:
            min_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in self.get_capsules(game_state)])
            features['distance_to_capsule'] = min_distance

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1000,
                'distance_to_food': -1,
                'defender_distance': -10,
                'distance_to_active_ghost': -50,
                'stop': -100,
                'reverse': -5,
                'distance_to_midline': -40,
                'distance_to_capsule': -5,
                'distance_to_scared_ghost': -20,
                'eat_scared_ghost': 200}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Een defensieve agent die zich aanpast wanneer een power capsule is gepakt.
    """

    def register_initial_state(self, game_state):
        """Initialiseer de agent en bepaal patrouillepunten."""
        super().register_initial_state(game_state)
        self.patrol_points = self.get_dynamic_patrol_points(game_state)
        self.patrol_target = random.choice(self.patrol_points)
        self.initial_capsules = self.get_capsules_you_are_defending(game_state)

    def get_dynamic_patrol_points(self, game_state):
        """Genereert flexibele patrouillepunten in het midden van de kaart."""
        mid_x = (game_state.data.layout.width // 2) - 1 if self.red else (game_state.data.layout.width // 2)
        patrol_positions = []

        for y in range(1, game_state.data.layout.height - 1, 4):
            if not game_state.has_wall(mid_x, y):
                patrol_positions.append((mid_x, y))

        return patrol_positions

    def has_enemy_taken_capsule(self, game_state):
        """Checkt of er een capsule verdwenen is van onze kant."""
        current_capsules = self.get_capsules_you_are_defending(game_state)
        return len(current_capsules) < len(self.initial_capsules)

    def get_features(self, game_state, action):
        """Bepaalt de kenmerken die belangrijk zijn voor de beslissing."""
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Basis verdediging
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        enemy_has_capsule = self.has_enemy_taken_capsule(game_state)

        # Vind invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if invaders:
            # Zoek de dichtstbijzijnde vijand zonder lambda
            closest_invader = None
            min_distance = float("inf")

            for invader in invaders:
                distance = self.get_maze_distance(my_pos, invader.get_position())
                if distance < min_distance:
                    min_distance = distance
                    closest_invader = invader

            features['invader_distance'] = min_distance

            # Stop met patrouilleren zodra een vijand wordt gezien!
            self.patrol_target = None

            # Bang als vijand een capsule heeft gepakt
            if enemy_has_capsule:
                features['scared_of_invader'] = 1

            # Beloning voor indringers pakken (als we niet bang zijn)
            if my_state.scared_timer == 0 and not enemy_has_capsule:
                features['invader_chase'] = 1 / (1 + min_distance)
        else:
            # Geen vijand? Ga verder met patrouilleren
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
            'patrol_distance': -5,
            'scared_of_invader': 300,  # Beloning voor vermijden als tegenstander een capsule heeft
            'stop': -100,
            'reverse': -20
        }