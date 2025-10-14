# logicAgents.py
# --------------
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a LogicAgent that uses
logicPlan.position_logic_plan, run the following command:

> python pacman.py -p LogicAgent -a fn=position_logic_plan

Commands to invoke other planning methods can be found in the project
description.

You should NOT change code in this file

Good luck and happy planning!
"""

from game import Directions
from game import Agent
from game import Actions
from game import Grid
from graphicsUtils import *
import util
import time
import warnings
import logicPlan
import random

util.VALIDATION_LISTS['search'] = [
        "^(@)$_",
        "हिंदीखरीदारी",
        "\\u200cآمباردا",
        "ſammen",
        " coachTry",
        "ſſung",
        " AcceptedLoading",
        "EnglishChoose",
        " queſto",
        " queſta"
]

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def get_action(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.legal_pacman_actions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of logicPlan.py       #
#######################################################

class LogicAgent(Agent):
    """
    This very general logic agent finds a path using a supplied planning
    algorithm for a supplied planning problem, then returns actions to follow that
    path.

    As a default, this agent runs position_logic_plan on a
    PositionPlanningProblem to find location (1,1)

    Options for fn include:
      position_logic_plan or plp
      food_logic_plan or flp
      food_ghost_logic_plan or fglp


    Note: You should NOT change any code in LogicAgent
    """

    def __init__(self, fn='position_logic_plan', prob='PositionPlanningProblem', plan_mod=logicPlan):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the planning function from the name and heuristic
        if fn not in dir(plan_mod):
            raise AttributeError(fn + ' is not a planning function in logicPlan.py.')
        func = getattr(plan_mod, fn)
        self.planning_function = lambda x: func(x)

        # Get the planning problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a planning problem type in logicAgents.py.')
        self.plan_type = globals()[prob]
        self.live_checking = False
        print('[LogicAgent] using problem type ' + prob)

    def register_initial_state(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.planning_function == None:
            raise Exception("No planning function provided for LogicAgent")
        starttime = time.time()
        problem = self.plan_type(state) # Makes a new planning problem

        self.actions = [] # In case planning_function times out
        self.actions  = self.planning_function(problem) # Find a path
        totalCost = problem.get_cost_of_actions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        # TODO Drop
        if '_expanded' in dir(problem):
            print('Nodes expanded: %d' % problem._expanded)

    def get_action(self, state):
        """
        Returns the next action in the path chosen earlier (in
        register_initial_state).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        # import ipdb; ipdb.set_trace()
        if 'action_index' not in dir(self): self.action_index = 0
        i = self.action_index
        self.action_index += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            print('Oh no! The Pacman agent created a plan that was too short!')
            print()
            return None
            # return Directions.STOP

class CheckSatisfiabilityAgent(LogicAgent):
    def __init__(self, fn='check_location_satisfiability', prob='LocMapProblem', plan_mod=logicPlan):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the planning function from the name and heuristic
        if fn not in dir(plan_mod):
            raise AttributeError(fn + ' is not a planning function in logicPlan.py.')
        func = getattr(plan_mod, fn)
        self.planning_function = lambda x: func(*x)

        # Get the planning problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a planning problem type in logicAgents.py.')
        self.plan_type = globals()[prob]
        print('[LogicAgent] using problem type ' + prob)
        self.live_checking = False

    def register_initial_state(self, state):
        if self.planning_function == None:
            raise Exception("No planning function provided for LogicAgent")
        starttime = time.time()
        self.problem = self.plan_type(state) # Makes a new planning problem

    def get_action(self, state):
        return "EndGame"

class LocalizeMapAgent(LogicAgent):
    """Parent class for localization, mapping, and slam"""
    def __init__(self, fn='position_logic_plan', prob='LocMapProblem', plan_mod=logicPlan, display=None, scripted_actions=[]):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the planning function from the name and heuristic
        if fn not in dir(plan_mod):
            raise AttributeError(fn + ' is not a planning function in logicPlan.py.')
        func = getattr(plan_mod, fn)
        self.planning_function = lambda x, y: func(x, y)

        # Get the planning problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a planning problem type in logicAgents.py.')
        self.plan_type = globals()[prob]
        print('[LogicAgent] using problem type ' + prob)
        self.visited_states = []
        self.display = display
        self.scripted_actions = scripted_actions
        self.live_checking = True

    def resetLocation(self):
        self.visited_states = []
        self.state = self.problem.get_start_state()
        self.visited_states.append(self.state)

    def addNoOp_t0(self):
        self.visited_states = [self.visited_states[0]] + list(self.visited_states)
        self.actions.insert(0, "Stop")

    def register_initial_state(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.planning_function == None:
            raise Exception("No planning function provided for LogicAgent")
        starttime = time.time()
        problem = self.plan_type(state) # Makes a new planning problem

        self.problem = problem
        self.state = self.problem.get_start_state()

        self.actions = self.scripted_actions
        self.resetLocation()
        self.planning_fn_output = self.planning_function(problem, self)
        # self.addNoOp_t0()

    def get_known_walls_non_walls_from_known_map(self, known_map):
        # map is 1 for known wall, 0 for 
        known_walls = [[(True if entry==1 else False) for entry in row] for row in known_map]
        known_non_walls = [[(True if entry==0 else False) for entry in row] for row in known_map]
        return known_walls, known_non_walls

class LocalizationLogicAgent(LocalizeMapAgent):
    def __init__(self, fn='localization', prob='LocalizationProblem', plan_mod=logicPlan, display=None, scripted_actions=[]):
        super(LocalizationLogicAgent, self).__init__(fn, prob, plan_mod, display, scripted_actions)
        self.num_timesteps = len(scripted_actions) if scripted_actions else 5

    def get_action(self, state):
        """
        Returns the next action in the path chosen earlier (in
        register_initial_state).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        # import ipdb; ipdb.set_trace()
        if 'action_index' not in dir(self): self.action_index = 0
        i = self.action_index
        self.action_index += 1

        planning_fn_output = None
        if i < self.num_timesteps:
            proposed_action = self.actions[i]
            planning_fn_output = next(self.planning_fn_output)
            self.drawPossibleStates(planning_fn_output, direction=self.actions[i])
        elif i < len(self.actions):
            proposed_action = self.actions[i]
        else:
            proposed_action = "EndGame"

        return proposed_action, planning_fn_output

    def move_to_next_state(self, action):
        oldX, oldY = self.state
        dx, dy = Actions.directionToVector(action)
        x, y = int(oldX + dx), int(oldY + dy)
        if self.problem.walls[x][y]:
            raise AssertionError("Taking an action that goes into wall")
            pass
        else:
            self.state = (x, y)
        self.visited_states.append(self.state)

    def get_percepts(self):
        x, y = self.state
        north_iswall = self.problem.walls[x][y+1]
        south_iswall = self.problem.walls[x][y-1]
        east_iswall = self.problem.walls[x+1][y]
        west_iswall = self.problem.walls[x-1][y]
        return [north_iswall, south_iswall, east_iswall, west_iswall]

    def get_valid_actions(self):
        x, y = self.state
        actions = []
        if not self.problem.walls[x][y+1]: actions.append('North')
        if not self.problem.walls[x][y-1]: actions.append('South')
        if not self.problem.walls[x+1][y]: actions.append('East')
        if not self.problem.walls[x-1][y]: actions.append('West')
        return actions

    def drawPossibleStates(self, possible_locations=None, direction="North", pacman_position=None):
        import __main__
        self.display.clearExpandedCells() # Erase previous colors
        self.display.colorCircleCells(possible_locations, direction=direction, pacman_position=pacman_position)

class MappingLogicAgent(LocalizeMapAgent):
    def __init__(self, fn='mapping', prob='MappingProblem', plan_mod=logicPlan, display=None, scripted_actions=[]):
        super(MappingLogicAgent, self).__init__(fn, prob, plan_mod, display, scripted_actions)
        self.num_timesteps = len(scripted_actions) if scripted_actions else 10

    def get_action(self, state):
        """
        Returns the next action in the path chosen earlier (in
        register_initial_state).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'action_index' not in dir(self): self.action_index = 0
        i = self.action_index
        self.action_index += 1

        planning_fn_output = None
        if i < self.num_timesteps:
            proposed_action = self.actions[i]
            planning_fn_output = next(self.planning_fn_output)
            self.draw_wall_beliefs(planning_fn_output, self.actions[i], self.visited_states[:i])
        elif i < len(self.actions):
            proposed_action = self.actions[i]
        else:
            proposed_action = "EndGame"

        return proposed_action, planning_fn_output

    def move_to_next_state(self, action):
        oldX, oldY = self.state
        dx, dy = Actions.directionToVector(action)
        x, y = int(oldX + dx), int(oldY + dy)
        if self.problem.walls[x][y]:
            raise AssertionError("Taking an action that goes into wall")
            pass
        else:
            self.state = (x, y)
        self.visited_states.append(self.state)

    def get_percepts(self):
        x, y = self.state
        north_iswall = self.problem.walls[x][y+1]
        south_iswall = self.problem.walls[x][y-1]
        east_iswall = self.problem.walls[x+1][y]
        west_iswall = self.problem.walls[x-1][y]
        return [north_iswall, south_iswall, east_iswall, west_iswall]

    def get_valid_actions(self):
        x, y = self.state
        actions = []
        if not self.problem.walls[x][y+1]: actions.append('North')
        if not self.problem.walls[x][y-1]: actions.append('South')
        if not self.problem.walls[x+1][y]: actions.append('East')
        if not self.problem.walls[x-1][y]: actions.append('West')
        return actions

    def draw_wall_beliefs(self, known_map=None, direction="North", visited_states_to_render=[]):
        import random
        import __main__
        from graphicsUtils import draw_background, refresh
        known_walls, known_non_walls = self.get_known_walls_non_walls_from_known_map(known_map)
        wall_grid = Grid(self.problem.walls.width, self.problem.walls.height, initial_value=False)
        wall_grid.data = known_walls
        all_true_wall_grid = Grid(self.problem.walls.width, self.problem.walls.height, initial_value=True)
        self.display.clearExpandedCells() # Erase previous colors
        self.display.drawWalls(wall_grid, formatColor(.9,0,0), all_true_wall_grid)
        refresh()

class SLAMLogicAgent(LocalizeMapAgent):
    def __init__(self, fn='slam', prob='SLAMProblem', plan_mod=logicPlan, display=None, scripted_actions=[]):
        super(SLAMLogicAgent, self).__init__(fn, prob, plan_mod, display, scripted_actions)
        self.scripted_actions = scripted_actions
        self.num_timesteps = len(self.scripted_actions) if self.scripted_actions else 10
        self.live_checking = True

    def get_action(self, state):
        """
        Returns the next action in the path chosen earlier (in
        register_initial_state).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        # import ipdb; ipdb.set_trace()
        if 'action_index' not in dir(self): self.action_index = 0
        i = self.action_index
        self.action_index += 1
        pacman_loc = self.visited_states[i]

        planning_fn_output = None
        if i < self.num_timesteps:
            proposed_action = self.actions[i]
            planning_fn_output = next(self.planning_fn_output)
            self.draw_wall_and_position_beliefs(
                known_map=planning_fn_output[0],
                possible_locations=planning_fn_output[1],
                direction=self.actions[i])
        elif i < len(self.actions):
            proposed_action = self.actions[i]
        else:
            proposed_action = "EndGame"

        # SLAM needs to handle illegal actions
        if proposed_action not in self.get_valid_actions(pacman_loc) and proposed_action not in ["Stop", "EndGame"]:
            proposed_action = "Stop"

        return proposed_action, planning_fn_output

    def move_to_next_state(self, action):
        oldX, oldY = self.state
        dx, dy = Actions.directionToVector(action)
        x, y = int(oldX + dx), int(oldY + dy)
        if self.problem.walls[x][y]:
            # raise AssertionError("Taking an action that goes into wall")
            pass
        else:
            self.state = (x, y)
        self.visited_states.append(self.state)

    def get_percepts(self):
        x, y = self.state
        north_iswall = self.problem.walls[x][y+1]
        south_iswall = self.problem.walls[x][y-1]
        east_iswall = self.problem.walls[x+1][y]
        west_iswall = self.problem.walls[x-1][y]
        num_adj_walls = sum([north_iswall, south_iswall, east_iswall, west_iswall])
        # percept format: [adj_to_>=1_wall, adj_to_>=2_wall, adj_to_>=3_wall]
        percept = [num_adj_walls >= i for i in range(1, 4)]
        return percept

    def get_valid_actions(self, state=None):
        if not state:
            state = self.state
        x, y = state
        actions = []
        if not self.problem.walls[x][y+1]: actions.append('North')
        if not self.problem.walls[x][y-1]: actions.append('South')
        if not self.problem.walls[x+1][y]: actions.append('East')
        if not self.problem.walls[x-1][y]: actions.append('West')
        return actions

    def draw_wall_and_position_beliefs(self, known_map=None, possible_locations=None,
            direction="North", visited_states_to_render=[], pacman_position=None):
        import random
        import __main__
        from graphicsUtils import draw_background, refresh
        known_walls, known_non_walls = self.get_known_walls_non_walls_from_known_map(known_map)
        wall_grid = Grid(self.problem.walls.width, self.problem.walls.height, initial_value=False)
        wall_grid.data = known_walls
        all_true_wall_grid = Grid(self.problem.walls.width, self.problem.walls.height, initial_value=True)

        # Recover list of non-wall coords:
        non_wall_coords = []
        for x in range(len(known_non_walls)):
            for y in range(len(known_non_walls[x])):
                if known_non_walls[x][y] == 1:
                    non_wall_coords.append((x, y))

        self.display.clearExpandedCells() # Erase previous colors

        self.display.drawWalls(wall_grid, formatColor(.9,0,0), all_true_wall_grid)
        self.display.colorCircleSquareCells(possible_locations, square_cells=non_wall_coords, direction=direction, pacman_position=pacman_position)
        refresh()

class PositionPlanningProblem(logicPlan.PlanningProblem):
    """
    A planning problem defines the state space, start state, goal test, successor
    function and cost function.  This planning problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this planning problem is fully specified; you should NOT change it.
    """

    def __init__(self, game_state, cost_fn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        game_state: A GameState object (pacman.py)
        cost_fn: A function from a planning state (tuple) to a non-negative number
        goal: A position in the game_state
        """
        self.walls = game_state.get_walls()
        self.start_state = game_state.get_pacman_position()
        if start != None: self.start_state = start
        self.goal = goal
        self.cost_fn = cost_fn
        self.visualize = visualize
        if warn and (game_state.get_num_food() != 1 or not game_state.has_food(*goal)):
            print('Warning: this does not look like a regular position planning maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def get_start_state(self):
        return self.start_state

    def get_goal_state(self):
        return self.goal

    def get_cost_of_actions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999. 

        This is included in the logic project solely for autograding purposes.
        You should not be calling it.
        """
        if actions == None: return 999999
        x,y= self.get_start_state()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.cost_fn((x,y))
        return cost
    
    def get_width(self):
        """
        Returns the width of the playable grid (does not include the external wall)
        Possible x positions for agents will be in range [1,width]
        """
        return self.walls.width-2

    def get_height(self):
        """
        Returns the height of the playable grid (does not include the external wall)
        Possible y positions for agents will be in range [1,height]
        """
        return self.walls.height-2

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionPlanningProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionPlanningProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

class LocMapProblem:
    """Parent class for Localization, Mapping, and SLAM."""
    def __init__(self, game_state, cost_fn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        self.walls = game_state.get_walls()
        self.start_state = game_state.get_pacman_position()
        if start != None: self.start_state = start
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def get_start_state(self):
        return self.start_state

    def get_width(self):
        """
        Returns the width of the playable grid (does not include the external wall)
        Possible x positions for agents will be in range [1,width]
        """
        return self.walls.width-2

    def get_height(self):
        """
        Returns the height of the playable grid (does not include the external wall)
        Possible y positions for agents will be in range [1,height]
        """
        return self.walls.height-2

class LocalizationProblem(LocMapProblem):
    pass

class MappingProblem(LocMapProblem):
    pass

class SLAMProblem(LocMapProblem):
    pass

class FoodPlanningProblem:
    """
    A planning problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A planning state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, starting_game_state):
        self.start = (starting_game_state.get_pacman_position(), starting_game_state.get_food())
        self.walls = starting_game_state.get_walls()
        self.starting_game_state = starting_game_state
        self._expanded = 0 # DO NOT CHANGE
        self.heuristic_info = {} # A dictionary for the heuristic to store information

    def get_start_state(self):
        return self.start

    def get_cost_of_actions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999. 

        This is included in the logic project solely for autograding purposes.
        You should not be calling it.
        """
        x,y= self.get_start_state()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost
    
    def get_width(self):
        """
        Returns the width of the playable grid (does not include the external wall)
        Possible x positions for agents will be in range [1,width]
        """
        return self.walls.width-2

    def get_height(self):
        """
        Returns the height of the playable grid (does not include the external wall)
        Possible y positions for agents will be in range [1,height]
        """
        return self.walls.height-2
