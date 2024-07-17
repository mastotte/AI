"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.student import search
from pacai.core.actions import Actions
from pacai.core import distance
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core.directions import Directions

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """
    def __init__(self, startingGameState):
        super().__init__()
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2
        self.startingGameState = startingGameState
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

        # *** Your Code Here ***
        self.visitedCorners = set()
        return None
    
    def startingState(self):
        return (self.startingPosition, (0, 0, 0, 0))
    
    def isGoal(self, state):
        if (state[1] == (1, 1, 1, 1)):
            return True
        
        return False

    def successorStates(self, state):

        currentPosition, vC = state
        visitedCorners = list(vC)
        for i in range(0, 4):
            if (currentPosition == self.corners[i]):
                visitedCorners[i] = 1
                
        successors = []
        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if not hitsWall:
                nextPosition = (nextx, nexty)
                successors.append(((nextPosition, tuple(visitedCorners)), action, 1))

        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)
    
def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """
    # Nearest unseen corner, plus manhattan distance of path through unseen corners
    coord, visitedCorners = state

    corners = problem.corners
    unvisitedCorners = []
    cur_pos = coord
    ret = 0

    for i in range(0, 4):
        if not visitedCorners[i]:
            unvisitedCorners.append(corners[i])
    
    while len(unvisitedCorners) != 0:
        h_values = []
        for corner in unvisitedCorners:
            dist = distance.manhattan(cur_pos, corner)
            h_values.append(dist)
        
        min_dist = min(h_values)
        min_corner = corners[h_values.index(min_dist)]

        ret += min_dist
        cur_pos = min_corner
        unvisitedCorners.remove(min_corner)

    return ret

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """
    position, foodGrid = state
    if (position in foodGrid.asList()):
        foodGrid.asList().remove(position)

    problem.heuristicInfo['unvisitedFood'] = foodGrid.asList()

    unvisitedFood = problem.heuristicInfo['unvisitedFood']

    if not unvisitedFood:
        return 0
        
    max_dist = -999
    for food in unvisitedFood:
        p = PositionSearchProblem(problem.startingGameState, costFn=lambda x: 1, goal=food,
                                start=position)
        path = search.breadthFirstSearch(p)
        dist = len(path)
        if (dist > max_dist):
            max_dist = dist

    return max_dist

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """
        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # problem = AnyFoodSearchProblem(gameState)

        # *** Your Code Here ***
        startPosition = gameState.getPacmanPosition()
        walls = gameState.getWalls()
        food = gameState.getFood()
        L = [(startPosition, [])]
        visited = [startPosition]
        while len(L) != 0:
            top = L.pop(0)
            cur, path = top
            x, y = cur
            if (x + 1, y) not in visited:
                if (walls[x + 1][y] is False):
                    node = ((x + 1, y), path + ['East'])
                    if (food[x + 1][y] is True):
                        return path + ['East']
                    else:
                        L.append(node)
                        visited.append((x + 1, y))

            if (x - 1, y) not in visited:
                if (walls[x - 1][y] is False):
                    node = ((x - 1, y), path + ['West'])
                    if (food[x - 1][y] is True):
                        return path + ['West']
                    else:
                        L.append(node)
                        visited.append((x - 1, y))

            if (x, y + 1) not in visited:
                if (walls[x][y + 1] is False):
                    node = ((x, y + 1), path + ['North'])
                    if (food[x][y + 1] is True):
                        return path + ['North']
                    else:
                        L.append(node)
                        visited.append((x, y + 1))

            if (x, y - 1) not in visited:
                if (walls[x][y - 1] is False):
                    node = ((x, y - 1), path + ['South'])
                    if (food[x][y - 1] is True):
                        return path + ['South']
                    else:
                        L.append(node)
                        visited.append((x, y - 1))
        
        return 0

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood()

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
