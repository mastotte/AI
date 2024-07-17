import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance

class Node:
    def __init__(self, state):
        self.adj = []
        self.value = float('-inf')
        self.state = state
        self.depth = -1
        self.best_move = ""
        self.came_from = ""
    
    def printNode(self):
        print("Came From: ", self.came_from, "Best Move: ", self.best_move)

def findPathToClosestDot(gameState):
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
    # capsules = gameState.getCapsules()
    # for c in capsules:
    #    x, y = c
    #    food[x][y] = True

    L = [(startPosition, [])]
    visited = [startPosition]
    while len(L) != 0:
        top = L.pop(0)
        cur, path = top
        x, y = cur

        if (food[x][y] is True):
            return path
        
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
    
    return []

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState, legalMoves):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]
    
    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        
        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        if (action == 'Stop'):
            return 0
        
        val = 0
        x, y = newPosition
        if (oldFood[x][y] is True):
            food_dist = []
        else:
            food_dist = findPathToClosestDot(successorGameState)
        
        ghost_distances = []
        for state in newGhostStates:
            ghost_distances.append(distance.manhattan(newPosition, state.getPosition()))

        ghost_dist = min(ghost_distances) + 0.01
        if (ghost_dist == 0.01):
            return float('-inf')

        f_dist = len(food_dist)
        
        if (f_dist != 0):
            ret = (1 / f_dist) + -(0.25 / (ghost_dist - 1)) * (0.25 / (ghost_dist - 1))
        else:
            ret = (1.05) + -(0.25 / (ghost_dist - 1)) * (0.25 / (ghost_dist - 1))
         
        return ret + val + currentGameState.getScore()

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # *** Your Code Here *** *
    def getAction(self, state):
        return self.value(state, self.getTreeDepth())
    
    def getTreeDepth(self):
        return super().getTreeDepth()
    
    def getEvaluationFunction(self):
        return super().getEvaluationFunction()
    
    def value(self, state, depth):
        best_move = None
        best_score = float('-inf')
        for d in state.getLegalActions(0):
            successor = state.generateSuccessor(0, d)
            v = self.min_value(successor, depth, 1)
            if v > best_score:
                best_score = v
                best_move = d

        return best_move
    
    def max_value(self, state, depth):
        if (depth == 0 or state.isLose() or state.isWin()):
            return self.getEvaluationFunction()(state)
        
        max_val = float('-inf')
        for d in state.getLegalActions(0):
            if (d != 'Stop'):
                successor = state.generateSuccessor(0, d)
                value = self.min_value(successor, depth, 1)
                max_val = max(max_val, value)

                # if (value >= max_val):
                #     max_val = value
                #    d = direction

        return max_val
        
    def min_value(self, state, depth, agentIndex):
        if (depth == 0 or state.isLose() or state.isWin()):
            return self.getEvaluationFunction()(state)
        
        min_val = float('inf')
        for direction in state.getLegalActions(agentIndex):
            if (direction != 'Stop'):
                successor = state.generateSuccessor(agentIndex, direction)
                if (agentIndex + 1 == state.getNumAgents()):
                    value = self.max_value(successor, depth - 1)
                else:
                    value = self.min_value(successor, depth, agentIndex + 1)

                min_val = min(min_val, value)
                # if (value <= min_val):
                #    min_val = value
                #    d = direction
        
        return min_val
  
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        v = self.AlphaBetaSearch(state)
        return v
     
    def getTreeDepth(self):
        return super().getTreeDepth()
    
    def getEvaluationFunction(self):
        return super().getEvaluationFunction()
    
    def AlphaBetaSearch(self, state):
        v = float('-inf')
        best_move = None

        for d in state.getLegalActions(0):
            if (d != 'Stop'):
                successor = state.generateSuccessor(0, d)
                score = self.min_value(successor, float('-inf'),
                                       float('inf'), self.getTreeDepth(), 1)
                if score > v:
                    v = score
                    best_move = d

        return best_move
    
    def max_value(self, state, alpha, beta, depth):
        if (state.isWin() or state.isLose() or depth == 0):
            return self.getEvaluationFunction()(state)
        
        legal = state.getLegalActions(0)
        v = float('-inf')
        for d in legal:
            if (d != 'Stop'):
                successor = state.generateSuccessor(0, d)
                v = max(v, self.min_value(successor, alpha, beta, depth, 1))
                if (v >= beta):
                    return v
                alpha = max(alpha, v)
        
        return v

    def min_value(self, state, alpha, beta, depth, agentIndex):
        if (state.isWin() or state.isLose() or depth == 0):
            return self.getEvaluationFunction()(state)
        
        v = float('inf')
        legal = state.getLegalActions(agentIndex)
        for d in legal:
            successor = state.generateSuccessor(agentIndex, d)
            if (agentIndex + 1 == state.getNumAgents()):
                v = min(v, self.max_value(successor, alpha, beta, depth - 1))
            else:
                v = min(v, self.min_value(successor, alpha, beta, depth, agentIndex + 1))
            if (v <= alpha):
                return v
            beta = min(beta, v)
        
        return v

  
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
    
    def getAction(self, state):
        return self.value(state, 0, self.getTreeDepth())[1]

    def getTreeDepth(self):
        return super().getTreeDepth()

    def getEvaluationFunction(self):
        return super().getEvaluationFunction()

    def value(self, state, agentIndex, depth):
        if (state.isWin() or state.isLose() or depth == 0):
            return self.getEvaluationFunction()(state), None

        if agentIndex == 0:
            return self.max_value(state, agentIndex, depth)
        else:
            return self.exp_value(state, agentIndex, depth)

    def max_value(self, state, agentIndex, depth):
        best_move = None
        v = float('-inf')

        for d in state.getLegalPacmanActions():
            successor = state.generatePacmanSuccessor(d)
            value, dir = self.value(successor, 1, depth - 1)

            if value > v:
                v = value
                best_move = d

        return v, best_move

    def exp_value(self, state, agentIndex, depth):
        val = 0
        legal_actions = state.getLegalActions(agentIndex)
        available_actions = len(legal_actions)

        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex, action)
            value, dir = self.value(successor, (agentIndex + 1) % state.getNumAgents(), depth)
            val += value

        if available_actions == 0:
            return self.getEvaluationFunction()(state), None

        ret = val / available_actions, None
        return ret
    
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: If the state is a win or lose, I return
    my value +-9999, so the +-9999 dominates the equation.
    Otherwise, I take the current game score + the reciprocal
    of the path to the closest dot.
    """
    if (currentGameState.isLose()):
        val = -9999
    elif (currentGameState.isWin()):
        val = 9999
    else:
        val = 0

    path = len(findPathToClosestDot(currentGameState))
    if (path == 0):
        path_return = 0
    else:
        path_return = 1 / path

    return currentGameState.getScore() + val + path_return

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
