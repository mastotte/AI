import random
from pacai.core.search.position import PositionSearchProblem
from pacai.student import search
from pacai.agents.capture.capture import CaptureAgent
from pacai.core import distance
from pacai.student.multiagents import ExpectimaxAgent
from pacai.student.multiagents import MinimaxAgent
from pacai.student.multiagents import ReflexAgent
from pacai.agents.base import BaseAgent
from pacai.core.directions import Directions
from pacai.util.priorityQueue import PriorityQueue
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.agents.capture.offense import OffensiveReflexAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent

# TO DO: route to closest spot on our side from enemy, not just to enemy
# offense?
def findPathToClosestDot(gameState, agentIndex):
    """
    Returns a path (a list of actions) to the closest dot, starting from gameState.
    """
    # Here are some useful elements of the startState
    # startPosition = gameState.getPacmanPosition()
    # food = gameState.getFood()
    # walls = gameState.getWalls()
    # problem = AnyFoodSearchProblem(gameState)

    # *** Your Code Here ***
    red_team = gameState.isOnRedTeam(agentIndex)
    startPosition = gameState.getAgentPosition(agentIndex)
    if red_team:
        food = gameState.getBlueFood()
    else:
        food = gameState.getRedFood()
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

def findPathToClosestDot(gameState, agentIndex):
    """
    Returns a path (a list of actions) to the closest dot, starting from gameState.
    """
    # Here are some useful elements of the startState
    # startPosition = gameState.getPacmanPosition()
    # food = gameState.getFood()
    # walls = gameState.getWalls()
    # problem = AnyFoodSearchProblem(gameState)

    # *** Your Code Here ***
    red_team = gameState.isOnRedTeam(agentIndex)
    startPosition = gameState.getAgentPosition(agentIndex)
    if red_team:
        food = gameState.getBlueFood()
        for x in food:
            print(x)

    else:
        food = gameState.getRedFood()
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

    def getAction(self, gameState, agentIndex):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions(agentIndex)
        self.agentIndex = agentIndex
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

        successorGameState = currentGameState.generateSuccessor(self.agentIndex, action)
        
        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        red_team = successorGameState.isOnRedTeam(self.agentIndex)
        if red_team:
            oldFood = currentGameState.getBlueFood()
            newGhostStates = successorGameState.getBlueTeamIndices()
        else:
            oldFood = currentGameState.getRedFood()
            newGhostStates = successorGameState.getRedTeamIndices()

        newPosition = successorGameState.getAgentPosition(self.agentIndex)
        

        if (action == 'Stop'):
            return 0
        
        val = 0
        x, y = newPosition
        if (oldFood[x][y] is True):
            food_dist = []
        else:
            food_dist = findPathToClosestDot(successorGameState, self.agentIndex)
        
        ghost_distances = []
        for state in newGhostStates:
            ghost_distances.append(distance.manhattan(newPosition, currentGameState.getAgentPosition(state)))

        ghost_dist = min(ghost_distances) + 0.01
        if (ghost_dist == 0.01):
            return float('-inf')

        f_dist = len(food_dist)
        
        if (f_dist != 0):
            ret = (1 / f_dist) + -(0.25 / (ghost_dist - 1)) * (0.25 / (ghost_dist - 1))
        else:
            ret = (1.05) + -(0.25 / (ghost_dist - 1)) * (0.25 / (ghost_dist - 1))
         
        return ret + val + currentGameState.getScore()
    
class SoloDefenseAgent(CaptureAgent):

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def chooseAction(self, gameState, agentIndex, legalMoves, width):

        print("Solo Defense Moves: ",legalMoves)
        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, agentIndex, action, width) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]
    
    def evaluationFunction(self, gameState, agentIndex, action, width):
        if (action == 'Stop'):
            return -998
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        my_pos = successorGameState.getAgentPosition(agentIndex)
        red_team = successorGameState.isOnRedTeam(agentIndex)
        blue_side = successorGameState.isOnBlueSide(my_pos)
        if (red_team and blue_side):
            return -999
        
        if ((not red_team) and (not blue_side)):
            return -999
        
        if red_team:
            opponents = successorGameState.getBlueTeamIndices()
        else:
            opponents = successorGameState.getRedTeamIndices()

        print("Opp: ",opponents, "Me: ",agentIndex)
        #enemies = [successorGameState.getAgentState(i) for i in opponents]
        if red_team:
            invaders = [gameState.getAgentPosition(a) for a in opponents if gameState.getAgentPosition(a) is not None and gameState.isOnRedSide(gameState.getAgentPosition(a))]
        else:
            invaders = [gameState.getAgentPosition(a) for a in opponents if gameState.getAgentPosition(a) is not None and gameState.isOnBlueSide(gameState.getAgentPosition(a))]

        if (len(invaders) == 0):
            invaders = [gameState.getAgentPosition(a) for a in opponents if gameState.getAgentPosition(a) is not None]

        my_pos = successorGameState.getAgentPosition(agentIndex)
        me_to_invader = min(distance.maze(my_pos, invader, successorGameState, red_team, width) for invader in invaders)

        return 1/ (me_to_invader + 0.02)
    
class TeamDefenseAgent(CaptureAgent):

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def chooseAction(self, gameState, agentIndex, allyIndex, legalMoves):

        
        print("Team Defense Moves: ",legalMoves)
        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, agentIndex, allyIndex, action) for action in legalMoves]
        print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]
    
    def evaluationFunction(self, gameState, agentIndex, allyIndex, action):
        
        if (action == 'Stop'):
            return -998
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        my_pos = successorGameState.getAgentPosition(agentIndex)
        red_team = successorGameState.isOnRedTeam(agentIndex)
        blue_side = successorGameState.isOnBlueSide(my_pos)


        if (red_team and blue_side):
            return -999
        
        if ((not red_team) and (not blue_side)):
            return -999
        
        if red_team:
            opponents = gameState.getBlueTeamIndices()
        else:
            opponents = gameState.getRedTeamIndices()

        print("Opp: ",opponents, "Me: ",agentIndex)
        #enemies = [successorGameState.getAgentState(i) for i in opponents]
        invaders = [gameState.getAgentPosition(a) for a in opponents if gameState.getAgentPosition(a) is not None]
        
        ally_pos = successorGameState.getAgentPosition(allyIndex)

        
        me_to_invader1 = distance.maze(my_pos, invaders[0], successorGameState, red_team, 32)
        me_to_invader2 = distance.maze(my_pos, invaders[1], successorGameState, red_team, 32)
        ally_to_invader1 = distance.maze(ally_pos, invaders[0], successorGameState, red_team, 32)
        ally_to_invader2 = distance.maze(ally_pos, invaders[1], successorGameState, red_team, 32)

        if (me_to_invader1 < 1 or me_to_invader2 < 1):
            return 999
        
        if (ally_to_invader1 + me_to_invader2 > me_to_invader1 + ally_to_invader2):
            return 1/ (me_to_invader1 + 0.02)
        else:
            return 1/ (me_to_invader2 + 0.02)
        

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)
        self.offensiveAgent = OffensiveReflexAgent(self.index)
        self.offensiveAgent.registerInitialState(gameState)

        # Your initialization code goes here, if you need any.
        self.max_time = gameState.time()
        self.height = gameState._layout.height
        self.width = gameState._layout.width
        self.midWidth = self.width/2
        self.midHeight = self.height/2
        print("Index before: ",self.index)
        #self.index = gameState.getAgentState()
        print("Index after: ",self.index)
        if (self.index == 0):
            self.team = "blue"
            self.ally_index = 2
        
        if (self.index == 1):
            self.team = "red"
            self.ally_index = 3

        if (self.index == 2):
            self.team = "blue"
            self.ally_index = 0
        
        if (self.index == 3):
            self.team = "red"
            self.ally_index = 1

        if (self.index == 1 or self.index == 3):
            self.enemy1_index = 0
            self.enemy2_index = 2
        else:
            self.enemy1_index = 1
            self.enemy2_index = 3

        


    def chooseAction(self, gameState):
        time = gameState.time()
        print("Time: ",time)
        if self.onOffense(gameState, self.index) is True or gameState.time() > self.max_time-130:
            action = self.getOffensiveMove(gameState)

        elif self.onOffense(gameState, self.index) is False:
            action = self.getDefensiveMove(gameState)

        return action
    
    def onOffense(self, gameState, agentIndex):
        my_pos = gameState.getAgentPosition(agentIndex)
        red_team = gameState.isOnRedTeam(agentIndex)
        blue_side = gameState.isOnBlueSide(my_pos)

        if (red_team and blue_side):
            return True
        
        elif ((not red_team) and (not blue_side)):
            return True

        else:
            return False
    
    def findPath(situation, position1, position2, gameState):
        x1, y1 = position1
        x2, y2 = position2

        walls = gameState.getWalls()

        if (walls[x1][y1]):
            raise ValueError('Position1 is a wall: ' + str(position1))

        if (walls[x2][y2]):
            raise ValueError('Position2 is a wall: ' + str(position2))

        prob = PositionSearchProblem(gameState, start = position1, goal = position2)

        return search.breadthFirstSearch(prob)
    
    def getZone(self, position):
        x, y = position
        if (self.recentlyDied(position)):
            return -1
        
        zone = 0
        if (x > self.midWidth):
            if (y > self.midHeight):
                zone = 2
            else:
                zone = 4
        
        if (x <= self.midWidth):
            if (y > self.midHeight):
                zone = 1
            else:
                zone = 3

        return zone
    
    def enemyRecentlyDied(self, gameState):
        red_team = gameState.isOnRedTeam(self.index)
        if red_team:
            enemy1_index, enemy2_index = gameState.getBlueTeamIndices()
        else:
            enemy1_index, enemy2_index = gameState.getRedTeamIndices()
        enemy1_pos = gameState.getAgentPosition(enemy1_index)
        enemy2_pos = gameState.getAgentPosition(enemy2_index)
        x1, _ = enemy1_pos
        x2, _ = enemy2_pos

        if red_team:
            if (x1 >= self.width - 2 or x2 >= self.width - 2):
                return True
        
        if not red_team:
            if (x1 <= 1 or x2 <= 1):
                return True

        return False

    def recentlyDied(self, position):
        x, y = position
        print("X: ",x)
        if self.team == "blue":
            if (x >= self.width - 2):
                return True
        
        if self.team == "red":
            if (x <= 2):
                return True
        
        return False
    
    def getGhostStates(self, gameState):
        ghost_pos = []
        enemy1_pos = gameState.getAgentPosition(self.enemy1_index)
        enemy2_pos = gameState.getAgentPosition(self.enemy2_index)
        z1 = self.getZone(enemy1_pos)
        z2 = self.getZone(enemy2_pos)
        if self.team == "red":
            if z1 % 2 == 0 and z1 != -1:
                ghost_pos.append(enemy1_pos)
            if z2 % 2 == 0 and z2 != -1:
                ghost_pos.append(enemy2_pos)
        
        if self.team == "blue":
            if z1 % 2 == 1 and z1 != -1:
                ghost_pos.append(enemy1_pos)
            if z2 % 2 == 1 and z2 != -1:
                ghost_pos.append(enemy2_pos)

        return ghost_pos

    def getDefensiveMove(self, gameState):  # returns action towards assignment
        print("Get defensive move")
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        my_pos = gameState.getAgentPosition(self.index)
        ally_pos = gameState.getAgentPosition(self.ally_index)

        if (self.onOffense(gameState, self.ally_index)):
            print("getting solo")
            return self.getSoloDefenseMove(gameState)
        else:
            if self.enemyRecentlyDied(gameState):
                print("defense to offense")
                return self.getOffensiveMove(gameState)
            else:
                print("getting team")
                return self.getTeamDefenseMove(gameState)
        if (invaders):
            if (self.onOffense(gameState, self.ally_index)):
                me_to_invader = min(distance.maze(my_pos, invader, gameState) for invader in invaders)
                return self.getSoloDefenseMove(gameState)
            else:
                if (len(invaders) == 1):
                    ally_to_invader = distance.maze(ally_pos, invaders[0], gameState)
                    me_to_invader = distance.maze(my_pos, invaders[0], gameState)
                    if (ally_to_invader < me_to_invader):
                        if (self.enemyRecentlyDied(gameState)):
                            return self.getOffensiveMove(gameState)
                        else:
                            pos = self.getGhostStates(gameState)
                            dist = distance.maze(my_pos, pos, gameState)
                            return 1/dist
                        
    def getSoloDefenseMove(self, gameState):
        prob = SoloDefenseAgent(gameState)
        legalMoves = gameState.getLegalActions(self.index)
        action = prob.chooseAction(gameState, self.index, legalMoves, self.width)
        return action
    
    def getTeamDefenseMove(self, gameState):
        prob = TeamDefenseAgent(gameState)
        legalMoves = gameState.getLegalActions(self.index)
        action = prob.chooseAction(gameState, self.index, self.ally_index, legalMoves)
        return action
    
    def getOffensiveMove(self, gameState):
        return self.offensiveAgent.chooseAction(gameState)
    
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
        
        startPosition = gameState.getAgentPosition(self.index)
        walls = gameState.getWalls()
        if (self.team == "red"):
            food = gameState.getBlueFood()
        else:
            food = gameState.getRedFood()
        
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
    
    def chooseAttackAction(self, gameState):
        """
        Chooses an action in attack mode, finding the nearest food and avoiding enemies.
        """
        foodList = self.getFood(gameState).asList()
        if len(foodList) > 0:
            myPos = gameState.getAgentPosition(self.index)
            # Implement a food scoring system
            bestFood, bestScore = None, float('inf')
            for food in foodList:
                distance = self.getMazeDistance(myPos, food)
                riskScore = self.assessRisk(gameState, food)
                score = distance + riskScore
                if score < bestScore:
                    bestScore = score
                    bestFood = food

            if bestFood:
                return self.modifiedAStarSearch(gameState, bestFood)

        return Directions.STOP

    def assessRisk(self, gameState, target):
        """
        Assess the risk of moving towards a target.
        """
        myPos = gameState.getAgentPosition(self.index)
        ghosts = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghostPositions = [ghost.getPosition() for ghost in ghosts if not ghost.isPacman and ghost.getPosition() != None]

        riskScore = 0
        for ghostPos in ghostPositions:
            distance = self.getMazeDistance(myPos, ghostPos)
            if distance < 1:
                # Increase risk score based on proximity
                riskScore += (3 - distance) * 10  # Higher score for closer ghosts

                if gameState.getAgentState(self.index).isScared():
                    # Further increase the risk if our agent is scared
                    riskScore += 20

        return riskScore


    def modifiedAStarSearch(self, gameState, target):
        """
        Improved A* algorithm that dynamically adjusts path costs.
        """
        startPosition = gameState.getAgentPosition(self.index)
        if startPosition == target:
            return Directions.STOP

        frontier = PriorityQueue()
        frontier.push((startPosition, []), 0)
        explored = set()
        ghosts = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghostPositions = [ghost.getPosition() for ghost in ghosts if not ghost.isPacman and ghost.getPosition() != None]

        while not frontier.isEmpty():
            position, path = frontier.pop()
            if position == target:
                return path[0]  # Return the first action of the path.

            explored.add(position)
            for successor, action, _ in self.getSuccessors(gameState, position):
                if successor not in explored:
                    new_path = path + [action]
                    cost = len(new_path)

                    # Dynamic cost adjustment based on ghost proximity
                    for ghostPos in ghostPositions:
                        ghostDistance = self.getMazeDistance(successor, ghostPos)
                        if ghostDistance < 2:
                            cost += 20  # Higher cost for paths near ghosts
                        elif ghostDistance < 5:
                            cost += 10  # Moderate cost increase for paths somewhat near ghosts

                    heuristic = self.getMazeDistance(successor, target)
                    frontier.push((successor, new_path), cost + heuristic)

        return Directions.STOP


        

