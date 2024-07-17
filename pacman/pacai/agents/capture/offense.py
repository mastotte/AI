from pacai.agents.capture.reflex import ReflexCaptureAgent

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
          
    def __init__(self, index, **kwargs):
        super().__init__(index)
        #print("init offense")

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        return features

    def getWeights(self, gameState, action):
        #print("get weights offense")
        return {
            'successorScore': 100,
            'distanceToFood': -1
        }
    

      #   ** ally_defend = 3
        #       my_defend = 1
        #       ally_attack = 4
        #       my_attack = 2

        # if myZone is my_defend:
        #   if enemy in my_defend:
        #       chaseClosestInvader()
        #
        #   if ally recently died:
        #       if invader(s) is present, chaseClosestInvader(my_pos)
        #   else: 
        #       enemy = findClosestEnemyToMiddle(my_pos)
        #       goToEnemy(enemyIndex)
        #
        #   if ally in zone ally_defend or ally_attack:
        #
        #       if enemy in my_attack
        #           enemy = findClosestEnemyToMiddle(my_pos)
        #           goToEnemy(enemyIndex)
        #       
        #       if enemy not in my_attack
        #           findClosestPellet(my_pos) *returns path to closest pellet
        #
        #   if ally in zone my_defend: (in same zone)
        #
        #       if enemy not in my_attack
        #           path to enemy = findClosestEnemyToMiddle(my_pos)
        #           enemy distance = length(path to enemy)
        #           path to pellet = findClosestPellet(my_pos)
        #           pellet distance = length(path to pellet)
        #           if pellet distance < enemy distance:
        #               path to pellet[0]
        #           else
        #               path to enemy[0]
        #
        #       if enemy in my_attack
        #           goToEnemy(enemyIndex)
        #
        #   if ally in zone my_attack:
        #
        #       if enemy in ally_defend
        #           chaseClosestInvader(my_pos)
        #
        #       if enemy not in ally_defend
        #           go to closest enemy to middle (don't cross middle)
        #
        # if myZone is my_attack or ally_attack:
        #   go to nearest pellet and avoid ghosts
        #
        # if myZone is ally_defend:
        #
        #   if ally in zone my_defend or my_attack:
        #       swap all zone values
        #       my_defend <-> ally_defend
        #       my_attack <-> ally_attack
        #
        #       if enemy in my_attack             <------ what to do if ghosts are on defense in different zones (in desired position)
        #           go to enemy in my_attack (don't cross middle)
        #       
        #       if enemy not in my_attack
        #           go to nearest pellet
        #   
        #   if ally in zone ally_defend or ally_attack: (in same zone)
        #       if no invaders
        #           go back to my zone
        #
        #       if invader
        #           chase invader
        #
        #
        # FUNCTIONS:
        #
        #   chaseClosestInvader() *returns first step in path towards the closest invader
        #   findEnemyClosestToMiddle() *returns coordinates of enemy closest to middle line
        #   findMyEnemy(my_attack) *returns the index of the enemy in my_attack zone
        #   goToEnemy(enemyIndex) *returns first step in path towards [x = middle, y = enemy's y level]
        #   
        #   findClosestPellet(my_pos) *returns path to closest pellet
        #   
