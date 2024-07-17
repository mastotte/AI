from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}
        # Compute the values here.
        for i in range(self.iters):
            
            newValues = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                else:
                    qValues = []
                    for action in self.mdp.getPossibleActions(state):
                        qValues.append(self.getActionValues(state, action))

                    newValues[state] = max(qValues)
    
            self.values = newValues

    def getActionValues(self, state, action):
        action_val = 0.0

        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            action_val += prob * (reward + self.discountRate * self.getValue(nextState))

        return action_val

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        
        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """
        return self.getPolicy(state)
    
    def getPolicy(self, state):
        """
        Returns the policy at the state (no exploration).
        """
        possibleActions = self.mdp.getPossibleActions(state)

        if not possibleActions:
            return None
        
        bestAction = None
        bestQValue = float('-inf')

        for action in possibleActions:
            qValue = self.getQValue(state, action)
            if qValue > bestQValue:
                bestAction = action
                bestQValue = qValue
        
        return bestAction

    def getQValue(self, state, action):
        """
        The q-value of the state-action pair (after the indicated number of value iteration passes).
        """

        if self.mdp.isTerminal(state):
            return 0.0  # Terminal state has a Q-value of 0.
        
        qValue = 0.0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in transitions:
            reward = self.mdp.getReward(state, action, nextState)
            value = self.getValue(nextState)
            qValue += prob * (reward + self.discountRate * value)
        
        return qValue
