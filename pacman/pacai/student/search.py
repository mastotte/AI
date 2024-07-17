"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def ENQUEUE(frontier, node):
    frontier.push(node)

def ENQUEUE_PRIORITY(frontier, node, p):
    frontier.push(node, p)

def GENERALSEARCH(problem, function, strategy, isRed, width):
    # returns a solution, or failure
    # initialize the search tree using the initial state of problemo
    if strategy == "DFS":
        L = Stack()
    elif strategy == "BFS":
        L = Queue()
    elif strategy == "ASTAR" or strategy == "UCS":
        L = PriorityQueue()
    else:
        print("Invalid Search Strategy: ", strategy)
        return
    
    visited = {problem.startingState()}
    parents = {problem.startingState(): None}
    p = []
    if (strategy == "ASTAR" or strategy == "UCS"):
        ENQUEUE_PRIORITY(L, (problem.startingState(), p), 0)
    else:
        ENQUEUE(L, (problem.startingState(), p))

    while (L.isEmpty() is False):
        top = L.pop()
        cur, path = top

        if problem.isGoal(cur):
            break

        for adj in problem.successorStates(cur):
            # for final project 
            if (isRed):
                x, y = adj[0]
                if x > width/2:
                    continue
            else:
                x, y = adj[0]
                if x <= width/2:
                    continue

            if adj[0] not in visited:
                if (strategy == "UCS"):
                    ENQUEUE_PRIORITY(L, (adj[0], path + [adj[1]]), adj[2])
                elif (strategy == "ASTAR"):
                    ENQUEUE_PRIORITY(L, (adj[0], path + [adj[1]]), function(adj[0]))
                else:
                    ENQUEUE(L, (adj[0], path + [adj[1]]))
                visited.add(adj[0])
                parents[adj[0]] = cur

    return path

def BESTFIRSTSEARCH(problem, EVALFN):
    def queueing_fn(node):
        return EVALFN(node)

    return GENERALSEARCH(problem, queueing_fn, "ASTAR")

def depthFirstSearch(problem):
    # Search the deepest nodes in the search tree first [p 85].
    return GENERALSEARCH(problem, ENQUEUE, "DFS")
    

def breadthFirstSearch(problem, isRed, width):
    # Search the shallowest nodes in the search tree first. [p 81]
    return GENERALSEARCH(problem, ENQUEUE, "BFS", isRed, width)

def uniformCostSearch(problem):
    # Search the node of least total cost first.
    return GENERALSEARCH(problem, ENQUEUE_PRIORITY, "UCS")

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    # *** Variables ***
    P = PriorityQueue()
    visited = {problem.startingState()}
    # *****************
    P.push((problem.startingState(), []), 1)
    while (P.isEmpty() is False):
        top = P.pop()
        cur, path = top
        if problem.isGoal(cur):
            # print("Goal Found: ",cur)
            break
        for adj in problem.successorStates(cur):
            if adj[0] not in visited:
                P.push((adj[0], path + [adj[1]]), heuristic(adj[0], problem) + len(path))
                visited.add(adj[0])

    return path