import random

import numpy as np
import numpy.typing as npt

from hw2.utils import utility, successors, Node, Tree, GameStrategy

"""
Alpha Beta Search
"""


def max_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the max value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """
    # TODO:
    # return 0, state
    move = None
    if utility(state, k) is not None:
        return utility(state, k), None  # value is utility, move is null
    v = np.NINF
    for a in successors(state, player='X'):
        v2, a2 = min_value(a, alpha=alpha, beta=beta, k=k)
        if v2 > v:
            v, move = v2, a
            alpha = max(alpha, v)
            # print("alpha ", alpha)
        if v >= beta:
            return v, move
    return v, move


def min_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the min value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """

    # # TODO:
    # return 0, state
    move = None
    if utility(state, k) is not None:
        return utility(state, k), None  # value is utility, move is null
    v = np.Inf
    for a in successors(state, player='O'):
        v2, a2 = max_value(a, alpha=alpha, beta=beta, k=k)
        if v2 < v:
            v, move = v2, a
            beta = min(beta, v)
            # print("beta ", beta)
        if v <= alpha:
            return v, move
    return v, move


"""
Monte Carlo Tree Search
"""


def select(tree: "Tree", state: npt.ArrayLike, k: int, alpha: float):
    """Starting from state, find a terminal node or node with unexpanded
    children. If all children of a node are in tree, move to the one with the
    highest UCT value.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
        alpha (float): exploration parameter
    Returns:
        np.ndarray: the game board state
    """

    # TODO:
    # if this node is a terminal node
    if utility(state, k) is not None:
        return state

    child_states = successors(state, player=tree.get(state).player)
    for s in child_states:
        if tree.get(s) is None:  # if the node is not in the tree
            return state  # the state is not in the tree


    # all states are in the tree
    max_uct = np.NINF
    max_uct_state = None
    for s in child_states:
        child_node = tree.get(s)
        if (child_node.parent.state == state).all(): # compare the result
            uct = child_node.w / child_node.N + \
                  alpha * np.sqrt(np.log(child_node.parent.N) / child_node.N)
            if uct > max_uct:
                max_uct = uct
                max_uct_state = s

    if max_uct_state is None:
        return state

    selected_state = select(tree, max_uct_state, k, alpha)
    return selected_state
    # find max uct


def expand(tree: "Tree", state: npt.ArrayLike, k: int):
    """Add a child node of state into the tree if it's not terminal and return
    tree and new state, or return current tree and state if it is terminal.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
    Returns:
        tuple[utils.Tree, np.ndarray]: the tree and the game state
    """

    if utility(state, k) is not None:
        return tree, state

    # TODO:
    # print("expand:")
    parent_player = tree.get(state).player
    child_player = "O" if parent_player == "X" else "X"
    child_states = successors(state, parent_player)
    for child_state in child_states:
        if tree.get(child_state) is None:
                tree.add(Node(child_state, tree.get(state), child_player, 0, 0))
                return tree, child_state

    return tree, state





def simulate(state: npt.ArrayLike, player: str, k: int):
    """Run one game rollout from state to a terminal state using random
    playout policy and return the numerical utility of the result.

    Args:
        state (np.ndarray): the game board state
        player (string): the player, `O` or `X`
        k (int): the number of consecutive marks
    Returns:
        float: the utility
    """

    # TODO:
    current = player
    # current = "O" if current == "X" else "X"
    while utility(state, k) is None:
        # print(state)
        state = random.choice(successors(state, current))
        current = "O" if current == "X" else "X"
    return utility(state, k)


def backprop(tree: "Tree", state: npt.ArrayLike, result: float):
    """Backpropagate result from state up to the root.
    All nodes on path have N, number of plays, incremented by 1.
    If result is a win for a node's parent player, w is incremented by 1.
    If result is a draw, w is incremented by 0.5 for all nodes.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        result (float): the result / utility value

    Returns:
        utils.Tree: the game tree
    """
    current_node = tree.get(state)
    while current_node.parent is not None:
        # print(current_node.player)
        current_node.N += 1
        if result == -1: # if winning, we add 1
            current_node.w += 1
        if result == 0:
            current_node.w += 0.5
        result *= -1
        current_node = current_node.parent

    # the root node
    current_node.N += 1
    if result == -1:
        current_node.w += 1
    if result == 0:
        current_node.w += 0.5

    return tree

    # current_node = tree.get(state)
    # while current_node.parent is not None:
    #     current_node.parent.n += 1
    #     if result == -1:
    #         current_node.n+=1

    # while current_node.parent is not None:
    #     if result == -1:
    #         current_node.parent.w += 1

    # if tree.get(state).parent is None:
    #     return tree
    #
    # tree.get(state).n += 1
    # if result == 1:
    #     tree.get(state).w += 1
    # if result ==0:
    #     tree.get(state).w += 0.5
    #
    # result *= -1

    # pass
    # this is the result for the new

    # if tree.get(state).parent is None:
    #     return tree
    #
    # if result == -1:
    #    tree.get(state).w += 1
    # if result == 1:
    #     pass
    #     # tree.get(state).w += 1
    # if result == 0:
    #     tree.get(state).w += 0.5
    #
    # result *= -1
    #
    # backprop(tree, tree.get(state).parent.state, result=result)


# ******************************************************************************
# ****************************** ASSIGNMENT ENDS *******************************
# ******************************************************************************


def MCTS(state: npt.ArrayLike, player: str, k: int, rollouts: int, alpha: float):
    # MCTS main loop: Execute MCTS steps rollouts number of times
    # Then return successor with highest number of rollouts
    tree = Tree(Node(state, None, player, 0, 1))

    for i in range(rollouts):
        # print(i)
        leaf = select(tree, state, k, alpha)
        # print("leaf ", leaf)# the new expanded node
        tree, new = expand(tree, leaf, k) # added to the tree
        result = simulate(new, tree.get(new).player, k)
        tree = backprop(tree, new, result)

    nxt = None
    plays = 0

    for s in successors(state, tree.get(state).player):
        if tree.get(s).N > plays:
            plays = tree.get(s).N
            nxt = s

    return nxt


def ABS(state: npt.ArrayLike, player: str, k: int):
    # ABS main loop: Execute alpha-beta search
    # X is maximizing player, O is minimizing player
    # Then return best move for the given player
    if player == "X":
        value, move = max_value(state, -float("inf"), float("inf"), k)
    else:
        value, move = min_value(state, -float("inf"), float("inf"), k)

    return value, move


def game_loop(
        state: npt.ArrayLike,
        player: str,
        k: int,
        Xstrat: GameStrategy = GameStrategy.RANDOM,
        Ostrat: GameStrategy = GameStrategy.RANDOM,
        rollouts: int = 0,
        mcts_alpha: float = 0.01,
        print_result: bool = False,
):
    # Plays the game from state to terminal
    # If random_opponent, opponent of player plays randomly, else same strategy as player
    # rollouts and alpha for MCTS; if rollouts is 0, ABS is invoked instead
    current = player
    while utility(state, k) is None:
        if current == "X":
            strategy = Xstrat
        else:
            strategy = Ostrat

        if strategy == GameStrategy.RANDOM:
            state = random.choice(successors(state, current))
        elif strategy == GameStrategy.ABS:
            _, state = ABS(state, current, k)
        else:
            state = MCTS(state, current, k, rollouts, mcts_alpha)

        current = "O" if current == "X" else "X"

        if print_result:
            print(state)

    return utility(state, k)
