"""
This seems like a snippet of code from an implementation of expectimax, since we can find the max_value and
exp_value functions. Expectimax alternates between trying to maximize the value during the player's turn and
then using the expected value to predict the opponent's move since it is used against players that play randomly
or just not optimally.
"""


# is_terminal(state) is a function that returns whether or not the specified state is a terminal state,
# or a state that is on the lowest layer of the game-tree/search space. Basically just a final state that
# defines the end of the game

# utility(state) is a function that returns the value of a terminal state. It probably is used to look at how "good"
# this specific terminal state is for the player and how should the next move be optimized.

# value(state) is a function that finds the value of the current state based on which player's turn it is and
# whether the state is a terminal state.
def value(state):
    # if the state is a terminal state
    if is_terminal(state):
        # returns how good the terminal state is
        return utility(state)
    # if the agent is the one trying to maximize
    if agent == MAX_AGENT:
        # returns the maximum value possible in the next moves
        return max_value(state)
    # if the agent is the opponent or the random one (in this case it's still called MIN_AGENT)
    if agent == MIN_AGENT:
        # returns the expected value for the next possible moves
        return exp_value(state)


# get_successors(state) is a function that returns a list or a similar data structure that includes all the
# possible successor states of the specified state.

# max_value(state) is a function that finds the maximum value for the next state by looking at all the
# next possible states. It is used to determine that max value for the player's move since the player would be
# trying to play optimally and maximizing the score
def max_value(state):
    # initializes value as -infinity to be as low as possible to find true maximums
    v = float('-inf')
    # looking at each of the possible next states
    for neighbor in get_successors(state):
        # updates maximum if larger
        v = max(v, value(neighbor))
    # returns maximum value
    return v


# get_prob(neighbor) is a function that returns the probability of the specified neighbor/next state
# being chosen from the current state. This function is used to calculate expected value.

# exp_value(state) is a function that finds the expected value for the next state by looking at all the
# next possible states and their probabilities. It is used to determine the expected value for the opponent's
# move since they are playing randomly and/or not optimally
def exp_value(state):
    # initializes value as 0 since expected value is the sum of the probabilities times the values
    v = 0
    # looking at each of the possible next states
    for neighbor in get_successors(state):
        # adding the probability times the value of the next move to get expected value
        p = get_prob(neighbor)
        v += p * value(neighbor)
    # returns expected value
    return v
