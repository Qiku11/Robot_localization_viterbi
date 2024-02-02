import sys
import numpy as np


def check_observation(observation, position, map):
    correct_block = ''
    # North
    if position[0] == 0:
        correct_block = correct_block + '1'
    else:
        if map[position[0] - 1][position[1]] == 'X':
            correct_block = correct_block + '1'
        else:
            correct_block = correct_block + '0'
    # South
    if position[0] == len(map) - 1:
        correct_block = correct_block + '1'
    else:
        if map[position[0] + 1][position[1]] == 'X':
            correct_block = correct_block + '1'
        else:
            correct_block = correct_block + '0'
    # West
    if position[1] == 0:
        correct_block = correct_block + '1'
    else:
        if map[position[0]][position[1] - 1] == 'X':
            correct_block = correct_block + '1'
        else:
            correct_block = correct_block + '0'
    # East
    if position[1] == len(map[0]) - 1:
        correct_block = correct_block + '1'
    else:
        if map[position[0]][position[1] + 1] == 'X':
            correct_block = correct_block + '1'
        else:
            correct_block = correct_block + '0'

    if len(correct_block) != len(observation):
        print("length is not same")
    d = 0
    for bit1, bit2 in zip(correct_block, observation):
        if bit1 != bit2:
            d += 1

    return d


def viterbi_forward(observation_space, state_space, initial_probs, transition_matrix, emission_matrix):
    trellis_matrix = np.zeros((len(state_space), len(observation_space)))

    for i in range(len(state_space)):
        trellis_matrix[i][0] = initial_probs[0][i] * emission_matrix[i][0]

    for j in range(1, len(observation_space)):
        for i in range(len(state_space)):
            trellis_matrix[i][j] = max(
                [trellis_matrix[k][j - 1] * transition_matrix[k][i] * emission_matrix[i][j] for k in
                 range(len(state_space))])

    return trellis_matrix


if __name__ == "__main__":
    file_path = sys.argv[1]
    f = open(file_path, "r")
    rows, cols = map(int, f.readline().split())

    map = []
    for i in range(rows):
        map_each_line = f.readline().strip().split()
        map.append(map_each_line)

    num_observations = int(f.readline().strip())

    time_steps = []
    for i in range(num_observations):
        steps_each_line = f.readline().strip()
        time_steps.append(steps_each_line)

    error_rate = float(f.readline().strip())

    # print(rows, cols)
    # print(map, len(map[0]))
    # print(num_observations)
    # print(time_steps)
    # print(error_rate)

    state_space = []
    count_zeros = 0
    for i in range(rows):
        for j in range(cols):
            if map[i][j] == "0":
                state_space.append((i, j))
                count_zeros += 1
    initial_probs = np.full((1, len(state_space)), 1 / count_zeros)
    #print(initial_probs)
    # print("state space:", state_space, len(state_space))

    transition_matrix = np.zeros((len(state_space), len(state_space)))
    for i in range(len(state_space)):
        current_state = state_space[i]
        for j in range(len(state_space)):
            next_state = state_space[j]
            if (abs(current_state[0] - next_state[0]) == 1 and current_state[1] == next_state[1]) or (
                    abs(current_state[1] - next_state[1]) == 1 and current_state[0] == next_state[0]):
                neighbors = []
                # North
                if current_state[0] > 0 and map[current_state[0] - 1][current_state[1]] == '0':
                    neighbors.append((current_state[0] - 1, current_state[1]))
                # South
                if current_state[0] < rows - 1 and map[current_state[0] + 1][current_state[1]] == '0':
                    neighbors.append((current_state[0] + 1, current_state[1]))
                # West
                if current_state[1] > 0 and map[current_state[0]][current_state[1] - 1] == '0':
                    neighbors.append((current_state[0], current_state[1] - 1))
                # East
                if current_state[1] < cols - 1 and map[current_state[0]][current_state[1] + 1] == '0':
                    neighbors.append((current_state[0], current_state[1] + 1))
                transition_matrix[i][j] = 1 / len(neighbors)

    # # print out the transition matrix
    # for i in range(len(state_space)):
    #     print(transition_matrix[i])


    emission_matrix = np.zeros((len(state_space), num_observations))
    for i in range(num_observations):
        observation = time_steps[i]
        for j in range(len(state_space)):
            d = check_observation(observation, state_space[j], map)
            emission_matrix[j, i] = ((1 - error_rate) ** (4 - d)) * (error_rate ** d)

    # for i in range(len(state_space)):
    #     print(emission_matrix[i])


    trellis = viterbi_forward(time_steps, state_space, initial_probs, transition_matrix, emission_matrix)

    # print(trellis)
    maps = []
    for i in range(num_observations):
        temp = np.zeros((rows, cols))
        for j in range(len(state_space)):
            position = state_space[j]
            temp[position[0]][position[1]] = trellis[j][i]
        maps.append(temp)
    maps = np.array(maps)
    np.savez("output.npz", *maps)

    #print(maps)
