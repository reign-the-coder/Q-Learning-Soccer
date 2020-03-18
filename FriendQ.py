from Env.soccer import Player, World, construct_env
from Env.testbench import create_state_comb, print_status
import numpy as np
import random
from Plotter import plot


def run():
    q_table = np.zeros((len(states), 5, 5))  # 5 actions per player
    gamma = 0.9
    alpha = 0.9
    q_values_71 = []

    for test in range(1000000):
        player_a, player_b = construct_env(world)
        goal = False
        current_state_value = states[world.map_player_state()]

        while not goal:
            action_a = random.choice(range(5))
            action_b = random.choice(range(5))
            actions = {player_a.player_id: action_a, player_b.player_id: action_b}

            next_state, rewards, goal = world.move(actions)
            next_state_value = states[next_state]

            if goal:
                max_action = 0
            else:
                max_action = np.max(q_table[next_state_value])

            # Update Q Time
            q_table[current_state_value, action_a, action_b] += alpha * (rewards['A'] + gamma * max_action - \
                                                                q_table[current_state_value, action_a, action_b])

            current_state_value = next_state_value

        alpha = alpha * 0.9999 if alpha > 0.001 else 0.001
        q_values_71.append(q_table[71, 1, 4])  # State 71 is the one in the paper and 1 is S and 4 is X

        if test % 1000 == 0:
            print(test)

    return q_values_71


if __name__ == '__main__':
    states = create_state_comb(range(8), range(8))  # 8 cells for two players
    world = World()
    q_values_71 = run()
    q_values_71_array = np.array(q_values_71)
    q_values_71_reduced = q_values_71_array[q_values_71_array != 0]
    np.save("friend-Q", np.array(q_values_71_reduced))
    plot("friend-Q.npy", "Friend-Q")
