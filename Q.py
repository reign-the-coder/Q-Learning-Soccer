from Env.soccer import Player, World, construct_env
from Env.testbench import create_state_comb, print_status
import numpy as np
import random
from Plotter import plot


def run():
    q_table_a = np.zeros((len(states), 5))  # 5 actions per player
    q_table_b = np.zeros((len(states), 5))  # 5 actions per player
    gamma = 0.9
    alpha = 0.9
    epsilon = 0.99
    q_values_71 = []
    test_step = 0

    for test in range(50000):
        player_a, player_b = construct_env(world)
        goal = False
        current_state_value = states[world.map_player_state()]

        while not goal:
            # if test_step > 25:
            #     break
            # test_step += 1
            explore = np.random.rand()
            if explore < epsilon:
                action_a = random.choice(range(5))
                action_b = random.choice(range(5))
            else:
                action_a = np.argmax(q_table_a[current_state_value])
                action_b = np.argmax(q_table_b[current_state_value])

            actions = {player_a.player_id: action_a, player_b.player_id: action_b}

            next_state, rewards, goal = world.move(actions)
            next_state_value = states[next_state]

            if goal:
                max_a = 0
                max_b = 0
            else:
                max_a = np.argmax(q_table_a[next_state_value])
                max_b = np.argmax(q_table_b[next_state_value])

            # Update Q Time
            q_table_a[current_state_value, action_a] += alpha * (rewards['A'] + gamma * max_a -
                                                                 q_table_a[current_state_value, action_a])
            q_table_b[current_state_value, action_b] += alpha * (rewards['B'] + gamma * max_b -
                                                                 q_table_b[current_state_value, action_b])

            current_state_value = next_state_value

        alpha = alpha * 0.9999 if alpha > 0.001 else 0.001
        epsilon = epsilon * 0.9999 if epsilon > 0.001 else 0.001
        q_values_71.append(q_table_a[71, 1])  # State 71 is the one in the paper and 1 is S

        if test % 1000 == 0:
            print(test)
        # test_step = 0

    return q_values_71


if __name__ == '__main__':
    states = create_state_comb(range(8), range(8))
    world = World()
    q_values_71 = run()
    q_values_71_array = np.array(q_values_71)
    q_values_71_reduced = q_values_71_array[q_values_71_array != 0]
    np.save("Q", np.array(q_values_71_reduced))
    plot("Q.npy", "Q")
