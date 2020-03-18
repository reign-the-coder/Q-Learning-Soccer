from Env.soccer import Player, World, construct_env
from Env.testbench import create_state_comb, print_status
import numpy as np
import random
from Plotter import plot
from cvxopt.solvers import options
from cvxopt import matrix, solvers

def run():
    options['show_progress'] = False

    q_table_a = np.ones((len(states), 5, 5))
    q_table_b = np.ones((len(states), 5, 5))
    # q_table_a = np.random.rand(len(states), 5, 5)  # 5 actions per player
    gamma = 0.9
    alpha = 0.8
    q_values_71 = []
    update_q = True

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

            # max_a = 0
            # max_b = 0

            # if not goal:
            # used for rationality constraints
            q_table_a_at_state = q_table_a[next_state_value]
            q_table_b_at_state = q_table_b[next_state_value]

            # probabilities must be >=0 25 variables for actions x actions
            probability_constraints = np.eye(25)
            # constraints for picking one action over another in q_table_a
            q_a_constraints = []
            # constraints for picking one action over another in q_table_b
            q_b_constraints = []

            for i in range(5):
                q_temp = []
                for j in range(5):
                    if i != j:
                        # constraints of picking one action over another
                        q_temp.append(q_table_a_at_state[i, :] - q_table_a_at_state[j, :])
                q_a_constraints.append(q_temp)

            row1 = np.hstack((np.vstack(q_a_constraints[0]), np.zeros((4, 20))))
            row2 = np.hstack((np.zeros((4, 5)), np.vstack(q_a_constraints[1]), np.zeros((4, 15))))
            row3 = np.hstack((np.zeros((4, 10)), np.vstack(q_a_constraints[2]), np.zeros((4, 10))))
            row4 = np.hstack((np.zeros((4, 15)), np.vstack(q_a_constraints[3]), np.zeros((4, 5))))
            row5 = np.hstack((np.zeros((4, 20)), np.vstack(q_a_constraints[4])))
            player_a_constraint_matrix = np.vstack((row1, row2, row3, row4, row5))

            for j in range(5):
                q_temp = []
                for i in range(5):
                    if i != j:
                        temp = q_table_b_at_state[:, j] - q_table_b_at_state[:, i]
                        q_temp.append(temp.T)
                q_b_constraints.append(q_temp)

            row1 = np.hstack((np.vstack(q_b_constraints[0]), np.zeros((4, 20))))
            row2 = np.hstack((np.zeros((4, 5)), np.vstack(q_b_constraints[1]), np.zeros((4, 15))))
            row3 = np.hstack((np.zeros((4, 10)), np.vstack(q_b_constraints[2]), np.zeros((4, 10))))
            row4 = np.hstack((np.zeros((4, 15)), np.vstack(q_b_constraints[3]), np.zeros((4, 5))))
            row5 = np.hstack((np.zeros((4, 20)), np.vstack(q_b_constraints[4])))
            player_b_constraint_matrix = np.vstack((row1, row2, row3, row4, row5))

            player_constraints = np.vstack((probability_constraints, player_a_constraint_matrix, player_b_constraint_matrix))

            # Gx = h
            G = matrix(player_constraints * -1)
            h = matrix(np.zeros(player_constraints.shape[0]))

            # C = np.zeros(25)
            # minimize Cx
            C = np.add(q_table_a_at_state, q_table_b_at_state).reshape(1, 25)
            C *= -1
            C = matrix(C[0])

            # Ax = b
            A = matrix(np.ones((1, 25)))
            b = matrix([1.])

            result = solvers.lp(C, G, h, A, b)['x']
            if result is not None:
                result_array = np.array(result)
                q_a_table_flatten = q_table_a_at_state.flatten().reshape(25, 1)
                q_b_table_flatten = q_table_b_at_state.flatten().reshape(25, 1)

                # Utilitarian CE maximizes sum of player's rewards
                max_a = 0 if goal else sum(result_array * q_b_table_flatten)
                max_b = 0 if goal else sum(result_array * q_a_table_flatten)

            # else:  # Not sure why this happens sometimes
            #     update_q = False

            # if update_q:
                # Update Q
                q_table_a[current_state_value, action_a, action_b] += alpha * (rewards['A'] + gamma * max_a - q_table_a[
                    current_state_value, action_a, action_b])
                q_table_b[current_state_value, action_a, action_b] += alpha * (rewards['B'] + gamma * max_b - q_table_b[
                    current_state_value, action_a, action_b])
                # update_q = True
                current_state_value = next_state_value

        alpha = alpha * 0.999985 if alpha > 0.001 else 0.001
        q_values_71.append(q_table_a[71, 1, 4])  # State 71 is the one in the paper and 1 is S and 4 is X

        if test % 1000 == 0:
            print(test)

    return q_values_71


if __name__ == '__main__':
    states = create_state_comb(range(8), range(8))
    world = World()
    q_values_71 = run()
    q_values_71_array = np.array(q_values_71)
    q_values_71_reduced = q_values_71_array[q_values_71_array != 0]
    np.save("CE-Q", np.array(q_values_71_reduced))
    plot("CE-Q.npy", "CE-Q")
