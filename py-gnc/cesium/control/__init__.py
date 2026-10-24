# def observability(mu, u, ground_stations, dt):
#     A = robot_jac_A(mu, u, dt)
#     C = robot_jac_C(mu, ground_stations)

#     N = A.shape[0]

#     O = C
#     for i in range(1, N):
#         O = np.vstack((O, C @ np.linalg.matrix_power(A, i)))

#     return O

# TODO: put control stuff from 212 and 205
