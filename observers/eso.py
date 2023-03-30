from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []
        self.x_hat = 0

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        print("-------------------------------------")
        print(self.A)
        print(self.states[-1])
        print((self.A @ self.states[-1].T))
        print(self.B @ [u])
        print(self.L @ [q - self.x_hat])


        z_hat_dot = self.A @ self.state + self.B @ [u] + self.L @ [q - self.x_hat]

        z_hat = self.state + (z_hat_dot * self.Tp)

        print(z_hat)

        self.x_hat = z_hat[0]
        self.state = z_hat

    def get_state(self):
        return self.state
