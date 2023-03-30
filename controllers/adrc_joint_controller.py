import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]], dtype=np.float32)
        B = np.array([[0],
                      [self.b],
                      [0]], dtype=np.float32)
        L = np.array([[3 * (-p)],
                      [3 * ((-p)** 2)],
                      [(-p) ** 3]], dtype=np.float32)
        W = np.array([1, 0, 0])
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        return NotImplementedError

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC
        q = x[0]

        x_hat, x_hat_dot, f = self.eso.get_state()
        v = q_d_ddot + self.kd * (q_d_dot - x_hat_dot) + self.kp * (q_d - q)

        u = (v - f)/self.b
        self.eso.update(q, u)

        return u
