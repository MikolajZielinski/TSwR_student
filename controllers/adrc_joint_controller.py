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
        self.b = b

        B = np.array([[0],
                      [b],
                      [0]], dtype=np.float32)
        
        self.eso.set_B(B)

    def calculate_control(self, i, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC
        q = x[0]

        x_hat, x_hat_dot, f = self.eso.get_state()
        v = q_d_ddot + self.kd * (q_d_dot - x_hat_dot) + self.kp * (q_d - q)

        u = (v - f)/self.b
        self.eso.update(q, u)

        l1 = 0.5
        r1 = 0.04
        m1 = 3.
        l2 = 0.4
        r2 = 0.04
        m2 = 2.4
        I_1 = 1 / 12 * m1 * (3 * r1 ** 2 + l1 ** 2)
        I_2 = 1 / 12 * m2 * (3 * r2 ** 2 + l2 ** 2)
        m3 = 0.5
        r3 = 0.05
        I_3 = 2. / 5 * m3 * r3 ** 2

        alfa = (m1 * ((l1 / 2) ** 2)) + I_1 + (m2 * ((l1 ** 2) + ((l2 / 2) ** 2))) + I_2 + (m3 * ((l1 ** 2) + (l2 ** 2))) + I_3
        beta = (m2 * l1 * (l2 / 2)) + (m3 * l1 * l2)
        gamma = (m2 * ((l2 / 2) ** 2)) + I_2 + (m3 * (l2 ** 2)) + I_3

        M = np.array([[alfa + (2 * beta * np.cos(x_hat)), gamma + (beta * np.cos(x_hat))],
                      [gamma + (beta * np.cos(x_hat)), gamma]])
    
        M_inv = np.linalg.inv(M)

        self.set_b(M_inv[i, i])

        return u
