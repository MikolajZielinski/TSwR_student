import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3

        self.Tp = Tp
        self.prev_u = [0, 0]
        self.prev_x = [0, 0, 0, 0]

        m1, m2, m3 = ManiuplatorModel(Tp), ManiuplatorModel(Tp), ManiuplatorModel(Tp)
        m1.set_m3_r3(0.1, 0.05)
        m2.set_m3_r3(0.01, 0.01)
        m3.set_m3_r3(1.0, 0.3)

        self.models = [m1, m2, m3]
        self.i = 0

    def choose_model(self, x, prev_u, prev_x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)

        M1_inv = np.linalg.inv(self.models[0].M(prev_x))
        M2_inv = np.linalg.inv(self.models[1].M(prev_x))
        M3_inv = np.linalg.inv(self.models[2].M(prev_x))

        A1 = np.zeros((4, 4))
        A1[:2, 2:] = np.eye(2)
        A1[2:, 2:] = -M1_inv @ self.models[0].C(prev_x)
        A2 = np.zeros((4, 4))
        A2[:2, 2:] = np.eye(2)
        A2[2:, 2:] = -M2_inv @ self.models[1].C(prev_x)
        A3 = np.zeros((4, 4))
        A3[:2, 2:] = np.eye(2)
        A3[2:, 2:] = -M3_inv @ self.models[2].C(prev_x)

        B1 = np.zeros((4,2))
        B1[2:, :] = M1_inv
        B2 = np.zeros((4,2))
        B2[2:, :] = M2_inv
        B3 = np.zeros((4,2))
        B3[2:, :] = M3_inv

        x1_dot = (A1 @ prev_x) + (B1 @ prev_u).T[0]
        x2_dot = (A2 @ prev_x) + (B2 @ prev_u).T[0]
        x3_dot = (A3 @ prev_x) + (B3 @ prev_u).T[0]

        x1 = x1_dot * self.Tp
        x2 = x2_dot * self.Tp
        x3 = x3_dot * self.Tp

        delta = x - prev_x

        e1 = np.abs(x1 - delta).mean()
        e2 = np.abs(x2 - delta).mean()
        e3 = np.abs(x3 - delta).mean()

        self.i = np.argmin([e1, e2, e3])

        print(self.i)
        print(x - prev_x)
        print(x1)
        print(x2)
        print(x3)
        print("--------------------------------------------------------------------------")

        pass

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x, self.prev_u, self.prev_x)
        q = x[:2]
        q_dot = x[2:]
        v = np.array(q_r_ddot + (1.0 * (q_r_dot - q_dot)) + (10.0 * (q_r - q))) # TODO: add feedback
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.prev_u = u
        self.prev_x = x

        return u
