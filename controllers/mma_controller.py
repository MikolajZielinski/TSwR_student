import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3

        m1, m2, m3 = ManiuplatorModel(Tp), ManiuplatorModel(Tp), ManiuplatorModel(Tp)
        m1.set_m3_r3(0.1, 0.05)
        m2.set_m3_r3(0.01, 0.01)
        m3.set_m3_r3(1.0, 0.3)

        self.models = [m1, m2, m3]
        self.i = 0

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)

        tau_1 = self.models[0].M(x) + self.models[0].C(x)
        tau_2 = self.models[1].M(x) + self.models[1].C(x)
        tau_3 = self.models[2].M(x) + self.models[2].C(x)

        print(x)
        print(tau_1)
        print(tau_2)
        print(tau_3)
        print("--------------------------------------------------------------------------")

        pass

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        v = np.array(q_r_ddot + (1.0 * (q_r_dot - q_dot)) + (10.0 * (q_r - q))) # TODO: add feedback
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
