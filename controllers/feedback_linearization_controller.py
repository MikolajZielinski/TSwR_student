import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x
        v = np.array(q_r_ddot + (0.0 * ([q1_dot, q2_dot] - q_r_dot)) + (0.0 * ([q1, q2] - q_r))).T

        tau = self.model.M(x) @ v + self.model.C(x) @ np.array([q1_dot, q2_dot]).T

        # print("------------------------------------------------------------------------------------------------------")
        # print(x)
        # print(q_r, q_r_dot, q_r_ddot)
        # print([q1_dot, q2_dot], q_r_dot, [q1_dot, q2_dot] - q_r_dot)
        # print(self.model.M(x), q_r_ddot, self.model.M(x) @ q_r_ddot)
        # print(self.model.C(x), np.array([q1_dot, q2_dot]).T, self.model.C(x) @ np.array([q1_dot, q2_dot]).T)
        # print("------------------------------------------------------------------------------------------------------")
        
        return tau
