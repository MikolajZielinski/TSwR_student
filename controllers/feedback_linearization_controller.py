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
        v = np.array(q_r_ddot + (1.0 * (q_r_dot - [q1_dot, q2_dot])) + (10.0 * (q_r - [q1, q2])))

        tau = self.model.M(x) @ v + self.model.C(x) @ np.array([q1_dot, q2_dot]).T

        # print("------------------------------------------------------------------------------------------------------")
        # print(q_r_ddot)
        # print([q1_dot, q2_dot], q_r_dot, ([q1_dot, q2_dot] - q_r_dot))
        # print([q1, q2], q_r, ([q1, q2] - q_r))
        # print((5.0 * ([q1, q2] - q_r)))
        # print("------------------------------------------------------------------------------------------------------")
        
        return tau
