import numpy as np

# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
# from models.ideal_model import IdealModel
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        M_inv = np.linalg.inv(self.model.M(q0))
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array([3 * (-p),
                           3 * (-p),
                           3 * ((-p)** 2),
                           3 * ((-p)** 2),
                           (-p) ** 3,
                           (-p) ** 3], dtype=np.float32)
        W = np.array([1, 1, 0, 0, 0, 0])
        A = np.zeros((6, 6))
        A[0:4, 2:] = np.eye(4)
        A[2:4, 2:4] = M_inv @ self.model.C(q0)
        B = np.array([np.zeros((2, 2)),
                      M_inv,
                      np.zeros((2, 2))], dtype=np.float32).reshape((6, 2))
        
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        qq = np.concatenate([q, q_dot], axis=0)
        
        M_inv = np.linalg.inv(self.model.M(qq))

        A = np.zeros((6, 6))
        A[0:4, 2:] = np.eye(4)
        A[2:4, 2:4] = -(M_inv @ self.model.C(qq))

        B = np.array([np.zeros((2, 2)),
                      M_inv,
                      np.zeros((2, 2))], dtype=np.float32).reshape((6, 2))

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        q = x[0:2]

        z_hat = self.eso.get_state()
        x_hat, x_hat_dot, f = z_hat[0:2], z_hat[2:4], z_hat[4:6]
        v = np.array([q_d_ddot]).T + self.Kd @ np.array([q_d_dot - x_hat_dot]).T + self.Kp @ np.array([q_d - q]).T

        # print("---------------------------------------------------------")
        
        u = self.model.M(z_hat[0:4]) @ (v - np.array([z_hat[4:]]).T) + np.array([self.model.C(z_hat[0:4]) @ z_hat[2:4]]).T

        self.update_params(x_hat, x_hat_dot)
        self.eso.update(q, u)

        # print(x)
        # print(x_hat, x_hat_dot)


        return u
