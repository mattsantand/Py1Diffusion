import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, csr_matrix
from tqdm import tqdm


class Solver1D(object):
    def __init__(self):
        self.__diff_coeff = None
        self.__c = None
        self.__tmin = None
        self.__tmax = None
        self.__xmin = None
        self.__xmax = None
        self.__dt = None
        self.__dx = None
        self.__A = None
        self.__B = None
        self.__space_steps = None
        self.__time_steps = None
        self.__r = None

    @property
    def r(self):
        return self.__r

    @property
    def tmin(self):
        return self.__tmin

    @property
    def tmax(self):
        return self.__tmax

    @property
    def xmin(self):
        return self.__xmin

    @property
    def xmax(self):
        return self.__xmax

    @property
    def c(self):
        return self.__c

    @property
    def diff_coeff(self):
        return self.__diff_coeff

    @diff_coeff.setter
    def diff_coeff(self, value):
        self.__diff_coeff = value

    @property
    def dt(self):
        return self.__dt

    @property
    def dx(self):
        return self.__dx

    @property
    def space_steps(self):
        return int(self.__space_steps)

    @property
    def time_steps(self):
        return int(self.__time_steps)

    @property
    def A(self):
        return self.__A

    def initialize_time(self, tmin, tmax, dt):
        self.__tmin = tmin
        self.__tmax = tmax
        self.__dt = dt
        self.__time_steps = (self.tmax - self.tmin) // self.dt

    def initialize_space(self, xmin, xmax, dx):
        self.__xmin = xmin
        self.__xmax = xmax
        self.__dx = dx
        # +2 to have ghost cells for the boundary conditions
        self.__space_steps = int((self.xmax - self.xmin) // self.dx + 2)

    def initialize_c(self):
        self.__c = np.zeros(shape=(self.space_steps, self.time_steps))
        self.__c[0, 0] = 1

    def _create_matrices(self, t_step=0):
        r = self.dt / 2 / self.dx ** 2
        self.__r = r
        self.__A = np.zeros(shape=(self.space_steps, self.space_steps))
        self.__B = np.zeros(shape=(self.space_steps, self.space_steps))

        # these two lines set the BC.
        # TODO: Verify these conditions. At the moment should be Dirichelet
        self.__A[0, 0] = 1
        self.__A[self.space_steps - 1, self.space_steps - 1] = 1

        for i in range(1, self.space_steps - 1):
            d_plus = (self.diff_coeff(self.c[i + 1, t_step]) + self.diff_coeff(self.c[i, t_step])) / 2
            d_minus = (self.diff_coeff(self.c[i, t_step]) + self.diff_coeff(self.c[i - 1, t_step])) / 2

            self.__A[i, i - 1] = -r * d_minus
            self.__A[i, i] = 1 + r * (d_plus + d_minus)
            self.__A[i, i + 1] = -r * d_plus

        self.__B[0, 0] = 1
        self.__B[self.space_steps - 1, self.space_steps - 1] = 1

        for i in range(1, self.space_steps - 1):
            d_plus = (self.diff_coeff(self.c[i + 1, t_step]) + self.diff_coeff(self.c[i, t_step])) / 2
            d_minus = (self.diff_coeff(self.c[i, t_step]) + self.diff_coeff(self.c[i - 1, t_step])) / 2

            self.__B[i, i - 1] = r * d_minus
            self.__B[i, i] = 1 - r * (d_plus + d_minus)
            self.__B[i, i + 1] = r * d_plus
        return self.__A, self.__B

    def solve_diffusion(self):
        for t in tqdm(range(1, self.time_steps), ncols=100):
            A, B = self._create_matrices()
            b = np.dot(B, self.c[:, t - 1])
            c = np.linalg.solve(A, b)
            self.__c[:, t] = c
        return self.c


if __name__ == '__main__':
    sim = Solver1D()
    sim.initialize_space(0, 10, 0.01)
    sim.initialize_time(0, 1, 0.002)
    sim.initialize_c()
    sim.diff_coeff = lambda x: 0.1 * np.exp(-0.1 * x)
    A, B = sim._create_matrices()
    print(sim.r)
    c = sim.solve_diffusion()
    plt.plot(c[1:, ::10])
    plt.show()
