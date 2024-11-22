import sys

import numpy as np
import matplotlib.pyplot as plt
from math import cos

N = [20, 100, 200, 1000]


class DifferentialEquationSolver:
    def __init__(self, f, y0, x0, end_border_x, is_reverse, h):
        self.f = f
        self.y0 = y0
        self.x0 = x0
        self.is_reverse = is_reverse
        if is_reverse:
            self.h = [-i for i in h]
        else:
            self.h = h
        self.right_border_x = end_border_x
        self.relative_errors = []
        self.absolute_errors_2nd = []
        self.absolute_errors_4th = []

        self.results_2nd = []
        self.results_4th = []

    def solve(self):
        self.runge_kutta_2nd_ord()
        self.runge_kutta_4th_ord()
        self.compute_errors()

    def runge_kutta_2nd_ord(self):
        self.results_2nd.clear()

        # applying runge-kutta n times
        for h in self.h:
            x0, y = self.x0, self.y0
            if self.is_reverse:
                n = int((x0 - self.right_border_x) / abs(h))
                # print(n)
            else:
                n = int((self.right_border_x - x0) / h)
                # print(n)
            # print("RUNGE_KUTTA_2ND_ORD")
            # print(f"N: {n}")
            interim_results = [(x0, y)]
            for _ in range(n):
                k1 = self.f(x0, y)
                k2 = self.f(x0 + 0.5 * h, y + 0.5 * k1 * h)

                y += (h * k2)
                x0 += h

                interim_results.append((x0, y))
            self.results_2nd.append(interim_results)
            # print(f"Result = {x0, y}")

    def runge_kutta_4th_ord(self):
        self.results_4th.clear()

        for h in self.h:
            x0, y = self.x0, self.y0
            # applying runge-kutta n times
            if self.is_reverse:
                n = int((x0 - self.right_border_x) / abs(h))
                # print(n)
            else:
                n = int((self.right_border_x - x0) / h)
                # print(n)
            # print("RUNGE_KUTTA_4TH_ORD")
            # print(f"N: {n}")
            interim_results = [(x0, y)]
            for _ in range(n):
                k1 = self.f(x0, y)
                k2 = self.f(x0 + (0.5 * h), y + (h * (0.5 * k1)))
                k3 = self.f(x0 + (0.5 * h), y + (h * (0.5 * k2)))
                k4 = self.f(x0 + h, y + (h * k3))

                y += ((h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
                x0 += h

                interim_results.append((x0, y))
                # print(f"y({x0}) = {y}")

            self.results_4th.append(interim_results)
            # print(f"Result = {x0, y}")

    def compute_errors(self):
        # Clear previous results
        self.relative_errors.clear()
        self.absolute_errors_2nd.clear()
        self.absolute_errors_4th.clear()

        steps = list(map(lambda x: x // 10,
                         map(lambda x: len(x), self.results_2nd)))

        for k, j in zip(range(len(steps) - 1),
                        range(len(self.results_2nd) - 1)):
            relative_errors = []

            matrix1 = [i[1] for i in
                       self.results_2nd[j][::steps[k]]]
            matrix2 = [i[1] for i in
                       self.results_2nd[j + 1][::steps[k + 1]]]

            for i in range(len(matrix1)):
                relative_error = abs(matrix1[i] - matrix2[i])
                relative_errors.append(relative_error)

            print(f"N = {N[k]}\n",
                  ", ".join(list(map(str, relative_errors[1:]))))

            self.relative_errors.append(relative_errors)
            self.absolute_errors_2nd.append(max(relative_errors))

        steps = list(map(lambda x: x // 10,
                         map(lambda x: len(x), self.results_4th)))

        for k, j in zip(range(len(steps) - 1),
                        range(len(self.results_4th) - 1)):
            relative_errors = []

            matrix1 = [i[1] for i in
                       self.results_4th[j][::steps[k]]]
            matrix2 = [i[1] for i in
                       self.results_4th[j + 1][::steps[k + 1]]]

            for i in range(len(matrix1)):
                relative_error = abs(matrix1[i] - matrix2[i])
                relative_errors.append(relative_error)

            print(f"N = {N[k]}\n",
                  ", ".join(list(map(str, relative_errors[1:]))))

            self.relative_errors.append(relative_errors)
            self.absolute_errors_4th.append(max(relative_errors))


def plot(expression1, expression2):
    plt.plot(N, np.log2(expression1.absolute_errors_2nd),
             label="Runge-Kutta 2nd Order A")
    plt.plot(N, np.log2(expression1.absolute_errors_4th),
             label="Runge-Kutta 4th Order A")

    plt.plot(N, np.log2(expression2.absolute_errors_2nd),
             label="Runge-Kutta 2nd Order B")
    plt.plot(N, np.log2(expression2.absolute_errors_4th),
             label="Runge-Kutta 4th Order B")
    plt.plot(0, 0, label="DAKO14I02", color="black", linewidth=3)

    plt.xlabel("N")
    plt.ylabel("Log2 of Absolute Errors")
    plt.title("Absolute Errors for Runge-Kutta Methods")
    plt.legend()
    plt.show()


def main():
    h = [0.1, 0.05, 0.01, 0.005, 0.001]  # Grid Steps
    example_a = [lambda x, y: x + cos(y), 30, 1, 2, False]
    a_solver = DifferentialEquationSolver(*example_a, h)
    a_solver.solve()
    example_b = [lambda x, y: x ** 2 + y ** 2, 1, 2, 1, True]
    b_solver = DifferentialEquationSolver(*example_b, h)
    b_solver.solve()
    plot(a_solver, b_solver)


if __name__ == '__main__':
    sys.stdout = open("result.txt", "w")
    main()
