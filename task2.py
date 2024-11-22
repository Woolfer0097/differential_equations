import sys

import numpy as np
import matplotlib.pyplot as plt

N = [20, 100, 200, 1000]


class SecondOrderODESolver:
    def __init__(self, f, g, y0, dy0, x0, end_x, h):
        self.f = f  # Function for y1'
        self.g = g  # Function for y2'
        self.y0 = y0
        self.dy0 = dy0
        self.x0 = x0
        self.end_x = end_x
        self.h = h
        self.relative_errors = []
        self.absolute_errors_2nd = []
        self.absolute_errors_4th = []

        self.results_2nd = []
        self.results_4th = []

    def solve(self):
        self.runge_kutta_2nd_order()
        self.runge_kutta_4th_order()
        self.compute_errors()

    def runge_kutta_2nd_order(self):
        self.results_2nd.clear()

        for h in self.h:
            x, y1, y2 = self.x0, self.y0, self.dy0
            n = int((self.end_x - self.x0) / h)
            interim_results = [(x, y1, y2)]

            for _ in range(n):
                k1_y1 = self.f(x, y1, y2)
                k1_y2 = self.g(x, y1, y2)

                k2_y1 = self.f(x + 0.5 * h, y1 + 0.5 * h * k1_y1,
                               y2 + 0.5 * h * k1_y2)
                k2_y2 = self.g(x + 0.5 * h, y1 + 0.5 * h * k1_y1,
                               y2 + 0.5 * h * k1_y2)

                y1 += h * k2_y1
                y2 += h * k2_y2
                x += h

                interim_results.append((x, y1, y2))
            self.results_2nd.append(interim_results)

    def runge_kutta_4th_order(self):
        self.results_4th.clear()

        for h in self.h:
            x, y1, y2 = self.x0, self.y0, self.dy0
            n = int((self.end_x - self.x0) / h)
            interim_results = [(x, y1, y2)]

            for _ in range(n):
                k1_y1 = self.f(x, y1, y2)
                k1_y2 = self.g(x, y1, y2)

                k2_y1 = self.f(x + 0.5 * h, y1 + 0.5 * h * k1_y1,
                               y2 + 0.5 * h * k1_y2)
                k2_y2 = self.g(x + 0.5 * h, y1 + 0.5 * h * k1_y1,
                               y2 + 0.5 * h * k1_y2)

                k3_y1 = self.f(x + 0.5 * h, y1 + 0.5 * h * k2_y1,
                               y2 + 0.5 * h * k2_y2)
                k3_y2 = self.g(x + 0.5 * h, y1 + 0.5 * h * k2_y1,
                               y2 + 0.5 * h * k2_y2)

                k4_y1 = self.f(x + h, y1 + h * k3_y1, y2 + h * k3_y2)
                k4_y2 = self.g(x + h, y1 + h * k3_y1, y2 + h * k3_y2)

                y1 += h / 6 * (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1)
                y2 += h / 6 * (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2)
                x += h

                interim_results.append((x, y1, y2))
            self.results_4th.append(interim_results)

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


def plot(expression1):
    plt.plot(N, np.log2(expression1.absolute_errors_2nd),
             label="Runge-Kutta 2nd Order A")
    plt.plot(N, np.log2(expression1.absolute_errors_4th),
             label="Runge-Kutta 4th Order A")
    plt.plot(0, 0, label="DAKO14I02", color="black", linewidth=3)
    plt.xlabel("N")
    plt.ylabel("Log2 of Absolute Errors")
    plt.title("Absolute Errors for Runge-Kutta Methods")
    plt.legend()
    plt.show()


def main():
    h = [0.1, 0.05, 0.01, 0.005, 0.001]  # Grid Steps

    # Define the system of first-order ODEs
    f = lambda x, y1, y2: y2
    g = lambda x, y1, y2: y1 * np.sin(x)

    y0 = 0  # Initial condition y(0)
    dy0 = 1  # Initial condition y'(0)
    x0 = 0  # Start of the interval
    end_x = 1  # End of the interval

    solver = SecondOrderODESolver(f, g, y0, dy0, x0, end_x, h)
    solver.solve()

    plot(solver)


if __name__ == "__main__":
    sys.stdout = open("result.txt", "w")
    main()
