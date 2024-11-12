import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Define differential equations and their string representations
differential_eqs = {
    "Lorenz System": (
        "dx/dt = σ(y - x), dy/dt = x(ρ - z) - y, dz/dt = xy - βz",
        lambda t, z, sigma, rho, beta: np.array([
            sigma * (z[1] - z[0]),
            z[0] * (rho - z[2]) - z[1],
            z[0] * z[1] - beta * z[2]
        ])
    ),
    "Rossler System": (
        "dx/dt = -y - z, dy/dt = x + ay, dz/dt = b + z(x - c)",
        lambda t, z, a, b, c: np.array([
            -z[1] - z[2],
            z[0] + a * z[1],
            b + z[2] * (z[0] - c)
        ])
    ),
    "Chua's Circuit": (
        "dx/dt = α(y - x - h(x)), dy/dt = x - y + z, dz/dt = -βy",
        lambda t, z, alpha, beta, h: np.array([
            alpha * (z[1] - z[0] - h(z[0])),
            z[0] - z[1] + z[2],
            -beta * z[1]
        ])
    ),
    "Lorenz '96 Model": (
        "dx_i/dt = (x_{i+1} - x_{i-2})x_{i-1} - x_i + F",
        lambda t, x, F: np.array([
            (x[(i + 1) % len(x)] - x[i - 2]) * x[i - 1] - x[i] + F for i in range(len(x))
        ])
    ),
}


# Implement the RK4 method
def rk4_method(f, y0, t, *params):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        k1 = dt * f(t[i - 1], y[i - 1], *params)
        k2 = dt * f(t[i - 1] + dt / 2, y[i - 1] + k1 / 2, *params)
        k3 = dt * f(t[i - 1] + dt / 2, y[i - 1] + k2 / 2, *params)
        k4 = dt * f(t[i - 1] + dt, y[i - 1] + k3, *params)
        y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


# Define the main function to solve the ODE and plot the results
def solve_ode_3d(eq_name, y0, t_span, num_points, f, eq_str, *params):
    t = np.linspace(t_span[0], t_span[1], num_points)

    start_time = time.time()

    y = rk4_method(f, y0, t, *params)

    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Plot the results in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(y[:, 0], y[:, 1], y[:, 2], label=eq_name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{eq_name} Solution in 3D (Elapsed time: {elapsed_time:.2f} ms)')
    plt.legend()

    plt.show()


# Example usage
t_span = (0, 50)  # time range
num_points = 10000  # number of points in the solution

# Get user input for differential equation selection
print("Select a differential equation from the following list:")
for i, eq in enumerate(differential_eqs):
    print(f"{i + 1}. {eq}")
eq_index = int(input("Enter the differential equation number: ")) - 1

selected_eq_name = list(differential_eqs.keys())[eq_index]
eq_str, eq_func = differential_eqs[selected_eq_name]

# Set parameters and initial conditions based on the selected equation
if selected_eq_name == "Lorenz System":
    sigma = float(input("Enter the parameter sigma: "))
    rho = float(input("Enter the parameter rho: "))
    beta = float(input("Enter the parameter beta: "))
    y0 = [float(input("Enter the initial condition x0: ")),
          float(input("Enter the initial condition y0: ")),
          float(input("Enter the initial condition z0: "))]
    params = (sigma, rho, beta)
elif selected_eq_name == "Rossler System":
    a = float(input("Enter the parameter a: "))
    b = float(input("Enter the parameter b: "))
    c = float(input("Enter the parameter c: "))
    y0 = [float(input("Enter the initial condition x0: ")),
          float(input("Enter the initial condition y0: ")),
          float(input("Enter the initial condition z0: "))]
    params = (a, b, c)
elif selected_eq_name == "Chua's Circuit":
    alpha = float(input("Enter the parameter alpha: "))
    beta = float(input("Enter the parameter beta: "))
    h = lambda x: m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
    m0 = -1.143
    m1 = -0.714
    y0 = [float(input("Enter the initial condition x0: ")),
          float(input("Enter the initial condition y0: ")),
          float(input("Enter the initial condition z0: "))]
    params = (alpha, beta, h)
elif selected_eq_name == "Lorenz '96 Model":
    F = float(input("Enter the parameter F: "))
    N = int(input("Enter the number of variables N (suggested N>=3): "))
    y0 = [float(input(f"Enter the initial condition x{i}: ")) for i in range(N)]
    params = (F,)
else:
    print("Invalid selection.")
    exit()

# Solve and plot the ODE in 3D
solve_ode_3d(selected_eq_name, y0, t_span, num_points, eq_func, eq_str, *params)