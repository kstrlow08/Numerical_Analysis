import numpy as np
import matplotlib.pyplot as plt
import time

# Define differential equations and their string representations
differential_eqs = {
    "Radioactive Decay Law": ("dy/dt = -ky", lambda t, y, k: -k * y),
    "Newton's Law of Cooling": ("dy/dt = -k(y - T_env)", lambda t, y, k, T_env: -k * (y - T_env)),
    "Simple Harmonic Oscillator": ("d2y/dt2 = -ky", lambda t, y, k, omega: np.array([y[1], -omega ** 2 * y[0]])),
    "Logistic Growth Model": ("dy/dt = r*y(1 - y/K)", lambda t, y, r, K: r * y * (1 - y / K)),
    "Lotka-Volterra Equations": (
        "dx/dt = ax - bxy, dy/dt = -cy + dxy",
        lambda t, z, a, b, c, d: np.array([a * z[0] - b * z[0] * z[1], -c * z[1] + d * z[0] * z[1]])
    )
}


# Implement the Euler method
def euler_method(f, y0, t, *params):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        y[i] = y[i - 1] + dt * f(t[i - 1], y[i - 1], *params)
    return y


# Implement the Improved Euler method
def improved_euler_method(f, y0, t, *params):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        y_pred = y[i - 1] + dt * f(t[i - 1], y[i - 1], *params)
        y[i] = y[i - 1] + dt * 0.5 * (f(t[i - 1], y[i - 1], *params) + f(t[i], y_pred, *params))
    return y


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


# Implement the Taylor method (second-order)
def taylor_method(f, df, y0, t, *params):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        y[i] = y[i - 1] + dt * f(t[i - 1], y[i - 1], *params) + 0.5 * (dt ** 2) * df(t[i - 1], y[i - 1], *params)
    return y


# Implement the Midpoint method
def midpoint_method(f, y0, t, *params):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        k1 = f(t[i - 1], y[i - 1], *params)
        k2 = f(t[i - 1] + dt / 2, y[i - 1] + k1 * dt / 2, *params)
        y[i] = y[i - 1] + k2 * dt
    return y


# Implement the Ralston method
def ralston_method(f, y0, t, *params):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        k1 = f(t[i - 1], y[i - 1], *params)
        k2 = f(t[i - 1] + (3 / 4) * dt, y[i - 1] + (3 / 4) * k1 * dt, *params)
        y[i] = y[i - 1] + (1 / 3) * k1 * dt + (2 / 3) * k2 * dt
    return y


# Implement the Adams-Bashforth method (two-step)
def adams_bashforth_method(f, y0, t, *params):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    if len(t) > 1:
        dt = t[1] - t[0]
        y[1] = y[0] + dt * f(t[0], y[0], *params)  # Using Euler for the first step
    for i in range(2, len(t)):
        dt = t[i] - t[i - 1]
        y[i] = y[i - 1] + dt * (1.5 * f(t[i - 1], y[i - 1], *params) - 0.5 * f(t[i - 2], y[i - 2], *params))
    return y


# Define the main function to solve the ODE and plot the results
def solve_ode(method, y0, t_span, num_points, f, eq_str, *params):
    t = np.linspace(t_span[0], t_span[1], num_points)

    methods = {
        'euler': euler_method,
        'improved_euler': improved_euler_method,
        'rk4': rk4_method,
        'taylor': taylor_method,
        'midpoint': midpoint_method,
        'ralston': ralston_method,
        'adams_bashforth': adams_bashforth_method
    }

    start_time = time.time()

    if method == 'taylor':
        # Define the second derivative for the Taylor method
        def df(t, y, *params):
            return f(t, y, *params) - 2 * t  # Example: d^2y/dt^2 for the given equation

        y = taylor_method(f, df, y0, t, *params)
    elif method in methods:
        y = methods[method](f, y0, t, *params)
    else:
        raise ValueError("Method not recognized.")

    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Plot the results
    plt.figure(figsize=(10, 8))  # Adjust the figure size to prevent text clipping
    if len(y0) == 1:
        plt.plot(t, y[:, 0], label=method)
        plt.ylabel('y(t)')
    else:
        for i in range(y.shape[1]):
            plt.plot(t, y[:, i], label=f'y{i}(t)')
    plt.xlabel('t')
    plt.title(f'Solution of ODE using {method} (Elapsed time: {elapsed_time:.2f} ms)')
    plt.legend()

    # Add space at the bottom for the equation
    plt.subplots_adjust(bottom=0.2)

    # Display the differential equation below the plot
    plt.figtext(0.5, 0.05, eq_str, wrap=True, horizontalalignment='center', fontsize=12)

    plt.show()


# Example usage
t_span = (0, 2)  # time range
num_points = 100  # number of points in the solution

# List of methods
methods = ['euler', 'improved_euler', 'rk4', 'taylor', 'midpoint', 'ralston', 'adams_bashforth']

while True:
    # Get user input for differential equation selection
    print("Select a differential equation from the following list (or enter 0 to exit):")
    for i, eq in enumerate(differential_eqs):
        print(f"{i + 1}. {eq}")
    eq_index = int(input("Enter the differential equation number: ")) - 1

    if eq_index == -1:
        print("Exiting the program.")
        break

    if 0 <= eq_index < len(differential_eqs):
        selected_eq_name = list(differential_eqs.keys())[eq_index]
        eq_str, eq_func = differential_eqs[selected_eq_name]

        # Get initial conditions and parameters based on the selected equation
        if selected_eq_name == "Radioactive Decay Law":
            k = float(input("Enter the decay constant k: "))
            y0 = [float(input("Enter the initial value y0: "))]
            params = (k,)
        elif selected_eq_name == "Newton's Law of Cooling":
            k = float(input("Enter the cooling constant k: "))
            T_env = float(input("Enter the environment temperature T_env: "))
            y0 = [float(input("Enter the initial temperature y0: "))]
            params = (k, T_env)
        elif selected_eq_name == "Simple Harmonic Oscillator":
            k = float(input("Enter the spring constant k: "))
            omega = float(input("Enter the angular frequency omega: "))
            y0 = [float(input("Enter the initial position y0: ")), float(input("Enter the initial velocity dy0: "))]
            params = (k, omega)
        elif selected_eq_name == "Logistic Growth Model":
            r = float(input("Enter the growth rate r: "))
            K = float(input("Enter the carrying capacity K: "))
            y0 = [float(input("Enter the initial population y0: "))]
            params = (r, K)
        elif selected_eq_name == "Lotka-Volterra Equations":
            a = float(input("Enter the prey growth rate a: "))
            b = float(input("Enter the predation rate b: "))
            c = float(input("Enter the predator death rate c: "))
            d = float(input("Enter the predator reproduction rate d: "))
            y0 = [float(input("Enter the initial prey population x0: ")), float(input("Enter the initial predator population y0: "))]
            params = (a, b, c, d)

        while True:
            # Get user input for method selection
            print("Select a method from the following list (or enter 0 to go back to equation selection):")
            for i, method in enumerate(methods):
                print(f"{i + 1}. {method}")
            method_index = int(input("Enter the method number: ")) - 1

            if method_index == -1:
                break

            if 0 <= method_index < len(methods):
                selected_method = methods[method_index]
                solve_ode(selected_method, y0, t_span, num_points, eq_func, eq_str, *params)

    else:
        print("Invalid selection. Please try again.")