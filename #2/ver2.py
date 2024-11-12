import numpy as np
import matplotlib.pyplot as plt
import time

# Define differential equations and their string representations
differential_eqs = {
    "Damped Harmonic Oscillator": (
        "d2y/dt2 + 2γ dy/dt + ω^2 y = 0",
        lambda t, y, gamma, omega: np.array([y[1], -2*gamma*y[1] - omega**2*y[0]])
    ),
    "Van der Pol Oscillator": (
        "d2y/dt2 - μ(1 - y^2)dy/dt + y = 0",
        lambda t, y, mu: np.array([y[1], mu*(1 - y[0]**2)*y[1] - y[0]])
    ),
    "Duffing Equation": (
        "d2y/dt2 + δ dy/dt + α y + β y^3 = γ cos(ωt)",
        lambda t, y, delta, alpha, beta, gamma, omega: np.array([y[1], -delta*y[1] - alpha*y[0] - beta*y[0]**3 + gamma*np.cos(omega*t)])
    ),
    "SIR Model (Epidemiology)": (
        "dS/dt = -βSI, dI/dt = βSI - γI, dR/dt = γI",
        lambda t, z, beta, gamma: np.array([-beta*z[0]*z[1], beta*z[0]*z[1] - gamma*z[1], gamma*z[1]])
    ),
    "Lorenz System": (
        "dx/dt = σ(y - x), dy/dt = x(ρ - z) - y, dz/dt = xy - βz",
        lambda t, z, sigma, rho, beta: np.array([sigma*(z[1] - z[0]), z[0]*(rho - z[2]) - z[1], z[0]*z[1] - beta*z[2]])
    ),
    "Brusselator (Chemical Oscillations)": (
        "dx/dt = A - (B + 1)x + x^2y, dy/dt = Bx - x^2y",
        lambda t, z, A, B: np.array([A - (B + 1)*z[0] + z[0]**2*z[1], B*z[0] - z[0]**2*z[1]])
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

# Define the main function to solve the ODE and plot the results
def solve_ode(method, y0, t_span, num_points, f, eq_str, *params):
    t = np.linspace(t_span[0], t_span[1], num_points)

    methods = {
        'euler': euler_method,
        'improved_euler': improved_euler_method,
        'rk4': rk4_method
    }

    start_time = time.time()

    if method in methods:
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
num_points = 1000  # number of points in the solution

# List of methods
methods = ['euler', 'improved_euler', 'rk4']

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
        if selected_eq_name == "Damped Harmonic Oscillator":
            gamma = float(input("Enter the damping coefficient gamma: "))
            omega = float(input("Enter the angular frequency omega: "))
            y0 = [float(input("Enter the initial position y0: ")), float(input("Enter the initial velocity dy0: "))]
            params = (gamma, omega)
        elif selected_eq_name == "Van der Pol Oscillator":
            mu = float(input("Enter the nonlinearity parameter mu: "))
            y0 = [float(input("Enter the initial position y0: ")), float(input("Enter the initial velocity dy0: "))]
            params = (mu,)
        elif selected_eq_name == "Duffing Equation":
            delta = float(input("Enter the damping coefficient delta: "))
            alpha = float(input("Enter the linear stiffness coefficient alpha: "))
            beta = float(input("Enter the nonlinear stiffness coefficient beta: "))
            gamma = float(input("Enter the amplitude of the driving force gamma: "))
            omega = float(input("Enter the frequency of the driving force omega: "))
            y0 = [float(input("Enter the initial position y0: ")), float(input("Enter the initial velocity dy0: "))]
            params = (delta, alpha, beta, gamma, omega)
        elif selected_eq_name == "SIR Model (Epidemiology)":
            beta = float(input("Enter the infection rate beta: "))
            gamma = float(input("Enter the recovery rate gamma: "))
            y0 = [float(input("Enter the initial susceptible population S0: ")),
                  float(input("Enter the initial infected population I0: ")),
                  float(input("Enter the initial recovered population R0: "))]
            params = (beta, gamma)
        elif selected_eq_name == "Lorenz System":
            sigma = float(input("Enter the parameter sigma: "))
            rho = float(input("Enter the parameter rho: "))
            beta = float(input("Enter the parameter beta: "))
            y0 = [float(input("Enter the initial condition x0: ")),
                  float(input("Enter the initial condition y0: ")),
                  float(input("Enter the initial condition z0: "))]
            params = (sigma, rho, beta)
        elif selected_eq_name == "Brusselator (Chemical Oscillations)":
            A = float(input("Enter the parameter A: "))
            B = float(input("Enter the parameter B: "))
            y0 = [float(input("Enter the initial condition x0: ")),
                  float(input("Enter the initial condition y0: "))]
            params = (A, B)
        else:
            print("Invalid selection.")
            continue

        # Iterate over methods and plot the results
        for method in methods:
            solve_ode(method, y0, t_span, num_points, eq_func, eq_str, *params)
    else:
        print("Invalid selection.")