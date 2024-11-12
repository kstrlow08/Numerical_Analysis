import tkinter as tk
from tkinter import ttk
from tkinter.colorchooser import askcolor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation


# Define differential equation and solvers
def f(t, y):
    return -2 * y + 2 * t


def df_dt(t, y):
    return 2


def df_dy(t, y):
    return -2


def euler_method(f, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i - 1] + h * f(t[i - 1], y[i - 1])
    return t, y


def improved_euler_method(f, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = h * f(t[i - 1], y[i - 1])
        k2 = h * f(t[i - 1] + h, y[i - 1] + k1)
        y[i] = y[i - 1] + 0.5 * (k1 + k2)
    return t, y


def rk4_method(f, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = h * f(t[i - 1], y[i - 1])
        k2 = h * f(t[i - 1] + 0.5 * h, y[i - 1] + 0.5 * k1)
        k3 = h * f(t[i - 1] + 0.5 * h, y[i - 1] + 0.5 * k2)
        k4 = h * f(t[i], y[i - 1] + k3)
        y[i] = y[i - 1] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


def taylor_method(f, df_dt, df_dy, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        fy = f(t[i - 1], y[i - 1])
        ft = df_dt(t[i - 1], y[i - 1])
        fyy = df_dy(t[i - 1], y[i - 1])
        y[i] = y[i - 1] + h * fy + 0.5 * h ** 2 * (ft + fyy * fy)
    return t, y


def midpoint_method(f, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = h * f(t[i - 1], y[i - 1])
        y_mid = y[i - 1] + 0.5 * k1
        y[i] = y[i - 1] + h * f(t[i - 1] + 0.5 * h, y_mid)
    return t, y


def ralston_method(f, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = h * f(t[i - 1], y[i - 1])
        k2 = h * f(t[i - 1] + 2 / 3 * h, y[i - 1] + 2 / 3 * k1)
        y[i] = y[i - 1] + 1 / 4 * k1 + 3 / 4 * k2
    return t, y


def heun_method(f, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = h * f(t[i - 1], y[i - 1])
        k2 = h * f(t[i - 1] + h, y[i - 1] + k1)
        y[i] = y[i - 1] + 0.5 * (k1 + k2)
    return t, y


def adams_bashforth_method(f, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros(len(t))
    y[0] = y0
    if len(t) > 1:
        k1 = h * f(t[0], y[0])
        y[1] = y[0] + k1
        for i in range(2, len(t)):
            y[i] = y[i - 1] + 1.5 * h * f(t[i - 1], y[i - 1]) - 0.5 * h * f(t[i - 2], y[i - 2])
    return t, y


# Global settings for graph display
graph_settings = {
    'Euler Method': {'show': True, 'color': 'black', 'style': '-'},
    'Improved Euler Method': {'show': True, 'color': 'black', 'style': '-'},
    'RK4 Method': {'show': True, 'color': 'black', 'style': '-'},
    'Taylor Method': {'show': True, 'color': 'black', 'style': '-'},
    'Midpoint Method': {'show': True, 'color': 'black', 'style': '-'},
    'Ralston Method': {'show': True, 'color': 'black', 'style': '-'},
    'Heun Method': {'show': True, 'color': 'black', 'style': '-'},
    'Adams-Bashforth Method': {'show': True, 'color': 'black', 'style': '-'},
    'y0': 1  # Default initial condition
}


# GUI Application
class DiffEqSolverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Differential Equation Solver")
        self.geometry("1000x800")
        self.current_frame = None
        self.show_frame(StartFrame)

    def show_frame(self, frame_class):
        new_frame = frame_class(self)
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = new_frame
        self.current_frame.pack(expand=True, fill='both')


class StartFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        label = tk.Label(self, text="Differential Equation Solver", font=("Helvetica", 24))
        label.pack(pady=20)
        start_button = ttk.Button(self, text="Start", command=lambda: master.show_frame(MenuFrame))
        start_button.pack(pady=20)


class MenuFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        view_button = ttk.Button(self, text="View", command=lambda: master.show_frame(ViewFrame))
        view_button.pack(side="left", expand=True, fill='both')
        mode_button = ttk.Button(self, text="Mode", command=lambda: master.show_frame(ModeFrame))
        mode_button.pack(side="left", expand=True, fill='both')
        settings_button = ttk.Button(self, text="Settings", command=lambda: master.show_frame(SettingsFrame))
        settings_button.pack(side="left", expand=True, fill='both')


class ViewFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.page = 0
        self.methods = [
            'Euler Method', 'Improved Euler Method', 'RK4 Method', 'Taylor Method',
            'Midpoint Method', 'Ralston Method', 'Heun Method', 'Adams-Bashforth Method'
        ]
        self.animation_running = False
        self.create_widgets()

    def create_widgets(self):
        back_button = ttk.Button(self, text="Back", command=lambda: self.master.show_frame(MenuFrame))
        back_button.pack(side="top", anchor="w")

        self.page_label = tk.Label(self, text=f"Page {self.page + 1}")
        self.page_label.pack(side="top")

        next_button = ttk.Button(self, text="Next Page", command=self.next_page)
        next_button.pack(side="bottom", anchor="e")

        prev_button = ttk.Button(self, text="Previous Page", command=self.prev_page)
        prev_button.pack(side="bottom", anchor="w")

        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(expand=True, fill='both')

        self.ani = None
        self.start_pause_button = ttk.Button(self, text="Start/Pause", command=self.toggle_animation)
        self.start_pause_button.pack(side="bottom")

        self.plot_method()

    def plot_method(self):
        t0, tf, h = 0, 10, 0.1
        y0 = graph_settings.get('y0', 1)
        methods = {
            'Euler Method': euler_method,
            'Improved Euler Method': improved_euler_method,
            'RK4 Method': rk4_method,
            'Taylor Method': taylor_method,
            'Midpoint Method': midpoint_method,
            'Ralston Method': ralston_method,
            'Heun Method': heun_method,
            'Adams-Bashforth Method': adams_bashforth_method
        }

        method_name = self.methods[self.page]

        self.ax.clear()
        self.ax.set_xlim(t0, tf)

        if self.ani:
            self.ani.event_source.stop()

        time_steps = np.arange(t0, tf, h)

        def update(frame):
            if graph_settings[method_name]['show']:
                method = methods[method_name]
                t, y = method(f, y0, t0, tf, h)
                self.ax.plot(t[:frame], y[:frame], graph_settings[method_name]['style'],
                             color=graph_settings[method_name]['color'], label=method_name)
                self.ax.legend()
            self.canvas.draw()

        self.ani = FuncAnimation(self.figure, update, frames=len(time_steps), interval=100)

    def next_page(self):
        self.page = (self.page + 1) % len(self.methods)
        self.page_label.config(text=f"Page {self.page + 1}")
        self.plot_method()

    def prev_page(self):
        self.page = (self.page - 1) % len(self.methods)
        self.page_label.config(text=f"Page {self.page + 1}")
        self.plot_method()

    def toggle_animation(self):
        if self.ani:
            if self.animation_running:
                self.ani.event_source.stop()
                self.start_pause_button.config(text="Start")
            else:
                self.ani.event_source.start()
                self.start_pause_button.config(text="Pause")
            self.animation_running = not self.animation_running


class SettingsFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        back_button = ttk.Button(self, text="Back", command=lambda: master.show_frame(MenuFrame))
        back_button.pack(side="top", anchor="w")

        y0_label = tk.Label(self, text="Initial Condition y0:")
        y0_label.pack(side="left")
        self.y0_entry = tk.Entry(self)
        self.y0_entry.pack(side="left")
        self.y0_entry.insert(0, graph_settings.get('y0', 1))
        set_y0_button = ttk.Button(self, text="Set y0", command=self.set_y0)
        set_y0_button.pack(side="left")

        self.create_settings_controls()

    def create_settings_controls(self):
        for method in graph_settings:
            if method != 'y0':
                self.create_method_controls(method)

    def create_method_controls(self, method):
        frame = tk.LabelFrame(self, text=method)
        frame.pack(fill="x", padx=5, pady=5)

        var_show = tk.BooleanVar(value=graph_settings[method]['show'])
        check_show = tk.Checkbutton(frame, text="Show", variable=var_show,
                                    command=lambda m=method, v=var_show: self.update_setting(m, 'show', v.get()))
        check_show.pack(side="left")

        color_button = ttk.Button(frame, text="Color",
                                  command=lambda m=method: self.choose_color(m))
        color_button.pack(side="left")

        style_menu = ttk.Combobox(frame, values=["-", "--", "-.", ":"], state="readonly")
        style_menu.set(graph_settings[method]['style'])
        style_menu.bind("<<ComboboxSelected>>",
                        lambda e, m=method, sm=style_menu: self.update_setting(m, 'style', sm.get()))
        style_menu.pack(side="left")

    def update_setting(self, method, setting, value):
        graph_settings[method][setting] = value

    def choose_color(self, method):
        color_code = askcolor(title=f"Choose color for {method}")
        if color_code[1] is not None:
            self.update_setting(method, 'color', color_code[1])

    def set_y0(self):
        y0 = self.y0_entry.get()
        try:
            y0 = float(y0)
            graph_settings['y0'] = y0
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter a valid number for y0.")


class ModeFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        back_button = ttk.Button(self, text="Back", command=lambda: master.show_frame(MenuFrame))
        back_button.pack(side="top", anchor="w")
        # Add mode selection UI here
        # ...


if __name__ == "__main__":
    app = DiffEqSolverApp()
    app.mainloop()
