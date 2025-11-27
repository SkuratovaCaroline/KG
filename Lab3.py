import tkinter as tk
from tkinter import ttk, messagebox
from time import perf_counter
from math import floor, ceil

# =========================
# Config
# =========================
SCALE = 20          # pixels per grid cell
GRID_SIZE = 30      # grid cells per axis (logical integer extent)
AXIS_TICK_STEP = 1  # integer step between tick labels


# =========================
# Utilities: mapping and drawing
# =========================
class GridMapper:
    """
    Maps integer coordinates (x, y) to canvas pixel rectangles,
    given scale and origin placement (centered or not).
    """
    def __init__(self, canvas: tk.Canvas, scale: int, grid_size: int, centered: bool = True):
        self.canvas = canvas
        self.scale = scale
        self.grid_size = grid_size
        self.centered = centered
        self.width = int(canvas["width"])
        self.height = int(canvas["height"])
        # Logical bounds: integers in [-grid_size//2, +grid_size//2] if centered,
        # else [0, grid_size-1]
        if centered:
            self.min_x = -grid_size // 2
            self.max_x = grid_size // 2
            self.min_y = -grid_size // 2
            self.max_y = grid_size // 2
            # Origin in the center cell
            self.origin_x = self.width // 2
            self.origin_y = self.height // 2
        else:
            self.min_x = 0
            self.max_x = grid_size - 1
            self.min_y = 0
            self.max_y = grid_size - 1
            self.origin_x = 0
            self.origin_y = 0

    def logical_bounds(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def to_canvas_rect(self, x: int, y: int):
        """
        Map integer (x,y) (with y increasing upward logically)
        to canvas rectangle coords. Tk canvas has y increasing downward,
        so we flip y.
        Top-left pixel of cell is:
          X = origin_x + x*scale if centered (shifted),
              else x*scale
          Y = origin_y - y*scale if centered (flip),
              else (height - (y+1)*scale) if you want origin bottom-left.
        We define consistent centered mapping: origin in center.
        """
        if self.centered:
            px = self.origin_x + x * self.scale
            py = self.origin_y - (y + 1) * self.scale  # top-left of cell at logical y
        else:
            # origin top-left, y grows downward logically (for simplicity)
            px = x * self.scale
            py = y * self.scale
        return (px, py, px + self.scale, py + self.scale)

    def in_bounds(self, x: int, y: int) -> bool:
        return (self.min_x <= x <= self.max_x) and (self.min_y <= y <= self.max_y)

    def draw_pixel(self, x: int, y: int, color: str = "#e74c3c"):
        if not self.in_bounds(x, y):
            return
        x0, y0, x1, y1 = self.to_canvas_rect(x, y)
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

    def draw_axes_and_grid(self):
        # Grid lines
        self.canvas.delete("all")
        # Background
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill="#ffffff", outline="")

        # Vertical grid lines
        for gx in range(self.min_x, self.max_x + 1):
            x0 = (self.origin_x + gx * self.scale) if self.centered else gx * self.scale
            self.canvas.create_line(x0, 0, x0, self.height, fill="#f0f0f0")
        # Horizontal grid lines
        # For centered, lines at origin_y - y*scale
        for gy in range(self.min_y, self.max_y + 1):
            y0 = (self.origin_y - gy * self.scale) if self.centered else gy * self.scale
            self.canvas.create_line(0, y0, self.width, y0, fill="#f0f0f0")

        # Axes
        x_axis_y = self.origin_y if self.centered else 0
        y_axis_x = self.origin_x if self.centered else 0
        self.canvas.create_line(0, x_axis_y, self.width, x_axis_y, fill="#000000", width=2)
        self.canvas.create_line(y_axis_x, 0, y_axis_x, self.height, fill="#000000", width=2)

        # Tick labels
        font = ("Arial", 9)
        for gx in range(self.min_x, self.max_x + 1, AXIS_TICK_STEP):
            x0 = (self.origin_x + gx * self.scale) if self.centered else gx * self.scale
            self.canvas.create_text(x0 + 3, x_axis_y + 10, text=str(gx), anchor="nw", font=font, fill="#555555")
        for gy in range(self.min_y, self.max_y + 1, AXIS_TICK_STEP):
            y0 = (self.origin_y - gy * self.scale) if self.centered else gy * self.scale
            self.canvas.create_text(y_axis_x + 3, y0 + 3, text=str(gy), anchor="nw", font=font, fill="#555555")


# =========================
# Algorithms
# =========================
def step_by_step_line(x0, y0, x1, y1):
    """
    Floating-point incremental 'пошаговый' algorithm:
    iterate along x if |dx| >= |dy| else along y,
    using slope m and rounding to nearest pixel.
    Returns plotted (x,y) list and elapsed time.
    """
    t0 = perf_counter()
    points = []
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) >= abs(dy):
        # iterate along x
        step = 1 if dx >= 0 else -1
        m = dy / dx if dx != 0 else 0.0
        y = y0
        for x in range(x0, x1 + step, step):
            points.append((x, round(y)))
            y += m * step
    else:
        # iterate along y
        step = 1 if dy >= 0 else -1
        m = dx / dy if dy != 0 else 0.0
        x = x0
        for y in range(y0, y1 + step, step):
            points.append((round(x), y))
            x += m * step
    elapsed = perf_counter() - t0
    return points, elapsed


def dda_line(x0, y0, x1, y1):
    """
    DDA: normalize by max(|dx|, |dy|), increment fractional,
    round each point to integer.
    """
    t0 = perf_counter()
    points = []
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        points.append((x0, y0))
        return points, perf_counter() - t0
    x_inc = dx / steps
    y_inc = dy / steps
    x = x0
    y = y0
    for _ in range(steps + 1):
        points.append((round(x), round(y)))
        x += x_inc
        y += y_inc
    elapsed = perf_counter() - t0
    return points, elapsed


def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham line algorithm (integer arithmetic).
    Supports any octant via reflection.
    """
    t0 = perf_counter()
    points = []

    dx = x1 - x0
    dy = y1 - y0
    steep = abs(dy) > abs(dx)

    # Rotate if steep to simplify (swap x and y)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        dx, dy = x1 - x0, y1 - y0

    # Ensure left-to-right
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        dx, dy = x1 - x0, y1 - y0

    y_step = 1 if dy >= 0 else -1
    dy = abs(dy)
    d = 2 * dy - dx  # decision

    y = y0
    for x in range(x0, x1 + 1):
        if steep:
            points.append((y, x))
        else:
            points.append((x, y))
        if d >= 0:
            y += y_step
            d -= 2 * dx
        d += 2 * dy

    elapsed = perf_counter() - t0
    return points, elapsed


def bresenham_circle(cx, cy, r):
    """
    Bresenham/Midpoint circle algorithm with 8-way symmetry.
    Center (cx, cy), integer radius r.
    """
    t0 = perf_counter()
    points = []

    x = 0
    y = r
    d = 3 - 2 * r

    def plot_circle_points(cx, cy, x, y, out_list):
        out_list.extend([
            (cx + x, cy + y),
            (cx - x, cy + y),
            (cx + x, cy - y),
            (cx - x, cy - y),
            (cx + y, cy + x),
            (cx - y, cy + x),
            (cx + y, cy - x),
            (cx - y, cy - x),
        ])

    while x <= y:
        plot_circle_points(cx, cy, x, y, points)
        if d <= 0:
            d += 4 * x + 6
        else:
            d += 4 * (x - y) + 10
            y -= 1
        x += 1

    elapsed = perf_counter() - t0
    # Optionally deduplicate points (due to symmetry overlaps when x==y)
    unique = list(dict.fromkeys(points))
    return unique, elapsed


# =========================
# UI: Application
# =========================
class RasterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rasterization Algorithms: Step-by-step, DDA, Bresenham (line), Bresenham (circle)")
        self.geometry(f"{SCALE*GRID_SIZE + 360}x{SCALE*GRID_SIZE + 60}")
        self.resizable(False, False)

        # Left: canvas
        canvas_width = SCALE * GRID_SIZE
        canvas_height = SCALE * GRID_SIZE
        self.canvas = tk.Canvas(self, width=canvas_width, height=canvas_height, bg="#ffffff", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Right: controls
        controls = ttk.Frame(self)
        controls.grid(row=0, column=1, padx=10, pady=10, sticky="ns")

        # Mapper
        self.centered_var = tk.BooleanVar(value=True)
        self.mapper = GridMapper(self.canvas, SCALE, GRID_SIZE, centered=self.centered_var.get())

        # Draw initial grid
        self.mapper.draw_axes_and_grid()

        # Algorithm selection
        ttk.Label(controls, text="Algorithm").grid(row=0, column=0, sticky="w")
        self.algo_var = tk.StringVar(value="Step-by-step")
        algo_menu = ttk.Combobox(controls, textvariable=self.algo_var, values=[
            "Step-by-step", "DDA", "Bresenham (line)", "Bresenham (circle)"
        ], state="readonly", width=25)
        algo_menu.grid(row=1, column=0, pady=5, sticky="w")

        # Inputs
        self.x0_var = tk.IntVar(value=2)
        self.y0_var = tk.IntVar(value=3)
        self.x1_var = tk.IntVar(value=11)
        self.y1_var = tk.IntVar(value=7)
        self.cx_var = tk.IntVar(value=0)
        self.cy_var = tk.IntVar(value=0)
        self.r_var = tk.IntVar(value=8)

        line_frame = ttk.LabelFrame(controls, text="Line endpoints (x0,y0)-(x1,y1)")
        line_frame.grid(row=2, column=0, pady=5, sticky="ew")
        self._add_spin(line_frame, "x0", self.x0_var, 0)
        self._add_spin(line_frame, "y0", self.y0_var, 1)
        self._add_spin(line_frame, "x1", self.x1_var, 2)
        self._add_spin(line_frame, "y1", self.y1_var, 3)

        circle_frame = ttk.LabelFrame(controls, text="Circle center (cx,cy) and radius r")
        circle_frame.grid(row=3, column=0, pady=5, sticky="ew")
        self._add_spin(circle_frame, "cx", self.cx_var, 0)
        self._add_spin(circle_frame, "cy", self.cy_var, 1)
        self._add_spin(circle_frame, "r", self.r_var, 2, minval=0)

        # Centered toggle
        center_chk = ttk.Checkbutton(controls, text="Centered origin", variable=self.centered_var, command=self.on_toggle_center)
        center_chk.grid(row=4, column=0, pady=5, sticky="w")

        # Buttons
        btn_frame = ttk.Frame(controls)
        btn_frame.grid(row=5, column=0, pady=10, sticky="ew")
        ttk.Button(btn_frame, text="Draw", command=self.on_draw).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.on_clear).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Draw axes/grid", command=self.on_redraw_grid).grid(row=0, column=2, padx=5)

        # Timing display
        self.time_var = tk.StringVar(value="Timing: —")
        ttk.Label(controls, textvariable=self.time_var).grid(row=6, column=0, pady=10, sticky="w")

        # Legend
        legend = ttk.LabelFrame(controls, text="Legend and notes")
        legend.grid(row=7, column=0, pady=5, sticky="ew")
        ttk.Label(legend, text="- Red: pixels plotted by algorithm\n- Black axes; light gray grid\n- Integer coordinates map to discrete cells\n- Scale defines pixel size per cell").grid(row=0, column=0, sticky="w")

        # Status
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(controls, textvariable=self.status_var, foreground="#2c3e50").grid(row=8, column=0, pady=10, sticky="w")

    def _add_spin(self, parent, label, var, row, minval=-GRID_SIZE, maxval=GRID_SIZE):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, pady=2, sticky="w")
        ttk.Label(f, text=f"{label}:").grid(row=0, column=0)
        spin = ttk.Spinbox(f, from_=minval, to=maxval, textvariable=var, width=6)
        spin.grid(row=0, column=1, padx=4)

    def on_toggle_center(self):
        self.mapper = GridMapper(self.canvas, SCALE, GRID_SIZE, centered=self.centered_var.get())
        self.mapper.draw_axes_and_grid()
        self.status_var.set("Origin mapping updated.")

    def on_redraw_grid(self):
        self.mapper.draw_axes_and_grid()
        self.status_var.set("Grid re-rendered.")

    def on_clear(self):
        self.mapper.draw_axes_and_grid()
        self.status_var.set("Cleared.")

    def on_draw(self):
        algo = self.algo_var.get()
        points = []
        elapsed = 0.0

        try:
            if algo == "Step-by-step":
                points, elapsed = step_by_step_line(self.x0_var.get(), self.y0_var.get(), self.x1_var.get(), self.y1_var.get())
            elif algo == "DDA":
                points, elapsed = dda_line(self.x0_var.get(), self.y0_var.get(), self.x1_var.get(), self.y1_var.get())
            elif algo == "Bresenham (line)":
                points, elapsed = bresenham_line(self.x0_var.get(), self.y0_var.get(), self.x1_var.get(), self.y1_var.get())
            elif algo == "Bresenham (circle)":
                points, elapsed = bresenham_circle(self.cx_var.get(), self.cy_var.get(), self.r_var.get())
            else:
                messagebox.showerror("Error", "Unknown algorithm selected.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Computation failed: {e}")
            return

        # Draw
        for (x, y) in points:
            self.mapper.draw_pixel(x, y, color="#e74c3c")

        self.time_var.set(f"Timing: {algo} -> {elapsed*1000:.3f} ms, {len(points)} pixels")
        self.status_var.set(f"Drawn {len(points)} pixels using {algo}.")

        # Optional: draw endpoints for line algorithms
        if "line" in algo or "Step" in algo or "DDA" in algo:
            ex0, ey0 = self.x0_var.get(), self.y0_var.get()
            ex1, ey1 = self.x1_var.get(), self.y1_var.get()
            self.mapper.draw_pixel(ex0, ey0, color="#2ecc71")
            self.mapper.draw_pixel(ex1, ey1, color="#2ecc71")

        if "circle" in algo:
            self.mapper.draw_pixel(self.cx_var.get(), self.cy_var.get(), color="#2ecc71")

    def run(self):
        self.mainloop()


if __name__ == "__main__":
    app = RasterApp()
    app.run()
