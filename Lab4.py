import matplotlib.pyplot as plt

# -----------------------------
# Алгоритм Лианга–Барски
# -----------------------------
def liang_barsky(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    dx = x2 - x1
    dy = y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u2: return None
                if t > u1: u1 = t
            else:
                if t < u1: return None
                if t < u2: u2 = t

    x1_clip = x1 + u1 * dx
    y1_clip = y1 + u1 * dy
    x2_clip = x1 + u2 * dx
    y2_clip = y1 + u2 * dy
    return (x1_clip, y1_clip, x2_clip, y2_clip)


# -----------------------------
# Визуализация
# -----------------------------
def visualize(segments, window):
    fig, ax = plt.subplots()
    xmin, ymin, xmax, ymax = window

    # Рисуем окно
    rect_x = [xmin, xmax, xmax, xmin, xmin]
    rect_y = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(rect_x, rect_y, 'b-', label="Окно")

    # Рисуем исходные отрезки и отсечённые
    for i, (x1, y1, x2, y2) in enumerate(segments):
        ax.plot([x1, x2], [y1, y2], 'r--', label="Исходный" if i == 0 else "")
        clipped = liang_barsky(x1, y1, x2, y2, xmin, ymin, xmax, ymax)
        if clipped:
            cx1, cy1, cx2, cy2 = clipped
            ax.plot([cx1, cx2], [cy1, cy2], 'g-', linewidth=2, label="Отсечённый" if i == 0 else "")

    ax.set_aspect('equal')
    ax.legend()
    plt.show()


# -----------------------------
# Основной запуск
# -----------------------------
if __name__ == "__main__":
    # Входные данные прямо в коде
    segments = [
        (10, 10, 80, 20),
        (-20, -10, 120, 50),
        (30, 90, 90, 30),
        (0, 0, 100, 100),
        (60, -20, 60, 120)
    ]
    window = (0, 0, 100, 80)  # xmin, ymin, xmax, ymax

    visualize(segments, window)
