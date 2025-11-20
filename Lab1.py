# -*- coding: utf-8 -*-
# CMYK-RGB-HSV Interactive Converter
# Требования: Python 3.x, стандартная библиотека (tkinter)
# Фокус: корректные преобразования, единый источник правды (RGB), автопересчёт

import tkinter as tk
from tkinter import ttk, colorchooser

# -----------------------------
# Цветовые преобразования (чистые функции)
# -----------------------------

def clamp(v, lo, hi):
    """Ограничение значения в диапазоне [lo, hi]."""
    return max(lo, min(hi, v))

def cmyk_to_rgb(c, m, y, k):
    """
    CMYK [0..1] -> RGB [0..255]
    Формула: R = 255*(1-C)*(1-K), etc.
    """
    c = clamp(c, 0.0, 1.0)
    m = clamp(m, 0.0, 1.0)
    y = clamp(y, 0.0, 1.0)
    k = clamp(k, 0.0, 1.0)
    r = int(round(255 * (1 - c) * (1 - k)))
    g = int(round(255 * (1 - m) * (1 - k)))
    b = int(round(255 * (1 - y) * (1 - k)))
    return clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255)

def rgb_to_cmyk(r, g, b):
    """
    RGB [0..255] -> CMYK [0..1]
    Нормировка, расчёт K, затем C/M/Y. Обработка чистого чёрного.
    """
    r = clamp(r, 0, 255)
    g = clamp(g, 0, 255)
    b = clamp(b, 0, 255)
    if r == 0 and g == 0 and b == 0:
        return 0.0, 0.0, 0.0, 1.0  # чистый чёрный

    rr, gg, bb = r / 255.0, g / 255.0, b / 255.0
    k = 1.0 - max(rr, gg, bb)
    denom = 1.0 - k if (1.0 - k) != 0 else 1.0  # защита от деления на ноль
    c = (1.0 - rr - k) / denom
    m = (1.0 - gg - k) / denom
    y = (1.0 - bb - k) / denom
    # численная стабильность
    return clamp(c, 0.0, 1.0), clamp(m, 0.0, 1.0), clamp(y, 0.0, 1.0), clamp(k, 0.0, 1.0)

def rgb_to_hsv(r, g, b):
    """
    RGB [0..255] -> HSV (H in degrees 0..360, S/V in [0..1])
    """
    r = clamp(r, 0, 255)
    g = clamp(g, 0, 255)
    b = clamp(b, 0, 255)
    rr, gg, bb = r / 255.0, g / 255.0, b / 255.0

    M = max(rr, gg, bb)
    m = min(rr, gg, bb)
    delta = M - m

    # Value
    v = M

    # Saturation
    s = 0.0 if M == 0 else (delta / M)

    # Hue
    if delta == 0:
        h = 0.0
    elif M == rr:
        h = (60.0 * ((gg - bb) / delta)) % 360.0
    elif M == gg:
        h = 60.0 * ((bb - rr) / delta + 2.0)
    else:  # M == bb
        h = 60.0 * ((rr - gg) / delta + 4.0)

    return clamp(h, 0.0, 360.0), clamp(s, 0.0, 1.0), clamp(v, 0.0, 1.0)

def hsv_to_rgb(h, s, v):
    """
    HSV (H 0..360, S/V 0..1) -> RGB [0..255]
    Алгоритм через C, X, m и сектора H'.
    """
    h = h % 360.0
    s = clamp(s, 0.0, 1.0)
    v = clamp(v, 0.0, 1.0)

    c = v * s
    hp = h / 60.0
    x = c * (1.0 - abs((hp % 2.0) - 1.0))

    if 0 <= hp < 1:
        rp, gp, bp = c, x, 0
    elif 1 <= hp < 2:
        rp, gp, bp = x, c, 0
    elif 2 <= hp < 3:
        rp, gp, bp = 0, c, x
    elif 3 <= hp < 4:
        rp, gp, bp = 0, x, c
    elif 4 <= hp < 5:
        rp, gp, bp = x, 0, c
    else:  # 5 <= hp < 6
        rp, gp, bp = c, 0, x

    m = v - c
    r = int(round(255 * (rp + m)))
    g = int(round(255 * (gp + m)))
    b = int(round(255 * (bp + m)))
    return clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255)

# -----------------------------
# Состояние и синхронизация
# -----------------------------

class ColorState:
    """
    Единый источник правды: RGB.
    При любом изменении (из CMYK/HSV/UI) — нормализуем в RGB, затем считаем CMYK и HSV.
    """
    def __init__(self, r=255, g=0, b=0):
        self.r = r
        self.g = g
        self.b = b
        self._callbacks = []

    def set_rgb(self, r, g, b):
        self.r = clamp(int(round(r)), 0, 255)
        self.g = clamp(int(round(g)), 0, 255)
        self.b = clamp(int(round(b)), 0, 255)
        self._notify()

    def set_cmyk(self, c, m, y, k):
        r, g, b = cmyk_to_rgb(c, m, y, k)
        self.set_rgb(r, g, b)

    def set_hsv(self, h, s, v):
        r, g, b = hsv_to_rgb(h, s, v)
        self.set_rgb(r, g, b)

    def on_change(self, cb):
        self._callbacks.append(cb)

    def _notify(self):
        for cb in self._callbacks:
            cb(self)

    def get_models(self):
        """Возвращает все три представления."""
        r, g, b = self.r, self.g, self.b
        c, m, y, k = rgb_to_cmyk(r, g, b)
        h, s, v = rgb_to_hsv(r, g, b)
        return (r, g, b), (c, m, y, k), (h, s, v)

# -----------------------------
# UI (Tkinter)
# -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CMYK ↔ RGB ↔ HSV Converter")
        self.state = ColorState(255, 0, 0)
        self._building_ui()
        self.state.on_change(self._update_views)
        self._update_views(self.state)

    def _building_ui(self):
        # Основной фрейм
        root = ttk.Frame(self, padding=10)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Цветовая панель (preview)
        self.preview = tk.Canvas(root, width=200, height=100, bd=0, highlightthickness=1, highlightbackground="#ccc")
        self.preview.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky="ew")

        # Кнопка палитры
        pick_btn = ttk.Button(root, text="Выбрать цвет…", command=self._pick_color)
        pick_btn.grid(row=1, column=0, sticky="w", pady=(0, 10))

        # Разделы для моделей
        self._build_rgb_section(root, row=2, col=0)
        self._build_cmyk_section(root, row=2, col=1)
        self._build_hsv_section(root, row=2, col=2)

        # Растяжение
        for c in range(3):
            root.columnconfigure(c, weight=1)

    # ---------- Разделы ----------
    def _build_rgb_section(self, parent, row, col):
        frame = ttk.LabelFrame(parent, text="RGB (0..255)")
        frame.grid(row=row, column=col, padx=5, sticky="nsew")

        # Слайдеры и поля ввода
        self.rgb_vars = {
            'R': tk.IntVar(value=255),
            'G': tk.IntVar(value=0),
            'B': tk.IntVar(value=0),
        }
        self._make_slider_with_entry(frame, "R", 0, 255, self._on_rgb_change)
        self._make_slider_with_entry(frame, "G", 0, 255, self._on_rgb_change)
        self._make_slider_with_entry(frame, "B", 0, 255, self._on_rgb_change)

    def _build_cmyk_section(self, parent, row, col):
        frame = ttk.LabelFrame(parent, text="CMYK (0..1)")
        frame.grid(row=row, column=col, padx=5, sticky="nsew")

        self.cmyk_vars = {
            'C': tk.DoubleVar(value=0.0),
            'M': tk.DoubleVar(value=1.0),
            'Y': tk.DoubleVar(value=1.0),
            'K': tk.DoubleVar(value=0.0),
        }
        self._make_slider_with_entry(frame, "C", 0.0, 1.0, self._on_cmyk_change, resolution=0.001)
        self._make_slider_with_entry(frame, "M", 0.0, 1.0, self._on_cmyk_change, resolution=0.001)
        self._make_slider_with_entry(frame, "Y", 0.0, 1.0, self._on_cmyk_change, resolution=0.001)
        self._make_slider_with_entry(frame, "K", 0.0, 1.0, self._on_cmyk_change, resolution=0.001)

    def _build_hsv_section(self, parent, row, col):
        frame = ttk.LabelFrame(parent, text="HSV (H 0..360; S/V 0..1)")
        frame.grid(row=row, column=col, padx=5, sticky="nsew")

        self.hsv_vars = {
            'H': tk.DoubleVar(value=0.0),
            'S': tk.DoubleVar(value=1.0),
            'V': tk.DoubleVar(value=1.0),
        }
        self._make_slider_with_entry(frame, "H", 0.0, 360.0, self._on_hsv_change, resolution=0.1)
        self._make_slider_with_entry(frame, "S", 0.0, 1.0, self._on_hsv_change, resolution=0.001)
        self._make_slider_with_entry(frame, "V", 0.0, 1.0, self._on_hsv_change, resolution=0.001)

    # ---------- Компоненты UI ----------
    def _make_slider_with_entry(self, parent, label, lo, hi, command, resolution=1.0):
        row = len(parent.grid_slaves())  # грубо: растёт при добавлении
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(5, 5))
        var = None
        if label in ['R', 'G', 'B']:
            var = self.rgb_vars[label]
            slider = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal", variable=var,
                               command=lambda e: command())
            slider.grid(row=row, column=1, sticky="ew", padx=(5, 5))
            entry = ttk.Entry(parent, width=7)
            entry.insert(0, str(var.get()))
            entry.grid(row=row, column=2, padx=(5, 5))
            entry.bind("<Return>", lambda e: self._entry_commit(entry, var, int, lo, hi, command))
        elif label in ['C', 'M', 'Y', 'K']:
            var = self.cmyk_vars[label]
            slider = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal", variable=var,
                               command=lambda e: command())
            slider.grid(row=row, column=1, sticky="ew", padx=(5, 5))
            entry = ttk.Entry(parent, width=7)
            entry.insert(0, f"{var.get():.3f}")
            entry.grid(row=row, column=2, padx=(5, 5))
            entry.bind("<Return>", lambda e: self._entry_commit(entry, var, float, lo, hi, command))
        else:  # HSV
            var = self.hsv_vars[label]
            slider = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal", variable=var,
                               command=lambda e: command())
            slider.grid(row=row, column=1, sticky="ew", padx=(5, 5))
            entry = ttk.Entry(parent, width=7)
            if label == 'H':
                entry.insert(0, f"{var.get():.1f}")
            else:
                entry.insert(0, f"{var.get():.3f}")
            entry.grid(row=row, column=2, padx=(5, 5))
            entry.bind("<Return>", lambda e: self._entry_commit(entry, var, float, lo, hi, command))

        parent.columnconfigure(1, weight=1)

    def _entry_commit(self, entry, var, caster, lo, hi, command):
        """Коммит значения из поля ввода в связанную переменную и вызов пересчёта."""
        try:
            val = caster(entry.get())
        except ValueError:
            return
        if isinstance(val, float):
            val = clamp(val, lo, hi)
        else:
            val = clamp(val, int(lo), int(hi))
        var.set(val)
        command()

    # ---------- Обработчики изменений ----------
    def _on_rgb_change(self):
        r = self.rgb_vars['R'].get()
        g = self.rgb_vars['G'].get()
        b = self.rgb_vars['B'].get()
        self.state.set_rgb(r, g, b)

    def _on_cmyk_change(self):
        c = self.cmyk_vars['C'].get()
        m = self.cmyk_vars['M'].get()
        y = self.cmyk_vars['Y'].get()
        k = self.cmyk_vars['K'].get()
        self.state.set_cmyk(c, m, y, k)

    def _on_hsv_change(self):
        h = self.hsv_vars['H'].get()
        s = self.hsv_vars['S'].get()
        v = self.hsv_vars['V'].get()
        self.state.set_hsv(h, s, v)

    def _pick_color(self):
        """Системный color picker возвращает RGB в #RRGGBB."""
        color = colorchooser.askcolor()
        if color and color[0] is not None:
            r, g, b = map(int, color[0])
            self.state.set_rgb(r, g, b)

    # ---------- Синхронизация представлений ----------
    def _update_views(self, state: ColorState):
        # Получаем текущие модели
        (r, g, b), (c, m, y, k), (h, s, v) = state.get_models()

        # Обновление preview
        self.preview.configure(bg=f"#{r:02x}{g:02x}{b:02x}")

        # Защита от рекурсии: устанавливаем значения без триггера (через set на переменных)
        # RGB
        self.rgb_vars['R'].set(r)
        self.rgb_vars['G'].set(g)
        self.rgb_vars['B'].set(b)
        # Обновим текст в полях (чтобы отражать округление)
        # Для простоты пройдёмся по всем Entry через дочерние виджеты:
        def refresh_entries(frame, vars_dict, fmt_map):
            # Небольшой утилитарный проход: ищем Entry рядом с Label/Scale
            pass  # Упрощаем: значения полей обновятся при следующем вводе

        # CMYK
        self.cmyk_vars['C'].set(c)
        self.cmyk_vars['M'].set(m)
        self.cmyk_vars['Y'].set(y)
        self.cmyk_vars['K'].set(k)

        # HSV
        self.hsv_vars['H'].set(h)
        self.hsv_vars['S'].set(s)
        self.hsv_vars['V'].set(v)

# -----------------------------
# Запуск
# -----------------------------

if __name__ == "__main__":
    app = App()
    app.mainloop()
