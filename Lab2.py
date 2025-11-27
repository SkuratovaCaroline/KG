#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Processing App: Morphological operations, nonlinear rank filters, and compression analysis.
Single-file Tkinter GUI. Requires: Pillow, numpy. Optional: scipy (faster rank filters).
Author: You
"""

import os
import io
import sys
import math
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np

# Try to import SciPy for faster filters; fallback to pure NumPy if unavailable.
_SCIPY_AVAILABLE = True
try:
    from scipy.ndimage import grey_erosion, grey_dilation
except Exception:
    _SCIPY_AVAILABLE = False

# ---------------------------
# Utility: Structuring Elements
# ---------------------------

def make_structuring_element(shape: str, size: int, angle_deg: int = 0):
    """
    Create a binary structuring element as a NumPy array with values {0,1}.
    shape: 'square', 'disk', 'line_h', 'line_v', 'line_angle'
    size: odd integer for symmetry; will be coerced to odd >=1
    angle_deg: only used for line_angle
    """
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    if shape == 'square':
        return np.ones((size, size), dtype=np.uint8)
    elif shape == 'disk':
        r = size // 2
        y, x = np.ogrid[-r:r+1, -r:r+1]
        mask = (x*x + y*y) <= r*r
        return mask.astype(np.uint8)
    elif shape == 'line_h':
        se = np.zeros((size, size), dtype=np.uint8)
        se[size//2, :] = 1
        return se
    elif shape == 'line_v':
        se = np.zeros((size, size), dtype=np.uint8)
        se[:, size//2] = 1
        return se
    elif shape == 'line_angle':
        # Bresenham-like line through center at angle
        se = np.zeros((size, size), dtype=np.uint8)
        cx, cy = size//2, size//2
        length = size
        theta = math.radians(angle_deg % 360)
        # Parametric points along line
        for t in np.linspace(-length/2, length/2, num=length*3):
            x = int(round(cx + t * math.cos(theta)))
            y = int(round(cy - t * math.sin(theta)))
            if 0 <= x < size and 0 <= y < size:
                se[y, x] = 1
        return se
    else:
        return np.ones((size, size), dtype=np.uint8)

# ---------------------------
# Morphology operations
# ---------------------------

def morph_erosion(img_gray: np.ndarray, se: np.ndarray):
    if _SCIPY_AVAILABLE:
        return grey_erosion(img_gray, footprint=se)
    # Fallback: sliding-window min
    return sliding_rank(img_gray, se, mode='min')

def morph_dilation(img_gray: np.ndarray, se: np.ndarray):
    if _SCIPY_AVAILABLE:
        return grey_dilation(img_gray, footprint=se)
    # Fallback: sliding-window max
    return sliding_rank(img_gray, se, mode='max')

def morph_opening(img_gray: np.ndarray, se: np.ndarray):
    return morph_dilation(morph_erosion(img_gray, se), se)

def morph_closing(img_gray: np.ndarray, se: np.ndarray):
    return morph_erosion(morph_dilation(img_gray, se), se)

# ---------------------------
# Rank filters (order statistics)
# ---------------------------

def sliding_rank(img_gray: np.ndarray, se: np.ndarray, mode='median', percentile=50):
    """
    Pure NumPy sliding window order-statistic filter respecting the footprint se.
    mode: 'median', 'min', 'max', 'percentile'
    percentile: used if mode == 'percentile' (0..100)
    """
    assert img_gray.ndim == 2, "Expect grayscale image"
    h, w = img_gray.shape
    se = (se > 0).astype(np.uint8)
    kh, kw = se.shape
    ph, pw = kh//2, kw//2

    # Pad image to handle borders
    pad_val = int(np.median(img_gray))
    padded = np.pad(img_gray, ((ph, ph), (pw, pw)), mode='constant', constant_values=pad_val)

    out = np.empty_like(img_gray)
    # Precompute indices of SE footprint
    coords = np.argwhere(se)
    # Iterate over pixels
    # Note: pure Python loops are slower; acceptable for moderate sizes. For large images, prefer SciPy.
    for y in range(h):
        y0 = y
        for x in range(w):
            x0 = x
            # extract values under footprint
            vals = []
            for (dy, dx) in coords:
                yy = y0 + dy
                xx = x0 + dx
                vals.append(padded[yy, xx])
            if mode == 'median':
                out[y, x] = np.median(vals)
            elif mode == 'min':
                out[y, x] = np.min(vals)
            elif mode == 'max':
                out[y, x] = np.max(vals)
            elif mode == 'percentile':
                p = float(percentile)
                p = min(100.0, max(0.0, p))
                out[y, x] = np.percentile(vals, p)
            else:
                out[y, x] = np.median(vals)
    return out.astype(img_gray.dtype)

def rank_median(img_gray: np.ndarray, se: np.ndarray):
    return sliding_rank(img_gray, se, mode='median')

def rank_min(img_gray: np.ndarray, se: np.ndarray):
    return sliding_rank(img_gray, se, mode='min')

def rank_max(img_gray: np.ndarray, se: np.ndarray):
    return sliding_rank(img_gray, se, mode='max')

def rank_percentile(img_gray: np.ndarray, se: np.ndarray, p: float):
    return sliding_rank(img_gray, se, mode='percentile', percentile=p)

# ---------------------------
# Image compression utilities
# ---------------------------

def bytes_for_png(pil_img: Image.Image):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG', optimize=True)
    return buf.getvalue()

def bytes_for_jpeg(pil_img: Image.Image, quality: int = 75):
    buf = io.BytesIO()
    # Convert to RGB to ensure JPEG compatibility
    img_rgb = pil_img.convert('RGB')
    img_rgb.save(buf, format='JPEG', quality=quality, optimize=True)
    return buf.getvalue()

def compression_ratio(original_bytes: bytes, compressed_bytes: bytes):
    if not original_bytes or not compressed_bytes:
        return 0.0
    return len(original_bytes) / len(compressed_bytes)

# ---------------------------
# Conversions
# ---------------------------

def pil_to_gray_np(img: Image.Image) -> np.ndarray:
    g = ImageOps.grayscale(img)
    arr = np.array(g)
    # normalize dtype to uint8 if needed
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr

def np_to_pil_gray(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L')

def ensure_odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1

# ---------------------------
# GUI Application
# ---------------------------

class ImageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing: Morphology, Rank Filters, Compression Analysis")
        self.geometry("1100x700")
        self.minsize(1000, 600)

        # State
        self.folder = None
        self.image_paths = []
        self.current_index = -1
        self.current_img = None  # PIL
        self.processed_img = None  # PIL

        # UI
        self._build_ui()

    def _build_ui(self):
        # Main layout: left controls, right previews
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        ctrl_frame = ttk.Frame(self, padding=8)
        ctrl_frame.grid(row=0, column=0, sticky="ns")
        ctrl_frame.columnconfigure(0, weight=1)

        view_frame = ttk.Frame(self, padding=8)
        view_frame.grid(row=0, column=1, sticky="nsew")
        view_frame.columnconfigure(0, weight=1)
        view_frame.rowconfigure(1, weight=1)

        # Controls
        ttk.Label(ctrl_frame, text="Dataset and navigation", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0,4))
        btn_folder = ttk.Button(ctrl_frame, text="Select folder...", command=self.select_folder)
        btn_folder.grid(row=1, column=0, sticky="ew", pady=2)

        self.lbl_count = ttk.Label(ctrl_frame, text="No folder selected")
        self.lbl_count.grid(row=2, column=0, sticky="w", pady=2)

        nav_frame = ttk.Frame(ctrl_frame)
        nav_frame.grid(row=3, column=0, sticky="ew", pady=4)
        btn_prev = ttk.Button(nav_frame, text="◀ Prev", command=self.prev_image)
        btn_prev.grid(row=0, column=0, padx=2)
        btn_next = ttk.Button(nav_frame, text="Next ▶", command=self.next_image)
        btn_next.grid(row=0, column=1, padx=2)

        ttk.Separator(ctrl_frame).grid(row=4, column=0, sticky="ew", pady=8)

        ttk.Label(ctrl_frame, text="Structuring element", font=("TkDefaultFont", 10, "bold")).grid(row=5, column=0, sticky="w", pady=(0,4))
        self.se_shape = tk.StringVar(value='disk')
        ttk.Label(ctrl_frame, text="Shape").grid(row=6, column=0, sticky="w")
        cb_shape = ttk.Combobox(ctrl_frame, textvariable=self.se_shape, values=['disk','square','line_h','line_v','line_angle'], state='readonly')
        cb_shape.grid(row=7, column=0, sticky="ew", pady=2)

        self.se_size = tk.IntVar(value=5)
        ttk.Label(ctrl_frame, text="Size (odd)").grid(row=8, column=0, sticky="w")
        sp_size = ttk.Spinbox(ctrl_frame, from_=1, to=99, increment=2, textvariable=self.se_size)
        sp_size.grid(row=9, column=0, sticky="ew", pady=2)

        self.se_angle = tk.IntVar(value=0)
        ttk.Label(ctrl_frame, text="Angle (line_angle)").grid(row=10, column=0, sticky="w")
        sp_angle = ttk.Spinbox(ctrl_frame, from_=0, to=359, textvariable=self.se_angle)
        sp_angle.grid(row=11, column=0, sticky="ew", pady=2)

        ttk.Separator(ctrl_frame).grid(row=12, column=0, sticky="ew", pady=8)

        ttk.Label(ctrl_frame, text="Processing", font=("TkDefaultFont", 10, "bold")).grid(row=13, column=0, sticky="w", pady=(0,4))
        self.proc_type = tk.StringVar(value='morph_opening')
        ttk.Label(ctrl_frame, text="Method").grid(row=14, column=0, sticky="w")
        cb_proc = ttk.Combobox(
            ctrl_frame,
            textvariable=self.proc_type,
            values=[
                'morph_erosion','morph_dilation','morph_opening','morph_closing',
                'rank_median','rank_min','rank_max','rank_percentile'
            ],
            state='readonly'
        )
        cb_proc.grid(row=15, column=0, sticky="ew", pady=2)

        self.percentile = tk.IntVar(value=50)
        ttk.Label(ctrl_frame, text="Percentile (rank_percentile)").grid(row=16, column=0, sticky="w")
        sp_pct = ttk.Spinbox(ctrl_frame, from_=0, to=100, textvariable=self.percentile)
        sp_pct.grid(row=17, column=0, sticky="ew", pady=2)

        btn_apply = ttk.Button(ctrl_frame, text="Apply to current image", command=self.apply_processing)
        btn_apply.grid(row=18, column=0, sticky="ew", pady=6)

        btn_save = ttk.Button(ctrl_frame, text="Save processed image as...", command=self.save_processed)
        btn_save.grid(row=19, column=0, sticky="ew", pady=2)

        ttk.Separator(ctrl_frame).grid(row=20, column=0, sticky="ew", pady=8)

        ttk.Label(ctrl_frame, text="Compression analysis", font=("TkDefaultFont", 10, "bold")).grid(row=21, column=0, sticky="w", pady=(0,4))
        self.jpeg_quality = tk.IntVar(value=75)
        ttk.Label(ctrl_frame, text="JPEG quality").grid(row=22, column=0, sticky="w")
        sp_q = ttk.Spinbox(ctrl_frame, from_=10, to=95, increment=5, textvariable=self.jpeg_quality)
        sp_q.grid(row=23, column=0, sticky="ew", pady=2)

        btn_analyze = ttk.Button(ctrl_frame, text="Analyze compression on dataset", command=self.analyze_compression)
        btn_analyze.grid(row=24, column=0, sticky="ew", pady=6)

        self.lbl_status = ttk.Label(ctrl_frame, text="", foreground="#555")
        self.lbl_status.grid(row=25, column=0, sticky="w", pady=4)

        # Previews
        ttk.Label(view_frame, text="Original vs Processed", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0,6))
        canvas_frame = ttk.Frame(view_frame)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.columnconfigure(1, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas_orig = tk.Canvas(canvas_frame, bg="#222")
        self.canvas_proc = tk.Canvas(canvas_frame, bg="#222")
        self.canvas_orig.grid(row=0, column=0, sticky="nsew", padx=(0,4))
        self.canvas_proc.grid(row=0, column=1, sticky="nsew", padx=(4,0))

        # Image refs to avoid GC
        self.tkimg_orig = None
        self.tkimg_proc = None

    # -----------------------
    # Dataset management
    # -----------------------

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select dataset folder")
        if not folder:
            return
        self.folder = folder
        self.image_paths = self._scan_images(folder)
        self.current_index = -1
        self.update_count_label()
        if self.image_paths:
            self.next_image()
        else:
            self.show_status("No images found in the selected folder.")

    def _scan_images(self, folder):
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        paths = []
        for root, _, files in os.walk(folder):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in exts:
                    paths.append(os.path.join(root, f))
        paths.sort()
        return paths

    def update_count_label(self):
        if not self.folder:
            self.lbl_count.config(text="No folder selected")
        else:
            self.lbl_count.config(text=f"Folder: {self.folder}\nImages found: {len(self.image_paths)}")

    def prev_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.load_current_image()

    def next_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.load_current_image()

    def load_current_image(self):
        try:
            path = self.image_paths[self.current_index]
            img = Image.open(path)
            self.current_img = img
            self.processed_img = None
            self.draw_images()
            self.show_status(f"Loaded: {path}")
        except Exception as e:
            self.show_error("Failed to load image", e)

    # -----------------------
    # Processing
    # -----------------------

    def apply_processing(self):
        if self.current_img is None:
            self.show_status("No image loaded.")
            return
        try:
            size = ensure_odd(self.se_size.get())
            shape = self.se_shape.get().strip()
            angle = int(self.se_angle.get())
            se = make_structuring_element(shape, size, angle)

            arr = pil_to_gray_np(self.current_img)
            method = self.proc_type.get()

            if method == 'morph_erosion':
                out = morph_erosion(arr, se)
            elif method == 'morph_dilation':
                out = morph_dilation(arr, se)
            elif method == 'morph_opening':
                out = morph_opening(arr, se)
            elif method == 'morph_closing':
                out = morph_closing(arr, se)
            elif method == 'rank_median':
                out = rank_median(arr, se)
            elif method == 'rank_min':
                out = rank_min(arr, se)
            elif method == 'rank_max':
                out = rank_max(arr, se)
            elif method == 'rank_percentile':
                p = int(self.percentile.get())
                out = rank_percentile(arr, se, p)
            else:
                out = arr

            self.processed_img = np_to_pil_gray(out)
            self.draw_images()
            self.show_status(f"Applied: {method} | SE: {shape} size={size} angle={angle}")
        except Exception as e:
            self.show_error("Processing failed", e)

    def save_processed(self):
        if self.processed_img is None:
            self.show_status("Nothing to save. Apply processing first.")
            return
        try:
            fpath = filedialog.asksaveasfilename(
                title="Save processed image",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("TIFF", "*.tif;*.tiff"), ("BMP", "*.bmp")]
            )
            if not fpath:
                return
            ext = os.path.splitext(fpath)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                self.processed_img.convert('RGB').save(fpath, format='JPEG', quality=90)
            elif ext in ('.tif', '.tiff'):
                self.processed_img.save(fpath, format='TIFF')
            elif ext == '.bmp':
                self.processed_img.save(fpath, format='BMP')
            else:
                self.processed_img.save(fpath, format='PNG', optimize=True)
            self.show_status(f"Saved: {fpath}")
        except Exception as e:
            self.show_error("Save failed", e)

    # -----------------------
    # Compression analysis
    # -----------------------

    def analyze_compression(self):
        if not self.image_paths:
            self.show_status("No dataset loaded.")
            return
        try:
            q = int(self.jpeg_quality.get())
            best = None  # (ratio, path, format)
            worst = None

            for path in self.image_paths:
                try:
                    img = Image.open(path)
                    # Original bytes (PNG lossless snapshot for baseline)
                    orig_bytes = bytes_for_png(img)
                    # Compressed variants
                    png_bytes = bytes_for_png(img)
                    jpeg_bytes = bytes_for_jpeg(img, quality=q)

                    # Ratios (original_size / compressed_size) -> higher is better compression
                    ratio_png = compression_ratio(orig_bytes, png_bytes)
                    ratio_jpeg = compression_ratio(orig_bytes, jpeg_bytes)

                    # Track best/worst over both formats
                    for fmt, ratio in [('PNG', ratio_png), ('JPEG', ratio_jpeg)]:
                        if not best or ratio > best[0]:
                            best = (ratio, path, fmt)
                        if not worst or ratio < worst[0]:
                            worst = (ratio, path, fmt)
                except Exception as inner_e:
                    # Continue on errors per image
                    print(f"Compression error for {path}: {inner_e}", file=sys.stderr)

            msg = (
                f"Compression analysis (JPEG quality={q}):\n"
                f"Best:  ratio={best[0]:.3f} | {best[2]} | {best[1]}\n"
                f"Worst: ratio={worst[0]:.3f} | {worst[2]} | {worst[1]}"
            )
            messagebox.showinfo("Compression results", msg)
            self.show_status("Compression analysis completed.")
        except Exception as e:
            self.show_error("Compression analysis failed", e)

    # -----------------------
    # Rendering
    # -----------------------

    def draw_images(self):
        self._draw_on_canvas(self.canvas_orig, self.current_img, which='orig')
        self._draw_on_canvas(self.canvas_proc, self.processed_img, which='proc')

    def _draw_on_canvas(self, canvas: tk.Canvas, img: Image.Image, which='orig'):
        canvas.delete("all")
        if img is None:
            canvas.create_text(
                canvas.winfo_width()//2 or 300,
                canvas.winfo_height()//2 or 200,
                text="No image",
                fill="#ccc"
            )
            if which == 'orig':
                self.tkimg_orig = None
            else:
                self.tkimg_proc = None
            return
        # Fit image to canvas while preserving aspect
        cw = canvas.winfo_width() or 500
        ch = canvas.winfo_height() or 400
        im_w, im_h = img.size
        scale = min(cw / im_w, ch / im_h)
        scale = max(0.05, min(scale, 1.0))  # Avoid huge upscales
        new_w = max(1, int(im_w * scale))
        new_h = max(1, int(im_h * scale))
        disp = img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
        tk_img = ImageTk.PhotoImage(disp)
        canvas.create_image(cw//2, ch//2, image=tk_img, anchor='center')
        if which == 'orig':
            self.tkimg_orig = tk_img
        else:
            self.tkimg_proc = tk_img

    # -----------------------
    # Feedback helpers
    # -----------------------

    def show_status(self, msg: str):
        self.lbl_status.config(text=msg)

    def show_error(self, title: str, e: Exception):
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        self.lbl_status.config(text=f"{title}: {err}")
        messagebox.showerror(title, f"{title}:\n{err}")

# ---------------------------
# Main
# ---------------------------

def main():
    app = ImageApp()
    # Ensure canvases redraw on resize
    def on_resize(event):
        app.draw_images()
    app.canvas_orig.bind("<Configure>", on_resize)
    app.canvas_proc.bind("<Configure>", on_resize)
    app.mainloop()

if __name__ == "__main__":
    main()
