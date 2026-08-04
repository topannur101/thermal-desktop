# -*- coding: utf-8 -*-
import sys
import os
import tkinter as tk
from tkinter import ttk
from collections import deque
import cv2
import numpy as np

# Cross-platform font
_FONT = "Ubuntu" if sys.platform.startswith("linux") else "Segoe UI"
from PIL import Image, ImageTk

from camera import ThermalCamera

try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    FilterGraph = None

from lib import (
    Mini2, PseudoColor, SceneMode, FlipMode,
)

# ── Color Palette ──────────────────────────────────────────────────────────────
BG_DARK       = "#0f1117"
BG_PANEL      = "#1a1d27"
BG_CARD       = "#22273a"
BG_HOVER      = "#2d3250"
ACCENT        = "#4f8ef7"
TEXT_PRIMARY  = "#e2e8f0"
TEXT_MUTED    = "#64748b"
TEXT_RED      = "#f87171"
TEXT_GREEN    = "#4ade80"
TEXT_BLUE     = "#60a5fa"
BORDER        = "#2d3250"
SLIDER_TROUGH = "#2d3250"


def _get_cameras():
    if FilterGraph is not None:
        graph = FilterGraph()
        devices = graph.get_input_devices()
        return devices if devices else ["No cameras found"]
    return [f"Camera {i}" for i in range(10)]


def _resolve_index(camera_str, selection_index):
    if FilterGraph is None:
        try:
            return int(camera_str.replace("Camera ", ""))
        except ValueError:
            return 0
    return selection_index


# ── Styled Widgets ─────────────────────────────────────────────────────────────

class SectionLabel(tk.Label):
    def __init__(self, parent, text, **kw):
        super().__init__(
            parent, text=text,
            bg=BG_PANEL, fg=ACCENT,
            font=(_FONT, 8, "bold"),
            anchor="w", **kw
        )


class DarkLabel(tk.Label):
    def __init__(self, parent, **kw):
        kw.setdefault("bg", BG_PANEL)
        kw.setdefault("fg", TEXT_PRIMARY)
        kw.setdefault("font", (_FONT, 9))
        super().__init__(parent, **kw)


class DarkButton(tk.Button):
    def __init__(self, parent, **kw):
        self._base_bg = kw.pop("base_bg", BG_HOVER)
        kw.setdefault("bg", self._base_bg)
        kw.setdefault("fg", TEXT_PRIMARY)
        kw.setdefault("activebackground", ACCENT)
        kw.setdefault("activeforeground", "#ffffff")
        kw.setdefault("relief", "flat")
        kw.setdefault("bd", 0)
        kw.setdefault("cursor", "hand2")
        kw.setdefault("font", (_FONT, 9))
        kw.setdefault("padx", 10)
        kw.setdefault("pady", 5)
        super().__init__(parent, **kw)
        self.bind("<Enter>", lambda e: self.config(bg=ACCENT))
        self.bind("<Leave>", lambda e: self.config(bg=self._base_bg))


class AccentButton(tk.Button):
    def __init__(self, parent, **kw):
        kw.setdefault("bg", ACCENT)
        kw.setdefault("fg", "#ffffff")
        kw.setdefault("activebackground", "#3b75e8")
        kw.setdefault("activeforeground", "#ffffff")
        kw.setdefault("relief", "flat")
        kw.setdefault("bd", 0)
        kw.setdefault("cursor", "hand2")
        kw.setdefault("font", (_FONT, 9, "bold"))
        kw.setdefault("padx", 12)
        kw.setdefault("pady", 6)
        super().__init__(parent, **kw)
        self.bind("<Enter>", lambda e: self.config(bg="#3b75e8"))
        self.bind("<Leave>", lambda e: self.config(bg=ACCENT))


class SliderRow(tk.Frame):
    """Labelled horizontal slider with live value readout."""
    def __init__(self, parent, label, from_=0, to=100, command=None, default=50, **kw):
        super().__init__(parent, bg=BG_PANEL, **kw)
        self._cmd = command
        self.var = tk.IntVar(value=default)

        tk.Label(self, text=label, width=20, anchor="w",
                 bg=BG_PANEL, fg=TEXT_PRIMARY, font=(_FONT, 9)
                 ).pack(side=tk.LEFT)

        self.slider = tk.Scale(
            self, from_=from_, to=to, orient=tk.HORIZONTAL,
            variable=self.var,
            bg=BG_PANEL, fg=TEXT_PRIMARY,
            troughcolor=SLIDER_TROUGH,
            highlightthickness=0, bd=0,
            activebackground=ACCENT,
            sliderrelief="flat",
            length=140,
            command=self._on_change,
            showvalue=False,
        )
        self.slider.pack(side=tk.LEFT, padx=(4, 4))

        tk.Label(
            self, textvariable=self.var, width=4,
            bg=BG_PANEL, fg=ACCENT,
            font=(_FONT, 9, "bold")
        ).pack(side=tk.LEFT)

    def _on_change(self, val):
        if self._cmd:
            self._cmd(int(val))

    def set(self, value):
        self.var.set(value)


class ToggleButton(tk.Button):
    """Two-state toggle button."""
    def __init__(self, parent, label_off, label_on, command=None, **kw):
        self._state = False
        self._label_off = label_off
        self._label_on = label_on
        self._ext_cmd = command
        super().__init__(
            parent,
            text=label_off,
            bg=BG_HOVER, fg=TEXT_MUTED,
            activebackground=TEXT_GREEN, activeforeground="#000",
            relief="flat", bd=0,
            cursor="hand2",
            font=(_FONT, 9),
            padx=10, pady=5,
            command=self._toggle,
            **kw
        )

    def _toggle(self):
        self._state = not self._state
        if self._state:
            self.config(bg=TEXT_GREEN, fg="#000", text=self._label_on)
        else:
            self.config(bg=BG_HOVER, fg=TEXT_MUTED, text=self._label_off)
        if self._ext_cmd:
            self._ext_cmd(1 if self._state else 0)


# ── Main Application ────────────────────────────────────────────────────────────

class ThermalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hdaniee Thermal Viewer")
        self.root.geometry("1120x680")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        self.is_running = True
        self.camera = None
        self.mini2 = None

        # Rolling temperature averages (5-frame window) for stable readings
        self._hist_max = deque(maxlen=5)
        self._hist_min = deque(maxlen=5)
        self._hist_ctr = deque(maxlen=5)

        self._init_mini2()
        self._build_ui()
        self._refresh_cameras()
        self._auto_connect()
        self._update_video()

    # ── Mini2 Init ────────────────────────────────────────────────────────────

    def _init_mini2(self):
        try:
            self.mini2 = Mini2()
            self._mini2_ok = True
        except Exception:
            self.mini2 = None
            self._mini2_ok = False

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        title_bar = tk.Frame(self.root, bg=BG_PANEL, height=44)
        title_bar.pack(fill=tk.X, side=tk.TOP)
        title_bar.pack_propagate(False)

        tk.Label(
            title_bar,
            text="  Hdaniee Thermal Viewer",
            bg=BG_PANEL, fg=TEXT_PRIMARY,
            font=(_FONT, 13, "bold"),
        ).pack(side=tk.LEFT, padx=14, pady=10)

        mini2_color = TEXT_GREEN if self._mini2_ok else TEXT_MUTED
        mini2_text  = "  Mini2 Connected" if self._mini2_ok else "  Mini2 Not Connected"
        self.status_pill = tk.Label(
            title_bar, text=mini2_text,
            bg=BG_PANEL, fg=mini2_color,
            font=(_FONT, 9),
        )
        self.status_pill.pack(side=tk.RIGHT, padx=14, pady=10)

        # Body
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True)

        # ── Left: video pane ──────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 6), pady=12)

        # Camera selector row
        cam_row = tk.Frame(left, bg=BG_DARK)
        cam_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(cam_row, text="Camera:", bg=BG_DARK, fg=TEXT_MUTED,
                 font=(_FONT, 9)).pack(side=tk.LEFT)

        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(
            cam_row, textvariable=self.camera_var, state="readonly", width=28
        )
        self.camera_combo.pack(side=tk.LEFT, padx=(4, 6))

        DarkButton(cam_row, text="Refresh", command=self._refresh_cameras).pack(side=tk.LEFT, padx=2)
        AccentButton(cam_row, text="Connect", command=self._reconnect_camera).pack(side=tk.LEFT, padx=4)

        # Video canvas
        self.vid_border = tk.Frame(left, bg=BORDER, bd=1, relief="flat")
        self.vid_border.pack(fill=tk.BOTH, expand=True)
        self.vid_border.pack_propagate(False)
        self.video_label = tk.Label(self.vid_border, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Temperature bar
        temp_bar = tk.Frame(left, bg=BG_PANEL, height=40)
        temp_bar.pack(fill=tk.X, pady=(6, 0))
        temp_bar.pack_propagate(False)

        self.max_temp_var    = tk.StringVar(value="MAX   --.-C")
        self.center_temp_var = tk.StringVar(value="CTR   --.-C")
        self.min_temp_var    = tk.StringVar(value="MIN   --.-C")

        for var, color in [
            (self.max_temp_var,    TEXT_RED),
            (self.center_temp_var, TEXT_GREEN),
            (self.min_temp_var,    TEXT_BLUE),
        ]:
            tk.Label(
                temp_bar, textvariable=var,
                bg=BG_PANEL, fg=color,
                font=(_FONT, 12, "bold"),
            ).pack(side=tk.LEFT, padx=22, pady=8)

        # ── Right: sidebar ────────────────────────────────────────────────────
        sidebar = tk.Frame(body, bg=BG_PANEL, width=286)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12), pady=12)
        sidebar.pack_propagate(False)

        # ttk styles
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Dark.TNotebook",
                         background=BG_PANEL, borderwidth=0, tabmargins=0)
        style.configure("Dark.TNotebook.Tab",
                         background=BG_CARD, foreground=TEXT_MUTED,
                         font=(_FONT, 9), padding=[10, 5])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG_HOVER)],
                  foreground=[("selected", TEXT_PRIMARY)])

        style.configure("TCombobox",
                         fieldbackground=BG_CARD,
                         background=BG_CARD,
                         foreground=TEXT_PRIMARY,
                         selectbackground=ACCENT,
                         selectforeground="#fff",
                         arrowcolor=ACCENT)
        style.map("TCombobox",
                  fieldbackground=[("readonly", BG_CARD)],
                  foreground=[("readonly", TEXT_PRIMARY)])

        nb = ttk.Notebook(sidebar, style="Dark.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=8)

        self._build_image_tab(nb)
        self._build_settings_tab(nb)
        self._build_tools_tab(nb)

    # ── Tab: Image ────────────────────────────────────────────────────────────

    def _build_image_tab(self, nb):
        tab = tk.Frame(nb, bg=BG_PANEL)
        nb.add(tab, text="Image")

        # Pseudo-Color
        SectionLabel(tab, text="PSEUDO-COLOR PALETTE").pack(fill=tk.X, padx=10, pady=(14, 2))
        self.color_var = tk.StringVar(value="Ironbow")
        color_combo = ttk.Combobox(
            tab, textvariable=self.color_var,
            values=[p.name for p in PseudoColor], state="readonly"
        )
        color_combo.pack(fill=tk.X, padx=10, pady=(0, 4))
        color_combo.bind("<<ComboboxSelected>>", self._on_pseudo_color)

        # Color swatch
        self.color_preview = tk.Canvas(tab, height=8, bd=0, highlightthickness=0)
        self.color_preview.pack(fill=tk.X, padx=10, pady=(0, 14))
        self._update_color_preview("Ironbow")

        # Scene Mode
        SectionLabel(tab, text="SCENE MODE").pack(fill=tk.X, padx=10, pady=(0, 2))
        self.scene_var = tk.StringVar(value="GeneralMode")
        scene_combo = ttk.Combobox(
            tab, textvariable=self.scene_var,
            values=[s.name for s in SceneMode], state="readonly"
        )
        scene_combo.pack(fill=tk.X, padx=10, pady=(0, 14))
        scene_combo.bind("<<ComboboxSelected>>", self._on_scene)

        # Flip Mode
        SectionLabel(tab, text="FLIP MODE").pack(fill=tk.X, padx=10, pady=(0, 6))
        flip_frame = tk.Frame(tab, bg=BG_PANEL)
        flip_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        labels = {"No_Flip": "None", "X_Flip": "Flip X", "Y_Flip": "Flip Y", "XY_Flip": "Flip XY"}
        for flip in FlipMode:
            lbl = labels.get(flip.name, flip.name)
            btn = tk.Button(
                flip_frame, text=lbl,
                bg=BG_CARD, fg=TEXT_MUTED,
                relief="flat", bd=0, cursor="hand2",
                font=(_FONT, 8), padx=4, pady=5,
                command=lambda f=flip: self._on_flip(f)
            )
            btn.pack(side=tk.LEFT, padx=2, pady=2, expand=True, fill=tk.X)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=BG_HOVER, fg=TEXT_PRIMARY))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=BG_CARD, fg=TEXT_MUTED))

    # ── Tab: Settings ─────────────────────────────────────────────────────────

    def _build_settings_tab(self, nb):
        tab = tk.Frame(nb, bg=BG_PANEL)
        nb.add(tab, text="Settings")

        SectionLabel(tab, text="IMAGE PROCESSING").pack(fill=tk.X, padx=10, pady=(14, 8))

        slider_defs = [
            ("Brightness",         0, 100, self._on_brightness, 50),
            ("Contrast",           0, 100, self._on_contrast,   50),
            ("Detail Enhance",     0, 100, self._on_detail,     50),
            ("SNR",                0, 100, self._on_snr,        50),
            ("TNR",                0, 100, self._on_tnr,        50),
        ]

        self.sliders = {}
        for label, from_, to, cmd, default in slider_defs:
            row = SliderRow(tab, label=label, from_=from_, to=to,
                            command=cmd, default=default)
            row.pack(fill=tk.X, padx=10, pady=4)
            self.sliders[label] = row

        tk.Frame(tab, bg=BG_PANEL, height=1).pack(fill=tk.X, padx=10, pady=10)
        DarkButton(tab, text="Reset All to Default",
                   command=self._reset_settings).pack(padx=10, pady=4, fill=tk.X)

    # ── Tab: Tools ────────────────────────────────────────────────────────────

    def _build_tools_tab(self, nb):
        tab = tk.Frame(nb, bg=BG_PANEL)
        nb.add(tab, text="Tools")

        # Zoom
        SectionLabel(tab, text="ZOOM  (10 = 1x  |  80 = 8x)").pack(fill=tk.X, padx=10, pady=(14, 2))
        self.zoom_slider = SliderRow(tab, label="Zoom Level",
                                     from_=10, to=80,
                                     command=self._on_zoom, default=10)
        self.zoom_slider.pack(fill=tk.X, padx=10, pady=(0, 14))

        # Calibration
        SectionLabel(tab, text="CALIBRATION").pack(fill=tk.X, padx=10, pady=(0, 6))
        AccentButton(tab, text="Run NUC Shutter Calibration",
                     command=self._do_nuc).pack(fill=tk.X, padx=10, pady=3)
        DarkButton(tab, text="Background Correction",
                   command=self._do_bg_correction).pack(fill=tk.X, padx=10, pady=3)

        # Auto Controls
        SectionLabel(tab, text="AUTO CONTROLS").pack(fill=tk.X, padx=10, pady=(14, 6))
        self.auto_shutter_btn = ToggleButton(
            tab, label_off="Auto Shutter   OFF",
            label_on="Auto Shutter   ON",
            command=self._on_auto_shutter,
        )
        self.auto_shutter_btn.pack(fill=tk.X, padx=10, pady=3)

        self.burn_prot_btn = ToggleButton(
            tab, label_off="Burn Protection  OFF",
            label_on="Burn Protection  ON",
            command=self._on_burn_protection,
        )
        self.burn_prot_btn.pack(fill=tk.X, padx=10, pady=3)

        # Shutter Manual
        SectionLabel(tab, text="MANUAL SHUTTER").pack(fill=tk.X, padx=10, pady=(14, 6))
        shutter_row = tk.Frame(tab, bg=BG_PANEL)
        shutter_row.pack(fill=tk.X, padx=10, pady=3)
        DarkButton(shutter_row, text="Open",
                   command=lambda: self._set_shutter(1)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        DarkButton(shutter_row, text="Close",
                   command=lambda: self._set_shutter(0)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

    # ── Camera Helpers ────────────────────────────────────────────────────────

    def _refresh_cameras(self):
        devices = _get_cameras()
        self.camera_combo["values"] = devices
        if devices:
            self.camera_combo.current(0)

    def _auto_connect(self):
        idx = 0
        if FilterGraph is not None and len(self.camera_combo["values"]) > 1:
            idx = 1
            self.camera_combo.current(1)
        self._connect_camera(idx)

    def _reconnect_camera(self):
        sel = self.camera_combo.current()
        if sel == -1:
            return
        idx = _resolve_index(self.camera_var.get(), sel)
        self._connect_camera(idx)

    def _connect_camera(self, index):
        if self.camera:
            self.camera.release()
        self.camera = ThermalCamera(camera_index=index)
        if not self.camera.connect():
            self.max_temp_var.set(f"Camera {index} failed to connect")
            self.camera = None

    # ── Video Loop ────────────────────────────────────────────────────────────

    def _update_video(self):
        if not self.is_running:
            return

        if self.camera:
            temp_array, color_frame = self.camera.get_frame()
            if temp_array is not None and color_frame is not None:
                # Resize frame to the actual container dimensions to avoid feedback loops
                lw = self.vid_border.winfo_width() - 2  # subtract border
                lh = self.vid_border.winfo_height() - 2
                if lw < 10 or lh < 10:          # widget not yet laid-out
                    lw, lh = 640, 480
                display = cv2.resize(color_frame, (lw, lh), interpolation=cv2.INTER_LINEAR)

                # Crosshair
                cx, cy = lw // 2, lh // 2
                cv2.drawMarker(display, (cx, cy), (255, 255, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=22,
                               thickness=1, line_type=cv2.LINE_AA)

                # HUD corner brackets
                blen, bt = 18, 2
                green = (80, 200, 120)
                for (x, y, sx, sy) in [
                    (20, 20, 1, 1), (lw - 20, 20, -1, 1),
                    (20, lh - 20, 1, -1), (lw - 20, lh - 20, -1, -1)
                ]:
                    cv2.line(display, (x, y), (x + sx * blen, y), green, bt)
                    cv2.line(display, (x, y), (x, y + sy * blen), green, bt)

                # Temperature — sample from raw temp_array, smooth with rolling avg
                oh, ow = temp_array.shape[:2]
                self._hist_max.append(float(np.max(temp_array)))
                self._hist_min.append(float(np.min(temp_array)))
                self._hist_ctr.append(float(temp_array[oh // 2, ow // 2]))

                max_t = sum(self._hist_max) / len(self._hist_max)
                min_t = sum(self._hist_min) / len(self._hist_min)
                ctr_t = sum(self._hist_ctr) / len(self._hist_ctr)

                self.max_temp_var.set(f"MAX   {max_t:.1f} C")
                self.min_temp_var.set(f"MIN   {min_t:.1f} C")
                self.center_temp_var.set(f"CTR   {ctr_t:.1f} C")

                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                self._photo = ImageTk.PhotoImage(image=pil)
                self.video_label.config(image=self._photo)

        self.root.after(33, self._update_video)

    # ── Mini2 Wrappers ────────────────────────────────────────────────────────

    def _safe_mini2(self, method_name, *args, **kwargs):
        if self.mini2 is None:
            return
        try:
            fn = getattr(self.mini2, method_name)
            fn(*args, **kwargs)
        except Exception as e:
            print(f"[Mini2] {e}")

    def _on_pseudo_color(self, event=None):
        self._update_color_preview(self.color_var.get())
        self._safe_mini2("set_pseudo_color", PseudoColor[self.color_var.get()])

    def _on_scene(self, event=None):
        self._safe_mini2("set_scene", SceneMode[self.scene_var.get()])

    def _on_flip(self, flip):
        self._safe_mini2("set_flip", flip)

    def _on_brightness(self, val):
        self._safe_mini2("set_brightness", int(val))

    def _on_contrast(self, val):
        self._safe_mini2("set_contrast", int(val))

    def _on_detail(self, val):
        self._safe_mini2("set_detail_enhancement", int(val))

    def _on_snr(self, val):
        self._safe_mini2("set_snr", int(val))

    def _on_tnr(self, val):
        self._safe_mini2("set_tnr", int(val))

    def _on_zoom(self, val):
        self._safe_mini2("set_zoom_centre", int(val))

    def _do_nuc(self):
        self._safe_mini2("do_shutter_calibration")

    def _do_bg_correction(self):
        self._safe_mini2("do_background_correction")

    def _on_auto_shutter(self, val):
        self._safe_mini2("set_auto_shutter_switch", val)

    def _on_burn_protection(self, val):
        self._safe_mini2("set_burn_protection", val)

    def _set_shutter(self, val):
        self._safe_mini2("set_shutter_position", val)

    def _reset_settings(self):
        defaults = {"Brightness": 50, "Contrast": 50, "Detail Enhance": 50, "SNR": 50, "TNR": 50}
        for label, val in defaults.items():
            self.sliders[label].set(val)
        self._on_brightness(50)
        self._on_contrast(50)
        self._on_detail(50)
        self._on_snr(50)
        self._on_tnr(50)

    # ── Color Preview ─────────────────────────────────────────────────────────

    PALETTE_COLORS = {
        "WhiteHot":  "#eeeeee",
        "BlackHot":  "#333333",
        "Ironbow":   "#ff4400",
        "Rainbow":   "#8844ff",
        "Sepia":     "#c87941",
        "Night":     "#003399",
        "Aurora":    "#00ff88",
        "RedHot":    "#ff2222",
        "Jungle":    "#22aa44",
        "Medical":   "#00ccff",
        "GoldenRed": "#ffaa00",
    }

    def _update_color_preview(self, name):
        color = self.PALETTE_COLORS.get(name, ACCENT)
        self.color_preview.config(bg=color)

    # ── Close ─────────────────────────────────────────────────────────────────

    def on_close(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
