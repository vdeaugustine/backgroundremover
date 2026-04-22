"""Video tab: frame extraction, list, preview, protect brush, and batch processing."""

import os
import shutil
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageTk

from backgroundremover import bg

from gui.color_cleanup_core import finalize_processed_cutout, resolve_color_cleanup_worker_count
from gui.constants import (
    COLOR_CLEANUP_THRESHOLD_MAX,
    COLOR_CLEANUP_THRESHOLD_MIN,
    COLOR_PROTECT_THRESHOLD_MAX,
    COLOR_PROTECT_THRESHOLD_MIN,
    DEDUPLICATION_THRESHOLD_MAX,
    DEDUPLICATION_THRESHOLD_MIN,
    DUPLICATE_FRAME_MAX_MEAN_DIFFERENCE,
    MAX_PENDING_COLOR_CLEANUP_TASKS_PER_WORKER,
    PREVIEW_ZOOM_STEP,
    PROTECT_BRUSH_RADIUS_DEFAULT,
    PROTECT_BRUSH_RADIUS_MAX,
    PROTECT_BRUSH_RADIUS_MIN,
    PROTECT_OVERLAY_COLOR,
)
from gui.device import DEVICE, DEVICE_NAME
from gui.frame_export import (
    build_export_filename,
    build_frame_similarity_signature,
    dedupe_frame_items,
)
from gui.protection_mask import (
    apply_protection_brush,
    erase_protection_brush,
    make_empty_protection_mask,
)
from gui.widgets import ModernStyle, RoundedButton


class VideoTabMixin:
    def create_video_tab(self, parent):
        """Create the redesigned video frame extraction tab.

        Layout: Header > Source Bar > Toolbar > Workspace (Frame List | Preview + Settings) > Footer.
        Export settings (prefix, save location, resize) are accessed via a dialog.
        """

        # ── SECTION 1: HEADER ──────────────────────────────────────
        header = tk.Frame(parent, bg=ModernStyle.BG_PRIMARY)
        header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            header,
            text="Video Frames",
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_TITLE,
            anchor="w",
        ).pack(side=tk.LEFT)

        tk.Label(
            header,
            text="Extract, process, and export individual frames from video",
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(12, 0), pady=(8, 0))

        # ── SECTION 2: SOURCE BAR ──────────────────────────────────
        source_bar = tk.Frame(
            parent,
            bg=ModernStyle.BG_SECONDARY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
        )
        source_bar.pack(fill=tk.X, pady=(0, 4))

        source_inner = tk.Frame(source_bar, bg=ModernStyle.BG_SECONDARY)
        source_inner.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(
            source_inner,
            text="Source",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.video_entry = tk.Entry(
            source_inner,
            textvariable=self.video_file,
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            insertbackground=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            font=ModernStyle.FONT_BODY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
            highlightcolor=ModernStyle.ACCENT,
        )
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 8))

        self.browse_video_btn = RoundedButton(
            source_inner,
            text="Browse",
            command=self.browse_video,
            width=80,
            height=32,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_SMALL,
        )
        self.browse_video_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.extract_frames_btn = RoundedButton(
            source_inner,
            text="Extract Frames",
            command=self.extract_frames,
            width=130,
            height=32,
            bg=ModernStyle.ACCENT,
            hover_bg=ModernStyle.ACCENT_HOVER,
            font=ModernStyle.FONT_SMALL,
        )
        self.extract_frames_btn.pack(side=tk.LEFT)

        # ── SECTION 3: TOOLBAR ─────────────────────────────────────
        toolbar = tk.Frame(
            parent,
            bg=ModernStyle.BG_SECONDARY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
        )
        toolbar.pack(fill=tk.X, pady=(0, 4))

        toolbar_inner = tk.Frame(toolbar, bg=ModernStyle.BG_SECONDARY)
        toolbar_inner.pack(fill=tk.X, padx=8, pady=5)

        # — Selection group —
        self.select_all_btn = RoundedButton(
            toolbar_inner, text="Select All",
            command=self.select_all_frames,
            width=85, height=28,
            bg=ModernStyle.BG_TERTIARY, hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_SMALL,
        )
        self.select_all_btn.pack(side=tk.LEFT, padx=(0, 3))
        self.select_all_btn.configure_state("disabled")

        self.clear_selection_btn = RoundedButton(
            toolbar_inner, text="Select None",
            command=self.clear_frame_selection,
            width=90, height=28,
            bg=ModernStyle.BG_TERTIARY, hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_SMALL,
        )
        self.clear_selection_btn.pack(side=tk.LEFT, padx=(0, 3))
        self.clear_selection_btn.configure_state("disabled")

        # Divider
        tk.Frame(toolbar_inner, bg=ModernStyle.BORDER, width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2,
        )

        # — Dedupe group —
        self.remove_duplicates_btn = RoundedButton(
            toolbar_inner, text="Dedupe",
            command=self.remove_duplicate_frames,
            width=70, height=28,
            bg=ModernStyle.BG_TERTIARY, hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_SMALL,
        )
        self.remove_duplicates_btn.pack(side=tk.LEFT, padx=(0, 3))
        self.remove_duplicates_btn.configure_state("disabled")

        self.undo_remove_duplicates_btn = RoundedButton(
            toolbar_inner, text="Undo",
            command=self.undo_last_video_edit,
            width=52, height=28,
            bg=ModernStyle.BG_TERTIARY, hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_SMALL,
        )
        self.undo_remove_duplicates_btn.pack(side=tk.LEFT, padx=(0, 3))
        self.undo_remove_duplicates_btn.configure_state("disabled")

        # Divider
        tk.Frame(toolbar_inner, bg=ModernStyle.BORDER, width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2,
        )

        # — Processing group —
        self.remove_bg_frames_btn = RoundedButton(
            toolbar_inner, text="Remove BG",
            command=self.process_remove_background,
            width=95, height=28,
            bg="#5856d6", hover_bg="#4a48c4",
            font=ModernStyle.FONT_SMALL,
        )
        self.remove_bg_frames_btn.pack(side=tk.LEFT, padx=(0, 2))
        self.remove_bg_frames_btn.configure_state("disabled")

        self.remove_bg_frames_options_btn = RoundedButton(
            toolbar_inner, text="⚙",
            command=self.open_video_background_options,
            width=28, height=28,
            bg=ModernStyle.BG_TERTIARY, hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_SMALL,
        )
        self.remove_bg_frames_options_btn.pack(side=tk.LEFT, padx=(0, 3))
        self.remove_bg_frames_options_btn.configure_state("disabled")

        self.apply_color_cleanup_frames_btn = RoundedButton(
            toolbar_inner, text="Color Cleanup",
            command=self.process_color_cleanup,
            width=105, height=28,
            bg=ModernStyle.BG_TERTIARY, hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_SMALL,
        )
        self.apply_color_cleanup_frames_btn.pack(side=tk.LEFT, padx=(0, 3))
        self.apply_color_cleanup_frames_btn.configure_state("disabled")

        # Divider
        tk.Frame(toolbar_inner, bg=ModernStyle.BORDER, width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2,
        )

        # — Protect-brush group —
        self.protect_mode_btn = RoundedButton(
            toolbar_inner, text="Protect",
            command=self._toggle_protect_mode,
            width=65, height=28,
            bg=ModernStyle.BG_TERTIARY, hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_SMALL,
        )
        self.protect_mode_btn.pack(side=tk.LEFT, padx=(0, 2))
        self.protect_mode_btn.configure_state("disabled")

        self.erase_protect_btn = RoundedButton(
            toolbar_inner, text="Erase Protect",
            command=self._toggle_erase_protect_mode,
            width=95, height=28,
            bg=ModernStyle.BG_TERTIARY, hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_SMALL,
        )
        self.erase_protect_btn.pack(side=tk.LEFT, padx=(0, 2))
        self.erase_protect_btn.configure_state("disabled")

        self.clear_protect_btn = RoundedButton(
            toolbar_inner, text="Clear Protect",
            command=self._clear_current_frame_protection,
            width=90, height=28,
            bg=ModernStyle.BG_TERTIARY, hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_SMALL,
        )
        self.clear_protect_btn.pack(side=tk.LEFT, padx=(0, 2))
        self.clear_protect_btn.configure_state("disabled")

        # Brush-size slider (compact, inline)
        tk.Label(
            toolbar_inner, text="Brush:",
            bg=ModernStyle.BG_SECONDARY, fg=ModernStyle.TEXT_SECONDARY,
            font=("SF Pro Text", 9),
        ).pack(side=tk.LEFT, padx=(4, 0))
        self._protect_radius_scale = tk.Scale(
            toolbar_inner,
            from_=PROTECT_BRUSH_RADIUS_MIN, to=PROTECT_BRUSH_RADIUS_MAX,
            resolution=1, orient=tk.HORIZONTAL,
            variable=self.protect_brush_radius,
            bg=ModernStyle.BG_SECONDARY, fg=ModernStyle.TEXT_PRIMARY,
            troughcolor=ModernStyle.BG_TERTIARY,
            activebackground=ModernStyle.ACCENT,
            highlightthickness=0, showvalue=False, sliderlength=10,
            width=10, length=70,
        )
        self._protect_radius_scale.pack(side=tk.LEFT, padx=(2, 0))

        # Divider before save group
        tk.Frame(toolbar_inner, bg=ModernStyle.BORDER, width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2,
        )


        self._export_settings_btn = RoundedButton(
            toolbar_inner, text="⚙",
            command=self._open_export_settings_dialog,
            width=28, height=28,
            bg=ModernStyle.BG_TERTIARY, hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_SMALL,
        )
        self._export_settings_btn.pack(side=tk.RIGHT)

        self.save_frames_btn = RoundedButton(
            toolbar_inner, text="Save Selected",
            command=self.save_selected_frames,
            width=110, height=28,
            bg=ModernStyle.ACCENT, hover_bg=ModernStyle.ACCENT_HOVER,
            font=ModernStyle.FONT_SMALL,
        )
        self.save_frames_btn.pack(side=tk.RIGHT, padx=(0, 3))
        self.save_frames_btn.configure_state("disabled")

        # ── SECTION 4: FOOTER (pack early so it anchors to bottom) ─
        footer = ttk.Frame(parent)
        footer.pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0))

        self.video_progress = ttk.Progressbar(
            footer, mode='indeterminate', style='TProgressbar',
        )
        self.video_progress.pack(fill=tk.X, pady=(0, 3))

        self.video_status_label = ttk.Label(
            footer,
            text=f"Ready. Frames save to {self.frame_output_dir.get()}",
            style='Small.TLabel',
        )
        self.video_status_label.pack(anchor=tk.W, pady=(0, 4))

        # ── SECTION 5: WORKSPACE ───────────────────────────────────
        workspace = ttk.Frame(parent)
        workspace.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        self.frame_results_paned = ttk.Panedwindow(workspace, orient=tk.HORIZONTAL)
        self.frame_results_paned.pack(fill=tk.BOTH, expand=True)

        # — LEFT: Frame list sidebar —
        sidebar = tk.Frame(
            self.frame_results_paned,
            bg=ModernStyle.BG_SECONDARY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
        )
        sidebar.configure(width=250)
        sidebar.pack_propagate(False)

        sidebar_header = tk.Frame(sidebar, bg=ModernStyle.BG_SECONDARY)
        sidebar_header.pack(fill=tk.X, padx=10, pady=(10, 4))

        tk.Label(
            sidebar_header, text="Frames",
            bg=ModernStyle.BG_SECONDARY, fg=ModernStyle.TEXT_PRIMARY,
            font=("SF Pro Text", 12, "bold"),
        ).pack(side=tk.LEFT)

        self._frame_count_label = tk.Label(
            sidebar_header, text="",
            bg=ModernStyle.BG_SECONDARY, fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
        )
        self._frame_count_label.pack(side=tk.RIGHT)

        self._selected_count_label = tk.Label(
            sidebar, text="",
            bg=ModernStyle.BG_SECONDARY, fg=ModernStyle.ACCENT,
            font=ModernStyle.FONT_SMALL,
        )
        self._selected_count_label.pack(anchor=tk.W, padx=10, pady=(0, 4))

        self.frame_list_canvas = tk.Canvas(
            sidebar, bg=ModernStyle.BG_SECONDARY,
            highlightthickness=0, bd=0,
        )
        self.frame_list_canvas.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=(0, 6),
        )

        frame_scrollbar = ttk.Scrollbar(
            sidebar, orient="vertical", command=self.frame_list_canvas.yview,
        )
        frame_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=(0, 6))
        self.frame_list_canvas.configure(yscrollcommand=frame_scrollbar.set)

        self.frame_list_inner = tk.Frame(
            self.frame_list_canvas, bg=ModernStyle.BG_SECONDARY,
        )
        self.frame_list_canvas_window = self.frame_list_canvas.create_window(
            (0, 0), window=self.frame_list_inner, anchor="nw",
        )
        self.frame_list_inner.bind("<Configure>", self._on_frame_list_configure)
        self.frame_list_canvas.bind("<Configure>", self._on_frame_list_canvas_resize)
        self._bind_frame_list_scrolling()

        # — RIGHT: Preview panel + settings —
        right_panel = tk.Frame(
            self.frame_results_paned,
            bg=ModernStyle.BG_SECONDARY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
        )
        self.frame_results_paned.add(sidebar, weight=1)
        self.frame_results_paned.add(right_panel, weight=4)

        # Preview header
        preview_header = tk.Frame(right_panel, bg=ModernStyle.BG_SECONDARY)
        preview_header.pack(fill=tk.X, padx=12, pady=(8, 4))

        tk.Label(
            preview_header, text="Preview",
            bg=ModernStyle.BG_SECONDARY, fg=ModernStyle.TEXT_PRIMARY,
            font=("SF Pro Text", 12, "bold"),
        ).pack(side=tk.LEFT)

        self._preview_info_label = tk.Label(
            preview_header, text="",
            bg=ModernStyle.BG_SECONDARY, fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
        )
        self._preview_info_label.pack(side=tk.RIGHT)

        # Preview canvas
        preview_canvas_frame = tk.Frame(right_panel, bg=ModernStyle.BG_TERTIARY)
        preview_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 4))
        preview_canvas_frame.grid_rowconfigure(0, weight=1)
        preview_canvas_frame.grid_columnconfigure(0, weight=1)

        self.frame_preview_canvas = tk.Canvas(
            preview_canvas_frame, bg=ModernStyle.BG_TERTIARY,
            highlightthickness=0, bd=0,
        )
        self.frame_preview_canvas.grid(row=0, column=0, sticky="nsew")

        preview_vbar = ttk.Scrollbar(
            preview_canvas_frame, orient=tk.VERTICAL,
            command=self.frame_preview_canvas.yview,
        )
        preview_vbar.grid(row=0, column=1, sticky="ns")
        preview_hbar = ttk.Scrollbar(
            preview_canvas_frame, orient=tk.HORIZONTAL,
            command=self.frame_preview_canvas.xview,
        )
        preview_hbar.grid(row=1, column=0, sticky="ew")
        self.frame_preview_canvas.configure(
            xscrollcommand=preview_hbar.set, yscrollcommand=preview_vbar.set,
        )

        self.frame_preview = self.frame_preview_canvas
        self.frame_preview_canvas.bind("<Button-1>", self._on_frame_preview_click)

        # Metadata strip — single line, pill style
        self._meta_bar = tk.Frame(right_panel, bg="#1c1c1e")
        self._meta_bar.pack(fill=tk.X, padx=12, pady=(0, 2))

        self.frame_preview_meta = tk.Label(
            self._meta_bar, text="Select a video and extract frames to begin",
            bg="#1c1c1e", fg=ModernStyle.TEXT_SECONDARY,
            font=("SF Pro Text", 11), anchor="w", justify=tk.LEFT,
        )
        self.frame_preview_meta.pack(side=tk.LEFT, padx=(8, 0), pady=4)

        # ── Collapsible settings panel ─────────────────────────────
        self._settings_visible = tk.BooleanVar(value=True)

        settings_header = tk.Frame(right_panel, bg=ModernStyle.BG_SECONDARY)
        settings_header.pack(fill=tk.X, padx=12, pady=(0, 2))

        self._settings_toggle_btn = tk.Button(
            settings_header,
            text="▼  Settings",
            command=self._toggle_settings_panel,
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_SECONDARY,
            activebackground=ModernStyle.BG_SECONDARY,
            activeforeground=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            font=("SF Pro Text", 10),
            cursor="hand2",
            padx=0,
            pady=2,
            bd=0,
            highlightthickness=0,
        )
        self._settings_toggle_btn.pack(side=tk.LEFT)

        # ── Settings body (collapsible) ────────────────────────────
        self._settings_body = tk.Frame(right_panel, bg=ModernStyle.BG_TERTIARY)
        self._settings_body.pack(fill=tk.X, padx=12, pady=(0, 4))

        settings_inner = tk.Frame(self._settings_body, bg=ModernStyle.BG_TERTIARY)
        settings_inner.pack(fill=tk.X, padx=10, pady=6)

        # ── Row 1: Dedupe threshold ────────────────────────────────
        dedupe_row = tk.Frame(settings_inner, bg=ModernStyle.BG_TERTIARY)
        dedupe_row.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            dedupe_row, text="Dedupe",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_PRIMARY,
            font=("SF Pro Text", 10, "bold"), width=9, anchor="w",
        ).pack(side=tk.LEFT)

        self.deduplication_threshold_scale = tk.Scale(
            dedupe_row,
            from_=DEDUPLICATION_THRESHOLD_MIN, to=DEDUPLICATION_THRESHOLD_MAX,
            resolution=0.001, orient=tk.HORIZONTAL,
            variable=self.deduplication_threshold,
            command=self._on_deduplication_threshold_change,
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_PRIMARY,
            troughcolor="#3a3a3c",
            activebackground=ModernStyle.ACCENT,
            highlightthickness=0, showvalue=False, sliderlength=12,
            font=ModernStyle.FONT_SMALL,
        )
        self.deduplication_threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 6))

        self.deduplication_threshold_value_label = tk.Label(
            dedupe_row,
            text=f"{self.deduplication_threshold.get():.3f}",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.ACCENT,
            font=("SF Pro Text", 10), width=5, anchor="e",
        )
        self.deduplication_threshold_value_label.pack(side=tk.RIGHT)

        tk.Label(
            dedupe_row, text="lower = more",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_SECONDARY,
            font=("SF Pro Text", 9),
        ).pack(side=tk.RIGHT, padx=(0, 4))

        # ── Row 2: Color Cleanup threshold ────────────────────────
        cleanup_row = tk.Frame(settings_inner, bg=ModernStyle.BG_TERTIARY)
        cleanup_row.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            cleanup_row, text="Tolerance",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_PRIMARY,
            font=("SF Pro Text", 10, "bold"), width=9, anchor="w",
        ).pack(side=tk.LEFT)

        self.video_cleanup_threshold_scale = tk.Scale(
            cleanup_row,
            from_=COLOR_CLEANUP_THRESHOLD_MIN, to=COLOR_CLEANUP_THRESHOLD_MAX,
            resolution=1, orient=tk.HORIZONTAL,
            variable=self.video_cleanup_threshold,
            command=self._on_video_cleanup_threshold_change,
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_PRIMARY,
            troughcolor="#3a3a3c",
            activebackground=ModernStyle.ACCENT,
            highlightthickness=0, showvalue=False, sliderlength=12,
            font=ModernStyle.FONT_SMALL,
        )
        self.video_cleanup_threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 6))

        self.video_cleanup_threshold_value_label = tk.Label(
            cleanup_row, text=f"Tol: {self.video_cleanup_threshold.get()}",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.ACCENT,
            font=("SF Pro Text", 10), width=5, anchor="e",
        )
        self.video_cleanup_threshold_value_label.pack(side=tk.RIGHT)

        # ── Row 3: Threshold color cleanup ────────────────────────
        cr1 = tk.Frame(settings_inner, bg=ModernStyle.BG_TERTIARY)
        cr1.pack(fill=tk.X, pady=(0, 3))

        tk.Label(
            cr1, text="Color Cleanup",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_PRIMARY,
            font=("SF Pro Text", 10, "bold"), width=13, anchor="w",
        ).pack(side=tk.LEFT)

        self.pick_cleanup_color_btn = RoundedButton(
            cr1, text="Sample",
            command=self._toggle_preview_color_pick,
            width=58, height=22,
            bg=ModernStyle.BG_SECONDARY, hover_bg=ModernStyle.BORDER,
            font=("SF Pro Text", 9),
        )
        self.pick_cleanup_color_btn.pack(side=tk.LEFT, padx=(0, 2))

        self.add_cleanup_color_btn = RoundedButton(
            cr1, text="Add",
            command=self._choose_video_cleanup_color,
            width=38, height=22,
            bg=ModernStyle.BG_SECONDARY, hover_bg=ModernStyle.BORDER,
            font=("SF Pro Text", 9),
        )
        self.add_cleanup_color_btn.pack(side=tk.LEFT, padx=(0, 2))

        self.clear_cleanup_colors_btn = RoundedButton(
            cr1, text="Clear",
            command=self._clear_video_cleanup_colors,
            width=42, height=22,
            bg=ModernStyle.BG_SECONDARY, hover_bg=ModernStyle.BORDER,
            font=("SF Pro Text", 9),
        )
        self.clear_cleanup_colors_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.video_cleanup_swatches_frame = tk.Frame(cr1, bg=ModernStyle.BG_TERTIARY)
        self.video_cleanup_swatches_frame.pack(side=tk.LEFT, padx=(0, 4))

        self.video_cleanup_colors_label = tk.Label(
            cr1, text="No colors",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_SECONDARY,
            font=("SF Pro Text", 9),
        )
        self.video_cleanup_colors_label.pack(side=tk.LEFT)

        # ── Row 4: Exact match cleanup ─────────────────────────────
        cr2 = tk.Frame(settings_inner, bg=ModernStyle.BG_TERTIARY)
        cr2.pack(fill=tk.X)

        tk.Label(
            cr2, text="Exact Match",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_PRIMARY,
            font=("SF Pro Text", 10, "bold"), width=13, anchor="w",
        ).pack(side=tk.LEFT)

        self.pick_exact_cleanup_color_btn = RoundedButton(
            cr2, text="Sample",
            command=self._toggle_exact_preview_color_pick,
            width=58, height=22,
            bg=ModernStyle.BG_SECONDARY, hover_bg=ModernStyle.BORDER,
            font=("SF Pro Text", 9),
        )
        self.pick_exact_cleanup_color_btn.pack(side=tk.LEFT, padx=(0, 2))

        self.add_exact_cleanup_color_btn = RoundedButton(
            cr2, text="Add",
            command=self._choose_video_exact_cleanup_color,
            width=38, height=22,
            bg=ModernStyle.BG_SECONDARY, hover_bg=ModernStyle.BORDER,
            font=("SF Pro Text", 9),
        )
        self.add_exact_cleanup_color_btn.pack(side=tk.LEFT, padx=(0, 2))

        self.clear_exact_cleanup_colors_btn = RoundedButton(
            cr2, text="Clear",
            command=self._clear_video_exact_cleanup_colors,
            width=42, height=22,
            bg=ModernStyle.BG_SECONDARY, hover_bg=ModernStyle.BORDER,
            font=("SF Pro Text", 9),
        )
        self.clear_exact_cleanup_colors_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.video_exact_cleanup_swatches_frame = tk.Frame(cr2, bg=ModernStyle.BG_TERTIARY)
        self.video_exact_cleanup_swatches_frame.pack(side=tk.LEFT, padx=(0, 4))

        self.video_exact_cleanup_colors_label = tk.Label(
            cr2, text="No colors",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_SECONDARY,
            font=("SF Pro Text", 10),
        )
        self.video_exact_cleanup_colors_label.pack(side=tk.LEFT)

        # ── Row 5: Protected-color tolerance ──────────────────────────
        protect_tol_row = tk.Frame(settings_inner, bg=ModernStyle.BG_TERTIARY)
        protect_tol_row.pack(fill=tk.X, pady=(6, 4))

        tk.Label(
            protect_tol_row, text="Protect Tol",
            bg=ModernStyle.BG_TERTIARY, fg="#FF9500",
            font=("SF Pro Text", 10, "bold"), width=9, anchor="w",
        ).pack(side=tk.LEFT)

        self.video_protected_threshold_scale = tk.Scale(
            protect_tol_row,
            from_=COLOR_PROTECT_THRESHOLD_MIN, to=COLOR_PROTECT_THRESHOLD_MAX,
            resolution=1, orient=tk.HORIZONTAL,
            variable=self.video_protected_threshold,
            command=self._on_video_protected_threshold_change,
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_PRIMARY,
            troughcolor="#3a3a3c",
            activebackground="#FF9500",
            highlightthickness=0, showvalue=False, sliderlength=12,
            font=ModernStyle.FONT_SMALL,
        )
        self.video_protected_threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 6))

        self.video_protected_threshold_value_label = tk.Label(
            protect_tol_row, text=f"Tol: {self.video_protected_threshold.get()}",
            bg=ModernStyle.BG_TERTIARY, fg="#FF9500",
            font=("SF Pro Text", 10), width=5, anchor="e",
        )
        self.video_protected_threshold_value_label.pack(side=tk.RIGHT)

        # ── Row 6: Protected color list ─────────────────────────────
        cr3 = tk.Frame(settings_inner, bg=ModernStyle.BG_TERTIARY)
        cr3.pack(fill=tk.X, pady=(0, 3))

        tk.Label(
            cr3, text="Protect Colors",
            bg=ModernStyle.BG_TERTIARY, fg="#FF9500",
            font=("SF Pro Text", 10, "bold"), width=13, anchor="w",
        ).pack(side=tk.LEFT)

        self.pick_protect_color_btn = RoundedButton(
            cr3, text="Sample",
            command=self._toggle_video_protect_color_pick,
            width=58, height=22,
            bg=ModernStyle.BG_SECONDARY, hover_bg=ModernStyle.BORDER,
            font=("SF Pro Text", 9),
        )
        self.pick_protect_color_btn.pack(side=tk.LEFT, padx=(0, 2))

        self.add_protect_color_btn = RoundedButton(
            cr3, text="Add",
            command=self._choose_video_protected_color,
            width=38, height=22,
            bg=ModernStyle.BG_SECONDARY, hover_bg=ModernStyle.BORDER,
            font=("SF Pro Text", 9),
        )
        self.add_protect_color_btn.pack(side=tk.LEFT, padx=(0, 2))

        self.clear_protect_colors_btn = RoundedButton(
            cr3, text="Clear",
            command=self._clear_video_protected_colors,
            width=42, height=22,
            bg=ModernStyle.BG_SECONDARY, hover_bg=ModernStyle.BORDER,
            font=("SF Pro Text", 9),
        )
        self.clear_protect_colors_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.video_protected_swatches_frame = tk.Frame(cr3, bg=ModernStyle.BG_TERTIARY)
        self.video_protected_swatches_frame.pack(side=tk.LEFT, padx=(0, 4))

        self.video_protected_colors_label = tk.Label(
            cr3, text="No colors",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_SECONDARY,
            font=("SF Pro Text", 9),
        )
        self.video_protected_colors_label.pack(side=tk.LEFT)

        tk.Label(
            settings_inner,
            text="\u26a0 Protected colors override cleanup when colours overlap (safety-first).",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_SECONDARY,
            font=("SF Pro Text", 9), anchor="w",
        ).pack(fill=tk.X, pady=(2, 4))

        # ── Initialize control states ──────────────────────────────
        # Create hidden references for export settings used by other methods
        self.frame_output_entry = None
        self.browse_frame_output_btn = None
        self.video_output_prefix_entry = None

        self._refresh_video_cleanup_controls()
        self._bind_frame_navigation()

    def _clear_frame_preview_canvas(self, message=""):
        """Clear the frame preview canvas and optionally show a placeholder message."""
        canvas = self.frame_preview_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 120)
        ch = max(canvas.winfo_height(), 80)
        if message:
            canvas.create_text(
                cw // 2, ch // 2, text=message,
                fill=ModernStyle.TEXT_SECONDARY,
                font=ModernStyle.FONT_SMALL, anchor=tk.CENTER,
            )

    def _update_sidebar_counts(self):
        """Refresh the frame count and selection count in the sidebar header."""
        total = len(self.frame_items)
        selected = self._selected_frame_count() if self.frame_items else 0
        if hasattr(self, '_frame_count_label'):
            self._frame_count_label.configure(text=f"{total} frames" if total else "")
        if hasattr(self, '_selected_count_label'):
            text = f"{selected} selected" if selected > 0 else ""
            self._selected_count_label.configure(text=text)

    def browse_video(self):
        """Open file dialog for input video"""
        filetypes = [
            ("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv *.webm"),
            ("All files", "*.*"),
        ]

        filename = filedialog.askopenfilename(
            title="Select Video",
            filetypes=filetypes,
        )

        if filename:
            # Force a clean state -- if a prior operation left the app in a
            # broken processing state, loading a new video should always recover.
            self.video_processing = False
            self.video_file.set(filename)
            self.video_output_prefix.set(os.path.splitext(os.path.basename(filename))[0])
            self._reset_extracted_frames(clear_cleanup_colors=True)
            self.video_status_label.configure(
                text="Video selected. Click Extract Frames to build the frame list.",
                foreground=ModernStyle.TEXT_SECONDARY,
            )

    def browse_frame_output(self):
        """Open folder dialog for frame output"""
        directory = filedialog.askdirectory(
            title="Choose Folder For Saved Frames",
            initialdir=self.frame_output_dir.get() or os.path.join(os.path.expanduser("~"), "Downloads"),
        )

        if directory:
            self.frame_output_dir.set(directory)
            self.video_status_label.configure(
                text=f"Custom frame save folder: {directory}",
                foreground=ModernStyle.TEXT_SECONDARY,
            )

    def _on_frame_list_configure(self, _event=None):
        self.frame_list_canvas.configure(scrollregion=self.frame_list_canvas.bbox("all"))

    def _on_frame_list_canvas_resize(self, event):
        self.frame_list_canvas.itemconfigure(self.frame_list_canvas_window, width=event.width)

    def _bind_frame_list_scrolling(self):
        """Enable mouse-wheel scrolling while the pointer is over the frame list"""
        self.root.bind_all("<MouseWheel>", self._on_frame_list_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_frame_list_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_frame_list_mousewheel, add="+")

    def _event_is_over_frame_list(self, event):
        """Return True when a mouse event originated over the frame list area"""
        x_root = getattr(event, "x_root", None)
        y_root = getattr(event, "y_root", None)
        if x_root is None or y_root is None:
            return False

        widget = self.root.winfo_containing(x_root, y_root)
        while widget is not None:
            if widget == self.frame_list_canvas or widget == self.frame_list_inner:
                return True
            widget = widget.master
        return False

    def _on_frame_list_mousewheel(self, event):
        """Scroll the frame list canvas when the pointer is over it"""
        if not self._event_is_over_frame_list(event):
            return None

        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return None
            step = -1 * int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)

        self.frame_list_canvas.yview_scroll(step, "units")
        return "break"

    def _reset_extracted_frames(self, clear_cleanup_colors=False):
        """Clear extracted frame state and temporary files"""
        for child in self.frame_list_inner.winfo_children():
            child.destroy()

        self.frame_items = []
        self.all_extracted_frame_items = []
        self.full_frame_count = 0
        self.current_frame_index = None
        self.frame_preview_photo = None
        self._video_edit_history = []
        self._clear_frame_preview_canvas("Extract frames to start reviewing them")
        self.frame_preview_meta.configure(text="")
        self._update_sidebar_counts()
        self.select_all_btn.configure_state("disabled")
        self.remove_duplicates_btn.configure_state("disabled")
        self.undo_remove_duplicates_btn.configure_state("disabled")
        self.clear_selection_btn.configure_state("disabled")
        self.save_frames_btn.configure_state("disabled")
        self.remove_bg_frames_btn.configure_state("disabled")
        self.remove_bg_frames_options_btn.configure_state("disabled")
        self.preview_color_pick_active = False
        self.video_protect_color_pick_active = False
        self._frame_preview_interaction_mode = "off"
        self._protect_brush_dragging = False
        if clear_cleanup_colors:
            self.video_cleanup_colors = []
            self.video_protected_colors = []
        self._refresh_video_cleanup_controls()
        self._refresh_protect_controls()
        self._cleanup_frame_temp_dir()

    def _cleanup_frame_temp_dir(self):
        if self.frame_temp_dir is not None:
            try:
                self.frame_temp_dir.cleanup()
            except Exception:
                pass
            self.frame_temp_dir = None

    def extract_frames(self):
        """Start frame extraction in a background thread"""
        if not self.video_file.get():
            messagebox.showerror("Error", "Please choose a video file first.")
            return

        if not os.path.exists(self.video_file.get()):
            messagebox.showerror("Error", "Selected video file does not exist.")
            return

        if self.video_processing:
            return

        self.video_processing = True
        self._reset_extracted_frames()
        self._set_video_action_states(is_busy=True)
        self.video_progress.start(10)
        self.video_status_label.configure(
            text="Extracting frames. Large videos can take a while.",
            foreground=ModernStyle.TEXT_SECONDARY,
        )

        thread = threading.Thread(target=self._extract_frames_thread, daemon=True)
        thread.start()

    def _extract_frames_thread(self):
        """Extract every frame from the selected video to a temp directory"""
        try:
            from moviepy import VideoFileClip

            temp_dir = tempfile.TemporaryDirectory(prefix="backgroundremover_frames_")
            frame_items = []

            with VideoFileClip(self.video_file.get()) as clip:
                fps = getattr(clip, "fps", None) or 0
                duration = getattr(clip, "duration", 0) or 0
                total_frames_estimate = max(1, int(round(duration * fps))) if fps else 0
                self.video_clip_metadata = {
                    "fps": fps,
                    "duration": duration,
                    "estimated_frames": total_frames_estimate,
                }

                for index, frame in enumerate(clip.iter_frames(dtype="uint8")):
                    frame_image = Image.fromarray(frame)
                    frame_path = os.path.join(temp_dir.name, f"frame_{index + 1:06d}.png")
                    frame_image.save(frame_path, "PNG")

                    thumbnail = frame_image.copy()
                    thumbnail.thumbnail((92, 72), Image.Resampling.LANCZOS)
                    compare_array = build_frame_similarity_signature(frame_image)

                    frame_items.append({
                        "index": index,
                        "name": f"Frame {index + 1}",
                        "path": frame_path,
                        "size": frame_image.size,
                        "thumbnail": thumbnail,
                        "compare_array": compare_array,
                    })

            self.root.after(0, lambda: self._finish_frame_extraction(temp_dir, frame_items))

        except Exception as e:
            self.root.after(0, lambda: self._on_frame_extraction_error(str(e)))

    def _finish_frame_extraction(self, temp_dir, frame_items):
        """Populate the UI after frame extraction finishes"""
        self.frame_temp_dir = temp_dir
        self.video_processing = False
        self.video_progress.stop()
        self.full_frame_count = len(frame_items)
        self.extract_frames_btn.configure_state("normal")

        if not frame_items:
            self.video_status_label.configure(
                text="No frames were extracted from that video.",
                foreground=ModernStyle.ERROR,
            )
            return

        # Set all_extracted_frame_items BEFORE _rebuild_frame_list so that
        # _set_video_action_states (called inside _rebuild_frame_list) sees
        # the full frame list and correctly enables Remove Duplicates.
        self.all_extracted_frame_items = [dict(item) for item in frame_items]
        self._rebuild_frame_list(frame_items)
        # Refresh button states again now that everything is fully populated.
        self._set_video_action_states(is_busy=False)

        estimated = self.video_clip_metadata.get("estimated_frames")
        actual_count = len(self.frame_items)
        details = f"{actual_count} frames extracted"
        if estimated and estimated != actual_count:
            details += f" (estimated {estimated})"

        self.video_status_label.configure(
            text=f"{details}. Click thumbnails on the left and check the frames you want to save.",
            foreground=ModernStyle.SUCCESS,
        )

    def _on_frame_extraction_error(self, error_msg):
        """Handle frame extraction failure"""
        self.video_processing = False
        self.video_progress.stop()
        self._set_video_action_states(is_busy=False)
        self.video_status_label.configure(text="Frame extraction failed.", foreground=ModernStyle.ERROR)
        messagebox.showerror("Error", f"Failed to extract frames:\n\n{error_msg}")

    def _rebuild_frame_list(self, frame_items, preserve_selection=False):
        """Replace the sidebar with a new set of frame items.

        Args:
            frame_items: List of frame item dicts to display.
            preserve_selection: When True, carry over the existing
                ``selected_var`` value from incoming items so that
                user selections survive operations like color cleanup.
        """
        # Snapshot the input list upfront so we are never iterating the
        # same list object that we are about to clear via self.frame_items.
        incoming = list(frame_items)

        for child in self.frame_list_inner.winfo_children():
            child.destroy()

        self.frame_items = []
        self.current_frame_index = None
        self.frame_preview_photo = None
        self._clear_frame_preview_canvas("Extract frames to start reviewing them")
        self.frame_preview_meta.configure(text="")
        self.preview_color_pick_active = False
        self._refresh_video_cleanup_controls()

        for item in incoming:
            frame_item = dict(item)
            # Preserve existing selection when the caller asks for it
            # (e.g. after color cleanup or background removal).
            existing_var = item.get("selected_var")
            was_selected = existing_var.get() if existing_var is not None else bool(item.get("_selected", False))
            initial_value = was_selected if preserve_selection else False
            frame_item["selected_var"] = tk.BooleanVar(value=initial_value)
            frame_item["widget"] = None

            # Regenerate thumbnail from the (possibly updated) file on disk
            # so the sidebar reflects the current image state.
            try:
                with Image.open(frame_item["path"]) as opened:
                    thumb = opened.copy()
                    thumb.thumbnail((92, 72), Image.Resampling.LANCZOS)
                    frame_item["thumbnail"] = thumb
                    frame_item["size"] = opened.size
            except Exception:
                pass  # keep the existing thumbnail/size if the file cannot be read

            self.frame_items.append(frame_item)
            self._add_frame_thumbnail(frame_item)

        if self.frame_items:
            self.current_frame_index = self.frame_items[0]["index"]
            self._show_frame_preview(self.current_frame_index)

        self._set_video_action_states(is_busy=False)
        self._on_frame_list_configure()
        self.frame_list_canvas.yview_moveto(0)

    def _set_video_action_states(self, is_busy):
        """Apply enabled/disabled state to frame action buttons."""
        if is_busy:
            self.extract_frames_btn.configure_state("disabled")
            self.select_all_btn.configure_state("disabled")
            self.remove_duplicates_btn.configure_state("disabled")
            self.undo_remove_duplicates_btn.configure_state("disabled")
            self.clear_selection_btn.configure_state("disabled")
            self.save_frames_btn.configure_state("disabled")
            self.remove_bg_frames_btn.configure_state("disabled")
            self.remove_bg_frames_options_btn.configure_state("disabled")
            self.pick_cleanup_color_btn.configure_state("disabled")
            self.add_cleanup_color_btn.configure_state("disabled")
            self.clear_cleanup_colors_btn.configure_state("disabled")
            self.protect_mode_btn.configure_state("disabled")
            self.erase_protect_btn.configure_state("disabled")
            self.clear_protect_btn.configure_state("disabled")
            return

        self.extract_frames_btn.configure_state("normal")
        has_frames = bool(self.frame_items)
        has_frame_visible = self.current_frame_index is not None
        has_selected_frames = has_frames and self._selected_frame_count() > 0
        can_undo_video_edit = bool(self._video_edit_history)
        can_dedupe = len(self.all_extracted_frame_items) > 1 or len(self.frame_items) > 1
        self.select_all_btn.configure_state("normal" if has_frames else "disabled")
        self.remove_duplicates_btn.configure_state("normal" if can_dedupe else "disabled")
        self.undo_remove_duplicates_btn.configure_state("normal" if can_undo_video_edit else "disabled")
        self.clear_selection_btn.configure_state("normal" if has_frames else "disabled")
        self.save_frames_btn.configure_state("normal" if has_selected_frames else "disabled")
        self.remove_bg_frames_btn.configure_state("normal" if has_selected_frames else "disabled")
        self.remove_bg_frames_options_btn.configure_state("normal" if has_selected_frames else "disabled")
        self.apply_color_cleanup_frames_btn.configure_state("normal" if has_selected_frames else "disabled")
        self.pick_cleanup_color_btn.configure_state("normal" if has_frame_visible else "disabled")
        self.add_cleanup_color_btn.configure_state("normal")
        self.clear_cleanup_colors_btn.configure_state("normal" if self.video_cleanup_colors else "disabled")
        # Protect brush: available whenever a frame is shown.
        self.protect_mode_btn.configure_state("normal" if has_frame_visible else "disabled")
        self.erase_protect_btn.configure_state("normal" if has_frame_visible else "disabled")
        self.clear_protect_btn.configure_state("normal" if has_frame_visible else "disabled")

    def _on_video_cleanup_threshold_change(self, _value=None):
        self.video_cleanup_threshold_value_label.configure(text=f"Tol: {self.video_cleanup_threshold.get()}")
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)
        else:
            self._refresh_video_cleanup_controls()

    def _cleanup_colors_summary(self, colors):
        if not colors:
            return "None selected"

        swatches = [format_rgb_color(color) for color in colors[:4]]
        if len(colors) > 4:
            swatches.append(f"+{len(colors) - 4} more")
        return ", ".join(swatches)

    def _refresh_video_cleanup_controls(self):
        if self.preview_color_pick_active:
            self.pick_cleanup_color_btn.set_text("Click Preview (Threshold)")
            self.frame_preview.configure(cursor="crosshair")
        elif getattr(self, 'exact_preview_color_pick_active', False):
            self.pick_cleanup_color_btn.set_text("Click Preview (Exact)")
            self.frame_preview.configure(cursor="crosshair")
        elif getattr(self, 'video_protect_color_pick_active', False):
            self.pick_cleanup_color_btn.set_text("Sample From Preview")
            self.frame_preview.configure(cursor="crosshair")
        else:
            self.pick_cleanup_color_btn.set_text("Sample From Preview")
            self.frame_preview.configure(cursor="")

        # Update protect sample button text.
        if hasattr(self, 'pick_protect_color_btn'):
            if getattr(self, 'video_protect_color_pick_active', False):
                self.pick_protect_color_btn.set_text("Click Preview")
            else:
                self.pick_protect_color_btn.set_text("Sample")

        self.video_cleanup_colors_label.configure(
            text=f"Cleanup colors: {self._cleanup_colors_summary(self.video_cleanup_colors)}. Tolerance: {self.video_cleanup_threshold.get()}"
        )
        self._rebuild_cleanup_swatches(self.video_cleanup_swatches_frame, self.video_cleanup_colors)

        if hasattr(self, 'video_exact_cleanup_colors_label'):
            self.video_exact_cleanup_colors_label.configure(
                text=f"Exact match colors: {self._cleanup_colors_summary(self.video_exact_cleanup_colors)}"
            )
            self._rebuild_cleanup_swatches(self.video_exact_cleanup_swatches_frame, self.video_exact_cleanup_colors)

        # Refresh protected-color controls.
        if hasattr(self, 'video_protected_colors_label'):
            self.video_protected_colors_label.configure(
                text=f"Protected: {self._cleanup_colors_summary(self.video_protected_colors)}. Tol: {self.video_protected_threshold.get()}"
            )
            self._rebuild_cleanup_swatches(self.video_protected_swatches_frame, self.video_protected_colors)

        if self.video_processing:
            self.pick_cleanup_color_btn.configure_state("disabled")
            self.add_cleanup_color_btn.configure_state("disabled")
            self.clear_cleanup_colors_btn.configure_state("disabled")
            if hasattr(self, 'pick_exact_cleanup_color_btn'):
                self.pick_exact_cleanup_color_btn.configure_state("disabled")
                self.add_exact_cleanup_color_btn.configure_state("disabled")
                self.clear_exact_cleanup_colors_btn.configure_state("disabled")
            if hasattr(self, 'pick_protect_color_btn'):
                self.pick_protect_color_btn.configure_state("disabled")
                self.add_protect_color_btn.configure_state("disabled")
                self.clear_protect_colors_btn.configure_state("disabled")
            return

        self.pick_cleanup_color_btn.configure_state("normal" if self.current_frame_index is not None else "disabled")
        self.add_cleanup_color_btn.configure_state("normal")
        self.clear_cleanup_colors_btn.configure_state("normal" if self.video_cleanup_colors else "disabled")
        
        if hasattr(self, 'pick_exact_cleanup_color_btn'):
            self.pick_exact_cleanup_color_btn.configure_state("normal" if self.current_frame_index is not None else "disabled")
            self.add_exact_cleanup_color_btn.configure_state("normal")
            self.clear_exact_cleanup_colors_btn.configure_state("normal" if getattr(self, 'video_exact_cleanup_colors', []) else "disabled")

        if hasattr(self, 'pick_protect_color_btn'):
            self.pick_protect_color_btn.configure_state("normal" if self.current_frame_index is not None else "disabled")
            self.add_protect_color_btn.configure_state("normal")
            self.clear_protect_colors_btn.configure_state("normal" if self.video_protected_colors else "disabled")

    def _toggle_preview_color_pick(self):
        if self.current_frame_index is None:
            messagebox.showerror("Error", "Select a frame preview before sampling a cleanup color.")
            return

        self.preview_color_pick_active = not self.preview_color_pick_active
        if self.preview_color_pick_active:
            self.exact_preview_color_pick_active = False
            self.video_protect_color_pick_active = False
        self._refresh_video_cleanup_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _toggle_exact_preview_color_pick(self):
        if self.current_frame_index is None:
            messagebox.showerror("Error", "Select a frame preview before sampling an exact color.")
            return

        self.exact_preview_color_pick_active = not getattr(self, 'exact_preview_color_pick_active', False)
        if getattr(self, 'exact_preview_color_pick_active', False):
            self.preview_color_pick_active = False
            self.video_protect_color_pick_active = False
        
        if getattr(self, 'exact_preview_color_pick_active', False):
            self.pick_exact_cleanup_color_btn.set_text("Click Preview")
        else:
            self.pick_exact_cleanup_color_btn.set_text("Sample Exact (Preview)")
            
        self._refresh_video_cleanup_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _choose_video_cleanup_color(self):
        chosen, hex_color = colorchooser.askcolor(
            title="Choose Cleanup Color",
            parent=self.root,
        )
        if chosen is None or hex_color is None:
            return

        color = tuple(int(round(channel)) for channel in chosen[:3])
        self._add_video_cleanup_color(color)

    def _choose_video_exact_cleanup_color(self):
        chosen, hex_color = colorchooser.askcolor(
            title="Choose Exact Cleanup Color",
            parent=self.root,
        )
        if chosen is None or hex_color is None:
            return

        color = tuple(int(round(channel)) for channel in chosen[:3])
        self._add_video_exact_cleanup_color(color)

    def _add_video_cleanup_color(self, color):
        normalized_color = tuple(max(0, min(255, int(channel))) for channel in color[:3])
        if normalized_color not in self.video_cleanup_colors:
            self.video_cleanup_colors.append(normalized_color)

        self.preview_color_pick_active = False
        self._refresh_video_cleanup_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _add_video_exact_cleanup_color(self, color):
        normalized_color = tuple(max(0, min(255, int(channel))) for channel in color[:3])
        if normalized_color not in getattr(self, 'video_exact_cleanup_colors', []):
            if not hasattr(self, 'video_exact_cleanup_colors'):
                self.video_exact_cleanup_colors = []
            self.video_exact_cleanup_colors.append(normalized_color)

        self.exact_preview_color_pick_active = False
        self.pick_exact_cleanup_color_btn.set_text("Sample Exact (Preview)")
        self._refresh_video_cleanup_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _clear_video_cleanup_colors(self):
        self.video_cleanup_colors = []
        self.preview_color_pick_active = False
        self._refresh_video_cleanup_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _clear_video_exact_cleanup_colors(self):
        self.video_exact_cleanup_colors = []
        self.exact_preview_color_pick_active = False
        if hasattr(self, 'pick_exact_cleanup_color_btn'):
            self.pick_exact_cleanup_color_btn.set_text("Sample Exact (Preview)")
        self._refresh_video_cleanup_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    # ── Video protected-color handlers ──────────────────────────────

    def _on_video_protected_threshold_change(self, _value=None):
        """Update the protect-tolerance label and refresh preview on slider change."""
        self.video_protected_threshold_value_label.configure(
            text=f"Tol: {self.video_protected_threshold.get()}"
        )
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)
        else:
            self._refresh_video_cleanup_controls()

    def _toggle_video_protect_color_pick(self):
        """Toggle sampling a protect-color from the frame preview."""
        if self.current_frame_index is None:
            messagebox.showerror("Error", "Select a frame preview before sampling a protected color.")
            return
        self.video_protect_color_pick_active = not self.video_protect_color_pick_active
        if self.video_protect_color_pick_active:
            # Exit other pick modes.
            self.preview_color_pick_active = False
            self.exact_preview_color_pick_active = False
        self._refresh_video_cleanup_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _choose_video_protected_color(self):
        """Open the system color chooser for a protected color."""
        chosen, hex_color = colorchooser.askcolor(
            title="Choose Protected Color",
            parent=self.root,
        )
        if chosen is None or hex_color is None:
            return
        color = tuple(int(round(ch)) for ch in chosen[:3])
        self._add_video_protected_color(color)

    def _add_video_protected_color(self, color):
        """Add a normalized RGB colour to the video protected-colors list."""
        normalized = tuple(max(0, min(255, int(ch))) for ch in color[:3])
        if normalized not in self.video_protected_colors:
            self.video_protected_colors.append(normalized)
        self.video_protect_color_pick_active = False
        self._refresh_video_cleanup_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _clear_video_protected_colors(self):
        """Clear all video protected colors."""
        self.video_protected_colors = []
        self.video_protect_color_pick_active = False
        self._refresh_video_cleanup_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _current_frame_item(self):
        return next((item for item in self.frame_items if item["index"] == self.current_frame_index), None)

    def _snapshot_video_frame_items(self):
        """Snapshot all frame items for undo history, including protection masks."""
        snapshot = []
        for item in self.frame_items:
            snapshot_item = {
                key: value
                for key, value in item.items()
                if key not in {"selected_var", "widget", "thumbnail_photo"}
            }
            selected_var = item.get("selected_var")
            snapshot_item["_selected"] = selected_var.get() if selected_var is not None else bool(item.get("_selected", False))
            # Deep-copy the protection mask so undo history is isolated.
            pm = item.get("protection_mask")
            if pm is not None:
                snapshot_item["protection_mask"] = pm.copy()
            snapshot.append(snapshot_item)
        return snapshot

    def _push_video_edit_history(self, action_name):
        if not self.frame_items:
            return

        self._video_edit_history.append(
            {
                "action_name": action_name,
                "frame_items": self._snapshot_video_frame_items(),
                "current_frame_index": self.current_frame_index,
            }
        )

        if len(self._video_edit_history) > 50:
            self._video_edit_history.pop(0)

    def _video_tab_is_active(self):
        if not hasattr(self, "notebook") or not hasattr(self, "video_tab"):
            return False
        return self.notebook.select() == str(self.video_tab)

    def _handle_video_undo_shortcut(self, event=None):
        if not self._video_tab_is_active():
            return None

        focused_widget = self.root.focus_get()
        if focused_widget is not None and focused_widget.winfo_class() in {"Entry", "TEntry", "Text", "Spinbox", "TCombobox"}:
            return None

        return self.undo_last_video_edit(event)

    def _select_frame_by_index(self, target_index):
        matching = next((item for item in self.frame_items if item["index"] == target_index), None)
        if matching is None:
            return False

        self._show_frame_preview(target_index)
        return True

    def select_previous_frame(self, _event=None):
        if not self.frame_items or self.current_frame_index is None:
            return "break"

        current_position = next((i for i, item in enumerate(self.frame_items) if item["index"] == self.current_frame_index), None)
        if current_position is None or current_position == 0:
            return "break"

        self._select_frame_by_index(self.frame_items[current_position - 1]["index"])
        return "break"

    def select_next_frame(self, _event=None):
        if not self.frame_items or self.current_frame_index is None:
            return "break"

        current_position = next((i for i, item in enumerate(self.frame_items) if item["index"] == self.current_frame_index), None)
        if current_position is None or current_position >= len(self.frame_items) - 1:
            return "break"

        self._select_frame_by_index(self.frame_items[current_position + 1]["index"])
        return "break"

    def _bind_frame_navigation(self):
        self.root.bind_all("<Up>", self.select_previous_frame, add="+")
        self.root.bind_all("<Down>", self.select_next_frame, add="+")
        self.root.bind_all("<Command-z>", self._handle_video_undo_shortcut, add="+")
        self.root.bind_all("<Command-Z>", self._handle_video_undo_shortcut, add="+")
        self.root.bind_all("<Control-z>", self._handle_video_undo_shortcut, add="+")
        self.root.bind_all("<Control-Z>", self._handle_video_undo_shortcut, add="+")

    def _on_frame_preview_click(self, event):
        is_threshold_pick = self.preview_color_pick_active
        is_exact_pick = getattr(self, 'exact_preview_color_pick_active', False)
        is_protect_pick = getattr(self, 'video_protect_color_pick_active', False)
        
        if not (is_threshold_pick or is_exact_pick or is_protect_pick) or self.current_frame_index is None or self.frame_preview_photo is None:
            return

        current_item = self._current_frame_item()
        if current_item is None:
            return

        display_width = self.frame_preview_photo.width()
        display_height = self.frame_preview_photo.height()
        cx = self.frame_preview_canvas.canvasx(event.x)
        cy = self.frame_preview_canvas.canvasy(event.y)
        
        # In _show_frame_preview we use anchor=tk.NW at (x0, y0)
        x_offset = getattr(self, '_frame_preview_x0', 0)
        y_offset = getattr(self, '_frame_preview_y0', 0)

        local_x = cx - x_offset
        local_y = cy - y_offset
        if local_x < 0 or local_y < 0 or local_x >= display_width or local_y >= display_height:
            return

        with Image.open(current_item["path"]) as opened_image:
            source_image = opened_image.convert("RGB")
            source_x = min(source_image.width - 1, max(0, int(local_x * source_image.width / display_width)))
            source_y = min(source_image.height - 1, max(0, int(local_y * source_image.height / display_height)))
            sampled_color = source_image.getpixel((source_x, source_y))

        if is_protect_pick:
            self._add_video_protected_color(sampled_color)
            self.video_status_label.configure(
                text=f"Added protected color {format_rgb_color(sampled_color)}.",
                foreground="#FF9500",
            )
        elif is_exact_pick:
            self._add_video_exact_cleanup_color(sampled_color)
            self.video_status_label.configure(
                text=f"Added exact cleanup color {format_rgb_color(sampled_color)}.",
                foreground=ModernStyle.SUCCESS,
            )
        else:
            self._add_video_cleanup_color(sampled_color)
            self.video_status_label.configure(
                text=f"Added cleanup color {format_rgb_color(sampled_color)}.",
                foreground=ModernStyle.SUCCESS,
            )

    # ── Protect-brush methods ──────────────────────────────────────────────

    def _toggle_protect_mode(self):
        """Toggle the protect-brush interaction mode on/off."""
        if self.current_frame_index is None:
            messagebox.showerror("Error", "Select a frame before painting a protection mask.")
            return
        if self._frame_preview_interaction_mode == "protect":
            self._exit_protect_mode()
        else:
            self._frame_preview_interaction_mode = "protect"
            # Exit color-pick modes.
            self.preview_color_pick_active = False
            self.exact_preview_color_pick_active = False
            self.video_protect_color_pick_active = False
            self._bind_protect_brush_events()
            self._refresh_protect_controls()
            self._show_frame_preview(self.current_frame_index)

    def _toggle_erase_protect_mode(self):
        """Toggle the erase-protect brush interaction mode on/off."""
        if self.current_frame_index is None:
            messagebox.showerror("Error", "Select a frame before erasing a protection mask.")
            return
        if self._frame_preview_interaction_mode == "erase_protect":
            self._exit_protect_mode()
        else:
            self._frame_preview_interaction_mode = "erase_protect"
            self.preview_color_pick_active = False
            self.exact_preview_color_pick_active = False
            self.video_protect_color_pick_active = False
            self._bind_protect_brush_events()
            self._refresh_protect_controls()
            self._show_frame_preview(self.current_frame_index)

    def _exit_protect_mode(self):
        """Return the frame preview to its default (no-brush) interaction mode."""
        self._frame_preview_interaction_mode = "off"
        self._protect_brush_dragging = False
        self._unbind_protect_brush_events()
        self._refresh_protect_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _refresh_protect_controls(self):
        """Update the visual state of protect toolbar buttons to match the active mode."""
        if not hasattr(self, "protect_mode_btn"):
            return
        mode = self._frame_preview_interaction_mode
        # Highlight the active mode with the accent colour; restore others to normal.
        protect_bg = ModernStyle.ACCENT if mode == "protect" else ModernStyle.BG_TERTIARY
        erase_bg = ModernStyle.ACCENT if mode == "erase_protect" else ModernStyle.BG_TERTIARY
        self.protect_mode_btn.bg_color = protect_bg
        self.protect_mode_btn.itemconfig(self.protect_mode_btn.rect, fill=protect_bg)
        self.erase_protect_btn.bg_color = erase_bg
        self.erase_protect_btn.itemconfig(self.erase_protect_btn.rect, fill=erase_bg)

        cursor = "crosshair" if mode in ("protect", "erase_protect") else ""
        self.frame_preview_canvas.configure(cursor=cursor)

    def _bind_protect_brush_events(self):
        """Connect mouse-press and drag events to the brush handler."""
        self.frame_preview_canvas.bind("<ButtonPress-1>", self._on_protect_brush_press)
        self.frame_preview_canvas.bind("<B1-Motion>", self._on_protect_brush_drag)
        self.frame_preview_canvas.bind("<ButtonRelease-1>", self._on_protect_brush_release)

    def _unbind_protect_brush_events(self):
        """Remove brush bindings and restore the default click handler."""
        self.frame_preview_canvas.unbind("<ButtonPress-1>")
        self.frame_preview_canvas.unbind("<B1-Motion>")
        self.frame_preview_canvas.unbind("<ButtonRelease-1>")
        # Re-attach the original click handler from _bind_frame_navigation.
        self.frame_preview_canvas.bind("<Button-1>", self._on_frame_preview_click)

    def _canvas_event_to_mask_coords(self, event):
        """Convert a canvas mouse event to source-image (mask) coordinates.

        Returns (mask_x_float, mask_y_float) or None if the event was outside
        the displayed image.
        """
        item = self._current_frame_item()
        if item is None:
            return None
        canvas = self.frame_preview_canvas
        cx = canvas.canvasx(event.x)
        cy = canvas.canvasy(event.y)
        x0 = getattr(self, "_frame_preview_x0", 0)
        y0 = getattr(self, "_frame_preview_y0", 0)
        dw, dh = getattr(self, "_frame_preview_display_size", (1, 1))
        local_x = cx - x0
        local_y = cy - y0
        if local_x < 0 or local_y < 0 or local_x >= dw or local_y >= dh:
            return None
        src_w, src_h = item["size"]
        mask_x = local_x * src_w / max(dw, 1)
        mask_y = local_y * src_h / max(dh, 1)
        return mask_x, mask_y

    def _apply_brush_stroke(self, event):
        """Apply one brush stroke at the current event position to the active frame's mask."""
        mode = self._frame_preview_interaction_mode
        if mode not in ("protect", "erase_protect"):
            return
        coords = self._canvas_event_to_mask_coords(event)
        if coords is None:
            return
        mx, my = coords
        item = self._current_frame_item()
        if item is None:
            return
        src_w, src_h = item["size"]
        # Ensure the frame has a protection mask.
        if item.get("protection_mask") is None:
            item["protection_mask"] = make_empty_protection_mask(src_w, src_h)
        radius = self.protect_brush_radius.get()
        if mode == "protect":
            apply_protection_brush(item["protection_mask"], mx, my, radius)
        else:
            erase_protection_brush(item["protection_mask"], mx, my, radius)

    def _on_protect_brush_press(self, event):
        """Start a brush drag stroke."""
        self._protect_brush_dragging = True
        self._apply_brush_stroke(event)
        # Refresh overlay on each stroke.
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _on_protect_brush_drag(self, event):
        """Continue the brush drag stroke."""
        if not self._protect_brush_dragging:
            return
        self._apply_brush_stroke(event)
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _on_protect_brush_release(self, event):
        """Finish the brush drag stroke."""
        self._protect_brush_dragging = False

    def _clear_current_frame_protection(self):
        """Remove the entire protection mask from the currently previewed frame."""
        item = self._current_frame_item()
        if item is None:
            return
        item["protection_mask"] = None
        self._exit_protect_mode()
        self.video_status_label.configure(
            text="Protection mask cleared for this frame.",
            foreground=ModernStyle.SUCCESS,
        )
        self._show_frame_preview(self.current_frame_index)

    def remove_duplicate_frames(self):
        """Start duplicate-frame removal in a background thread."""
        if len(self.all_extracted_frame_items) <= 1 or self.video_processing:
            return

        self._push_video_edit_history("remove duplicates")

        self.video_processing = True
        self._set_video_action_states(is_busy=True)
        self.video_progress.start(10)
        self.video_status_label.configure(
            text=f"Comparing frames for duplicates at threshold {self.deduplication_threshold.get():.3f}...",
            foreground=ModernStyle.TEXT_SECONDARY,
        )

        thread = threading.Thread(target=self._remove_duplicates_thread, daemon=True)
        thread.start()

    def _remove_duplicates_thread(self):
        """Filter visually duplicate frames without reopening the video."""
        try:
            unique_items = dedupe_frame_items(
                self.all_extracted_frame_items,
                threshold=self.deduplication_threshold.get(),
            )
            self.root.after(0, lambda: self._finish_duplicate_removal(unique_items))
        except Exception as e:
            self.root.after(0, lambda: self._on_duplicate_removal_error(str(e)))

    def _finish_duplicate_removal(self, unique_items):
        """Update the UI after duplicate removal completes."""
        self.video_processing = False
        self.video_progress.stop()

        if not unique_items:
            unique_items = self.frame_items[:1]

        removed_count = len(self.frame_items) - len(unique_items)
        self._rebuild_frame_list(unique_items)
        self.video_status_label.configure(
            text=(
                f"{self.full_frame_count} frames extracted, {len(self.frame_items)} unique frames kept "
                f"({removed_count} removed) at threshold {self.deduplication_threshold.get():.3f}. "
                f"Use Undo or Cmd+Z to restore the previous step."
            ),
            foreground=ModernStyle.SUCCESS,
        )
        # Re-evaluate states now that the snapshot exists.
        self._set_video_action_states(is_busy=False)

    def _on_duplicate_removal_error(self, error_msg):
        """Handle duplicate-removal failure without losing extracted frames."""
        self.video_processing = False
        self.video_progress.stop()
        if self._video_edit_history and self._video_edit_history[-1]["action_name"] == "remove duplicates":
            self._video_edit_history.pop()
        self._set_video_action_states(is_busy=False)
        self.video_status_label.configure(
            text="Removing duplicate frames failed.",
            foreground=ModernStyle.ERROR,
        )
        messagebox.showerror("Error", f"Failed to remove duplicate frames:\n\n{error_msg}")

    def undo_remove_duplicate_frames(self):
        """Backward-compatible alias for the shared video undo action."""
        return self.undo_last_video_edit()

    def undo_last_video_edit(self, _event=None):
        """Restore the frame list to the state it was in before the last video edit step."""
        if not self._video_edit_history or self.video_processing:
            return "break"

        snapshot = self._video_edit_history.pop()
        self._rebuild_frame_list(snapshot["frame_items"], preserve_selection=True)
        target_index = snapshot.get("current_frame_index")
        if target_index is not None:
            self._select_frame_by_index(target_index)

        self.video_status_label.configure(
            text=f"Undo complete: restored the state before {snapshot['action_name']}.",
            foreground=ModernStyle.SUCCESS,
        )
        self._set_video_action_states(is_busy=False)
        return "break"

    def _add_frame_thumbnail(self, frame_item):
        """Create a single clickable frame thumbnail row"""
        index = frame_item["index"]

        item_frame = tk.Frame(
            self.frame_list_inner,
            bg=ModernStyle.BG_TERTIARY,
            highlightthickness=2,
            highlightbackground=ModernStyle.BG_TERTIARY,
            cursor="hand2",
        )
        item_frame.pack(fill=tk.X, padx=8, pady=6)

        photo = ImageTk.PhotoImage(frame_item["thumbnail"])
        frame_item["thumbnail_photo"] = photo
        frame_item["widget"] = item_frame

        check = tk.Checkbutton(
            item_frame,
            variable=frame_item["selected_var"],
            command=self._update_frame_save_state,
            bg=ModernStyle.BG_TERTIARY,
            activebackground=ModernStyle.BG_TERTIARY,
            selectcolor=ModernStyle.BG_SECONDARY,
        )
        check.pack(side=tk.LEFT, padx=(8, 4))

        thumb_label = tk.Label(
            item_frame,
            image=photo,
            bg=ModernStyle.BG_TERTIARY,
        )
        thumb_label.pack(side=tk.LEFT, padx=(0, 8), pady=8)

        text_label = tk.Label(
            item_frame,
            text=frame_item["name"],
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_BODY,
            anchor="w",
        )
        text_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        for widget in (item_frame, thumb_label, text_label):
            widget.bind("<Button-1>", lambda _event, idx=index: self._show_frame_preview(idx))

    def _show_frame_preview(self, index):
        """Display the selected frame in the preview canvas, with a protection overlay if one exists."""
        matching = next((item for item in self.frame_items if item["index"] == index), None)
        if matching is None:
            return

        self.current_frame_index = index
        with Image.open(matching["path"]) as opened_image:
            image = opened_image.copy()
        # Keep it somewhat constrained but allow scrolling.
        image.thumbnail((1200, 900), Image.Resampling.LANCZOS)
        dw, dh = image.size

        # Draw protection overlay if the frame has one.
        pm = matching.get("protection_mask")
        if pm is not None and np.any(pm):
            # Scale the mask to display resolution.
            src_w, src_h = matching["size"]
            mask_img = Image.fromarray((pm * 255).astype(np.uint8), mode="L")
            mask_resized = mask_img.resize((dw, dh), Image.Resampling.NEAREST)
            overlay = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
            oc = PROTECT_OVERLAY_COLOR
            overlay_color_layer = Image.new("RGBA", (dw, dh), oc)
            overlay.paste(overlay_color_layer, mask=mask_resized)
            base = image.convert("RGBA")
            composited = Image.alpha_composite(base, overlay)
            self.frame_preview_photo = ImageTk.PhotoImage(composited)
        else:
            self.frame_preview_photo = ImageTk.PhotoImage(image)

        canvas = self.frame_preview_canvas
        canvas.delete("all")
        cw = max(canvas.winfo_width(), 1)
        ch = max(canvas.winfo_height(), 1)
        x0 = max(0, (cw - dw) // 2)
        y0 = max(0, (ch - dh) // 2)
        self._frame_preview_x0 = x0
        self._frame_preview_y0 = y0
        # Store display dimensions for brush coordinate mapping.
        self._frame_preview_display_size = (dw, dh)

        canvas.create_image(x0, y0, image=self.frame_preview_photo, anchor=tk.NW)
        canvas.config(scrollregion=(0, 0, max(cw, dw), max(ch, dh)))

        selected_count = self._selected_frame_count()
        width, height = matching["size"]
        sel_icon = "\u2713" if matching['selected_var'].get() else "\u2717"
        meta_text = f"{matching['name']}  \u00b7  {width}\u00d7{height}  \u00b7  {sel_icon} {'Selected' if matching['selected_var'].get() else 'Not selected'}  \u00b7  {selected_count} of {len(self.frame_items)} selected"
        if self.preview_color_pick_active or getattr(self, 'exact_preview_color_pick_active', False):
            meta_text += "  \u00b7  Click preview to sample a cleanup color"
        elif getattr(self, 'video_protect_color_pick_active', False):
            meta_text += "  \u00b7  \U0001f6e1 Click preview to sample a protected color"
        mode = self._frame_preview_interaction_mode
        if mode == "protect":
            meta_text += "  \u00b7  \U0001f6e1 Paint protection \u2014 drag to protect pixels"
        elif mode == "erase_protect":
            meta_text += "  \u00b7  \u2702 Erase protection \u2014 drag to unprotect pixels"
        self.frame_preview_meta.configure(text=meta_text)
        self._update_sidebar_counts()
        if hasattr(self, '_preview_info_label'):
            self._preview_info_label.configure(text=f"{matching['name']}  \u00b7  {width}\u00d7{height}")
        self._refresh_frame_highlight()
        self._refresh_video_cleanup_controls()
        self._refresh_protect_controls()

    def _refresh_frame_highlight(self):
        """Refresh sidebar highlight for the current frame"""
        for item in self.frame_items:
            if item.get("widget") is None:
                continue
            is_current = item["index"] == self.current_frame_index
            item["widget"].configure(
                highlightbackground=ModernStyle.ACCENT if is_current else ModernStyle.BG_TERTIARY
            )

    def _selected_frame_count(self):
        return sum(1 for item in self.frame_items if item["selected_var"].get())

    def _update_frame_save_state(self):
        """Refresh save button state and preview metadata"""
        self._set_video_action_states(is_busy=self.video_processing)
        self._update_sidebar_counts()

        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def select_all_frames(self):
        """Mark every extracted frame for saving"""
        for item in self.frame_items:
            item["selected_var"].set(True)
        self._update_frame_save_state()

    def clear_frame_selection(self):
        """Clear all frame selections"""
        for item in self.frame_items:
            item["selected_var"].set(False)
        self._update_frame_save_state()

    def save_selected_frames(self):
        """Save the selected frames to the target folder"""
        if not self.frame_items:
            messagebox.showerror("Error", "Extract frames before trying to save them.")
            return

        selected_items = [item for item in self.frame_items if item["selected_var"].get()]
        if not selected_items:
            messagebox.showerror("Error", "Select at least one frame to save.")
            return

        target_dir = self.frame_output_dir.get() or os.path.join(os.path.expanduser("~"), "Downloads")
        os.makedirs(target_dir, exist_ok=True)

        if self.video_processing:
            return

        self.video_processing = True
        self.video_progress.start(10)
        self._set_video_action_states(is_busy=True)
        self.video_status_label.configure(
            text=f"Saving {len(selected_items)} frame(s) to {target_dir}",
            foreground=ModernStyle.TEXT_SECONDARY,
        )

        try:
            target_width = int(self.video_export_width.get().strip()) if self.video_export_width.get().strip() else 0
            target_height = int(self.video_export_height.get().strip()) if self.video_export_height.get().strip() else 0
        except ValueError:
            messagebox.showerror("Error", "Invalid export Width or Height.")
            self.video_processing = False
            self._set_video_action_states(is_busy=False)
            self.video_progress.stop()
            return
            
        target_size = (target_width, target_height) if target_width > 0 and target_height > 0 else None

        thread = threading.Thread(
            target=self._save_selected_frames_thread,
            args=(selected_items, target_dir, target_size),
            daemon=True,
        )
        thread.start()

    def open_video_background_options(self):
        """Open a small dialog for batch background-removal settings."""
        if not self.frame_items or self.video_processing:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Background Removal Settings")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.configure(bg=ModernStyle.BG_PRIMARY)

        model_var = tk.StringVar(value=self.video_bg_model_choice.get())
        alpha_var = tk.BooleanVar(value=self.video_bg_alpha_matting.get())
        crop_var = tk.BooleanVar(value=self.video_bg_auto_crop.get())

        container = ttk.Frame(dialog, padding="20")
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Batch Background Removal", style='Title.TLabel').pack(anchor=tk.W, pady=(0, 10))
        ttk.Label(
            container,
            text="Choose how selected frames should be processed before saving.",
            style='Subtitle.TLabel',
        ).pack(anchor=tk.W, pady=(0, 16))

        ttk.Label(container, text="AI Model").pack(anchor=tk.W)
        model_frame = ttk.Frame(container)
        model_frame.pack(fill=tk.X, pady=(8, 15))

        for value, text in (
            ("u2net", "General (Recommended)"),
            ("u2netp", "Fast"),
            ("u2net_human_seg", "People & Portraits"),
        ):
            tk.Radiobutton(
                model_frame,
                text=text,
                variable=model_var,
                value=value,
                bg=ModernStyle.BG_PRIMARY,
                fg=ModernStyle.TEXT_PRIMARY,
                selectcolor=ModernStyle.BG_TERTIARY,
                activebackground=ModernStyle.BG_PRIMARY,
                activeforeground=ModernStyle.ACCENT,
                font=ModernStyle.FONT_BODY,
                padx=5,
            ).pack(anchor=tk.W)
        tk.Checkbutton(
            container,
            text="Enable Alpha Matting (higher quality edges)",
            variable=alpha_var,
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_PRIMARY,
            selectcolor=ModernStyle.BG_TERTIARY,
            activebackground=ModernStyle.BG_PRIMARY,
            activeforeground=ModernStyle.ACCENT,
            font=ModernStyle.FONT_BODY,
            padx=5,
        ).pack(anchor=tk.W, pady=(0, 5))

        tk.Checkbutton(
            container,
            text="Auto Crop To Content (clip to hug contents)",
            variable=crop_var,
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_PRIMARY,
            selectcolor=ModernStyle.BG_TERTIARY,
            activebackground=ModernStyle.BG_PRIMARY,
            activeforeground=ModernStyle.ACCENT,
            font=ModernStyle.FONT_BODY,
            padx=5,
        ).pack(anchor=tk.W, pady=(0, 5))

        padding_frame = ttk.Frame(container)
        padding_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(padding_frame, text="Crop Padding (px):").pack(side=tk.LEFT)
        padding_val_label = ttk.Label(padding_frame, text=str(self.video_bg_crop_padding.get()), style='Small.TLabel')
        padding_val_label.pack(side=tk.RIGHT)
        
        padding_var = tk.IntVar(value=self.video_bg_crop_padding.get())
        
        def update_padding_label(val):
            padding_val_label.configure(text=str(int(float(val))))

        tk.Scale(
            padding_frame,
            from_=0,
            to=100,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=padding_var,
            command=update_padding_label,
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_PRIMARY,
            troughcolor=ModernStyle.BG_TERTIARY,
            activebackground=ModernStyle.ACCENT,
            highlightthickness=0,
            font=ModernStyle.FONT_SMALL,
        ).pack(fill=tk.X, pady=(5, 0))

        button_row = ttk.Frame(container)
        button_row.pack(fill=tk.X)

        def start_with_settings():
            self.video_bg_model_choice.set(model_var.get())
            self.video_bg_alpha_matting.set(alpha_var.get())
            self.video_bg_auto_crop.set(crop_var.get())
            self.video_bg_crop_padding.set(padding_var.get())
            dialog.destroy()
            self.remove_background_and_save_selected_frames()

        RoundedButton(
            button_row,
            text="Cancel",
            command=dialog.destroy,
            width=120,
            height=38,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        ).pack(side=tk.RIGHT)

        RoundedButton(
            button_row,
            text="Apply + Run",
            command=start_with_settings,
            width=150,
            height=38,
            bg=ModernStyle.ACCENT,
            hover_bg=ModernStyle.ACCENT_HOVER,
        ).pack(side=tk.RIGHT, padx=(0, 10))

        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_rooty() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.grab_set()
        dialog.focus_set()

    def process_remove_background(self):
        """Apply AI background removal inline to update previews."""
        self._process_frames_inline(action="remove_background")

    def process_color_cleanup(self):
        """Apply Color Cleanup inline to update previews."""
        self._process_frames_inline(action="color_cleanup")

    def _process_frames_inline(self, action):
        if not self.frame_items:
            messagebox.showerror("Error", "Extract frames before processing them.")
            return

        selected_items = [item for item in self.frame_items if item["selected_var"].get()]
        if not selected_items:
            messagebox.showerror("Error", "Select at least one frame to process.")
            return

        if self.video_processing:
            return

        action_name = "background removal" if action == "remove_background" else "color cleanup"
        self._push_video_edit_history(action_name)
        self.video_processing = True
        self.video_progress.start(10)
        self._set_video_action_states(is_busy=True)
        
        action_name = "Removing backgrounds" if action == "remove_background" else "Applying color cleanup"
        self.video_status_label.configure(
            text=f"{action_name} for {len(selected_items)} selected frame(s)...",
            foreground=ModernStyle.TEXT_SECONDARY,
        )

        thread = threading.Thread(
            target=self._process_frames_inline_thread,
            args=(
                selected_items,
                action,
                self.video_bg_model_choice.get(),
                self.video_bg_alpha_matting.get(),
                list(self.video_cleanup_colors),
                self.video_cleanup_threshold.get(),
                list(self.video_exact_cleanup_colors),
                self.video_bg_auto_crop.get(),
                self.video_bg_crop_padding.get(),
                list(self.video_protected_colors),
                self.video_protected_threshold.get(),
            ),
            daemon=True,
        )
        thread.start()

    def _drain_completed_export_futures(self, pending_futures, saved_paths_by_position, wait_for_all=False):
        """Collect completed async cleanup/save tasks and preserve export order."""
        if not pending_futures:
            return

        done_futures, _ = wait(
            pending_futures.keys(),
            return_when=ALL_COMPLETED if wait_for_all else FIRST_COMPLETED,
        )
        for completed_future in done_futures:
            position = pending_futures.pop(completed_future)
            saved_paths_by_position[position] = completed_future.result()

    def _process_frames_inline_thread(self, selected_items, action, model_name, alpha_matting, cleanup_colors, cleanup_threshold, exact_colors, auto_crop, crop_padding, protected_colors=None, protected_threshold=0):
        """Process frames inline, updating them in temp directory."""
        import uuid
        try:
            net = None
            if action == "remove_background":
                self.root.after(
                    0,
                    lambda: self.video_status_label.configure(
                        text="Loading AI model (first run may take a moment)...",
                        foreground=ModernStyle.TEXT_SECONDARY,
                    ),
                )
                net = self._load_model(model_name)
                # Warm up MPS with a tiny dummy inference so the first real
                # frame doesn't appear to hang during JIT compilation.
                if DEVICE.type == "mps":
                    _dummy = Image.new("RGB", (64, 64), (128, 128, 128))
                    _ = self._predict(net, np.array(_dummy))
                    if hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    del _dummy

            worker_count = resolve_color_cleanup_worker_count(len(selected_items))
            max_pending_tasks = max(1, worker_count * MAX_PENDING_COLOR_CLEANUP_TASKS_PER_WORKER)
            total = len(selected_items)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                pending_futures = {}

                for position, item in enumerate(selected_items, start=1):
                    self.root.after(
                        0,
                        lambda current=position, t=total: self.video_status_label.configure(
                            text=f"Processing frame {current}/{t}...",
                            foreground=ModernStyle.TEXT_SECONDARY,
                        ),
                    )
                    with Image.open(item["path"]) as opened_image:
                        img = opened_image.convert("RGB") if action == "remove_background" else opened_image.convert("RGBA")

                    if action == "remove_background":
                        cutout = self._create_cutout_for_image(img, net, alpha_matting)
                        # Ensure MPS operations complete before continuing
                        if DEVICE.type == "mps" and hasattr(torch.mps, "synchronize"):
                            torch.mps.synchronize()
                    else:
                        cutout = img

                    temp_dir_path = self.frame_temp_dir.name if hasattr(self.frame_temp_dir, 'name') else str(self.frame_temp_dir)
                    destination = os.path.join(
                        temp_dir_path,
                        f"processed_{uuid.uuid4().hex[:8]}.png",
                    )

                    cols = cleanup_colors if action == "color_cleanup" else []
                    exacts = exact_colors if action == "color_cleanup" else []
                    # Honour the per-frame protection mask for colour cleanup.
                    item_protection_mask = item.get("protection_mask")

                    p_cols = protected_colors if action == "color_cleanup" else []
                    p_thr = protected_threshold if action == "color_cleanup" else 0

                    future = executor.submit(
                        finalize_processed_cutout,
                        cutout,
                        destination,
                        cols,
                        cleanup_threshold,
                        exacts,
                        auto_crop if action == "remove_background" else False,
                        None,  # Resize happens on "save selected"
                        crop_padding if action == "remove_background" else 0,
                        item_protection_mask,
                        p_cols,
                        p_thr,
                    )
                    pending_futures[future] = item

                    if len(pending_futures) >= max_pending_tasks:
                        done_futures, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)
                        for done_f in done_futures:
                            done_item = pending_futures.pop(done_f)
                            new_path = done_f.result()
                            done_item["path"] = new_path
                            done_item["compare_array"] = build_frame_similarity_signature(Image.open(new_path))

                # Drain remaining futures
                if pending_futures:
                    done_futures, _ = wait(pending_futures.keys(), return_when=ALL_COMPLETED)
                    for done_f in done_futures:
                        done_item = pending_futures.pop(done_f)
                        new_path = done_f.result()
                        done_item["path"] = new_path
                        done_item["compare_array"] = build_frame_similarity_signature(Image.open(new_path))

            self.root.after(0, self._on_frames_processed_inline)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._on_frame_save_error(str(e)))

    def _on_frames_processed_inline(self):
        """Update the UI after frames are processed inline.

        Preserves existing frame selections so the user does not have
        to re-select all frames after every cleanup or BG removal.

        Wrapped in try/except so a failure in the UI rebuild never
        leaves the app in a permanently broken state with disabled buttons.
        """
        try:
            self.video_processing = False
            self.video_progress.stop()
            self._rebuild_frame_list(self.frame_items, preserve_selection=True)
            self._set_video_action_states(is_busy=False)
            self.video_status_label.configure(
                text="Processed frames updated in preview.",
                foreground=ModernStyle.SUCCESS,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            # Ensure the app is always recoverable.
            self.video_processing = False
            self.video_progress.stop()
            self._set_video_action_states(is_busy=False)
            self.video_status_label.configure(
                text=f"Preview refresh failed: {exc}. Frames may still be recoverable with Undo.",
                foreground=ModernStyle.ERROR,
            )

    def _save_selected_frames_thread(self, selected_items, target_dir, target_size=None):
        """Copy selected frame PNGs to the final output folder, optionally resizing them."""
        try:
            saved_paths = []
            output_prefix = self._resolved_video_output_prefix()

            for position, item in enumerate(selected_items, start=1):
                destination = os.path.join(
                    target_dir,
                    build_export_filename(output_prefix, position),
                )
                if target_size is not None and target_size[0] > 0 and target_size[1] > 0:
                    with Image.open(item["path"]) as img:
                        target_width, target_height = target_size
                        img_width, img_height = img.size
                        left = (img_width - target_width) // 2
                        top = (img_height - target_height) // 2
                        right = (img_width + target_width) // 2
                        bottom = (img_height + target_height) // 2
                        cropped = img.crop((left, top, right, bottom))
                        cropped.save(destination, "PNG")
                else:
                    shutil.copy2(item["path"], destination)
                saved_paths.append(destination)

            self.root.after(0, lambda: self._on_frames_saved(saved_paths, target_dir))
        except Exception as e:
            self.root.after(0, lambda: self._on_frame_save_error(str(e)))

    def _on_frames_saved(self, saved_paths, target_dir):
        """Handle successful frame export"""
        self.video_processing = False
        self.video_progress.stop()
        self._set_video_action_states(is_busy=False)
        self.video_status_label.configure(
            text=f"Saved {len(saved_paths)} frame(s) to {target_dir}",
            foreground=ModernStyle.SUCCESS,
        )
        messagebox.showinfo("Success", f"Saved {len(saved_paths)} frame(s) to:\n\n{target_dir}")

    def _on_frame_save_error(self, error_msg):
        """Handle frame export or inline-processing failure.

        Always re-enables UI controls so the user can retry, undo, or
        load a new video without restarting the app.
        """
        self.video_processing = False
        self.video_progress.stop()
        self._set_video_action_states(is_busy=False)
        self.video_status_label.configure(
            text=f"Processing failed: {error_msg[:120]}",
            foreground=ModernStyle.ERROR,
        )
        messagebox.showerror("Error", f"Processing failed:\n\n{error_msg}")

    def remove_background_and_save_selected_frames(self):
        """Batch remove backgrounds from selected frames and save them to the output directory."""
        if not self.frame_items:
            messagebox.showerror("Error", "Extract frames before processing them.")
            return

        selected_items = [item for item in self.frame_items if item["selected_var"].get()]
        if not selected_items:
            messagebox.showerror("Error", "Select at least one frame to process.")
            return

        target_dir = self.frame_output_dir.get()
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output directory:\n{e}")
                return

        # Resolve optional target size from export settings.
        target_size = None
        try:
            w = self.video_export_width.get().strip()
            h = self.video_export_height.get().strip()
            if w and h:
                target_size = (int(w), int(h))
        except ValueError:
            pass

        self.video_processing = True
        self.video_progress.start(10)
        self._set_video_action_states(is_busy=True)
        self.video_status_label.configure(
            text=f"Starting background removal for {len(selected_items)} frame(s)...",
            foreground=ModernStyle.TEXT_SECONDARY,
        )

        thread = threading.Thread(
            target=self._remove_background_and_save_selected_frames_thread,
            args=(
                selected_items,
                target_dir,
                self.video_bg_model_choice.get(),
                self.video_bg_alpha_matting.get(),
                target_size,
                self.video_bg_auto_crop.get(),
                self.video_bg_crop_padding.get(),
            ),
            daemon=True,
        )
        thread.start()

    def _remove_background_and_save_selected_frames_thread(self, selected_items, target_dir, model_name, alpha_matting, target_size, auto_crop, crop_padding):
        """Worker thread: remove BG and save each frame sequentially.

        GPU inference (predict + cutout) cannot be safely parallelised across
        threads in PyTorch, so frames are processed one at a time.
        """
        try:
            self.root.after(
                0,
                lambda: self.video_status_label.configure(
                    text="Loading AI model (first run may take a moment)...",
                    foreground=ModernStyle.TEXT_SECONDARY,
                ),
            )
            net = self._load_model(model_name)

            # Warm up MPS so the first real frame doesn't appear to hang.
            if DEVICE.type == "mps":
                _dummy = Image.new("RGB", (64, 64), (128, 128, 128))
                _ = self._predict(net, np.array(_dummy))
                if hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
                del _dummy

            total = len(selected_items)
            output_prefix = self._resolved_video_output_prefix()
            saved_paths = []

            for position, item in enumerate(selected_items, start=1):
                self.root.after(
                    0,
                    lambda p=position, t=total: self.video_status_label.configure(
                        text=f"Removing background {p}/{t}...",
                        foreground=ModernStyle.TEXT_SECONDARY,
                    ),
                )

                # Load frame and run AI inference.
                with Image.open(item["path"]) as opened_image:
                    img = opened_image.convert("RGB")

                cutout = self._create_cutout_for_image(img, net, alpha_matting)

                # Ensure MPS ops finish before the next frame.
                if DEVICE.type == "mps" and hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()

                destination = os.path.join(target_dir, build_export_filename(output_prefix, position))

                # Finalize: apply crop+padding, optional resize, save.
                finalize_processed_cutout(
                    cutout,
                    destination,
                    [],         # cleanup_colors — not applied in this path
                    0,          # cleanup_threshold
                    [],         # exact_colors
                    auto_crop,
                    target_size,
                    crop_padding,
                )
                saved_paths.append(destination)

            self.root.after(0, lambda: self._on_background_frames_saved(saved_paths, target_dir))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._on_background_frames_save_error(str(e)))


    def _on_background_frames_saved(self, saved_paths, target_dir):
        """Handle successful batch background removal for selected frames."""
        self.video_processing = False
        self.video_progress.stop()
        self._set_video_action_states(is_busy=False)
        self.video_status_label.configure(
            text=f"Backgrounds removed and saved for {len(saved_paths)} frame(s) to {target_dir}",
            foreground=ModernStyle.SUCCESS,
        )
        messagebox.showinfo(
            "Success",
            f"Background removed and saved for {len(saved_paths)} frame(s) to:\n\n{target_dir}",
        )

    def _on_background_frames_save_error(self, error_msg):
        """Handle batch background-removal failure for selected frames."""
        self.video_processing = False
        self.video_progress.stop()
        self._set_video_action_states(is_busy=False)
        self.video_status_label.configure(
            text="Batch background removal failed.",
            foreground=ModernStyle.ERROR,
        )
        messagebox.showerror("Error", f"Failed to remove backgrounds from selected frames:\n\n{error_msg}")

