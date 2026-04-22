"""Image tab UI, previews, cleanup, and single-image / sprite processing."""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageTk

from backgroundremover import bg

from gui.color_cleanup_core import apply_color_cleanup, finalize_processed_cutout
from gui.color_format import format_rgb_color, tk_rgb_from_color_tuple
from gui.constants import (
    COLOR_CLEANUP_THRESHOLD_MAX,
    COLOR_CLEANUP_THRESHOLD_MIN,
    COLOR_PROTECT_THRESHOLD_MAX,
    COLOR_PROTECT_THRESHOLD_MIN,
    COLOR_SWATCH_MAX_VISIBLE,
    COLOR_SWATCH_SIZE,
    PREVIEW_BASE_HEIGHT,
    PREVIEW_BASE_WIDTH,
    PREVIEW_MAX_DISPLAY_SIDE,
    PREVIEW_ZOOM_MAX,
    PREVIEW_ZOOM_MIN,
    PREVIEW_ZOOM_STEP,
)
from gui.device import DEVICE, DEVICE_NAME
from gui.frame_export import (
    build_export_filename,
    build_frame_similarity_signature,
    crop_to_visible_bounds,
    resolve_output_prefix,
)
from gui.protection_mask import make_empty_protection_mask
from gui.widgets import ModernStyle, RoundedButton


class ImageTabMixin:
    def create_image_tab(self, parent):
        """Create the image background removal tab"""
        
        main_layout = ttk.Frame(parent)
        main_layout.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        left_col = ttk.Frame(main_layout)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        right_col = ttk.Frame(main_layout)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        bottom_layout = ttk.Frame(parent)
        bottom_layout.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

        title_frame = ttk.Frame(left_col)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(title_frame, text="Background Remover", style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, text="Remove backgrounds from images instantly with AI", 
                 style='Subtitle.TLabel').pack(anchor=tk.W, pady=(5, 0))
        
        # Device status
        device_text = f"⚡ {DEVICE_NAME} Acceleration Active" if DEVICE_NAME != "CPU" else f"💻 Running on {DEVICE_NAME}"
        fg_color = ModernStyle.SUCCESS if DEVICE_NAME != "CPU" else ModernStyle.TEXT_SECONDARY
        self.device_label = ttk.Label(title_frame, text=device_text, style='Success.TLabel')
        self.device_label.pack(anchor=tk.W, pady=(10, 0))
        
        # File selection section
        file_frame = ttk.Frame(left_col)
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Input file
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Input Image(s)").pack(anchor=tk.W)
        
        input_entry_frame = ttk.Frame(input_frame)
        input_entry_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.input_entry = tk.Entry(input_entry_frame, 
                                   textvariable=self.input_file,
                                   bg=ModernStyle.BG_TERTIARY,
                                   fg=ModernStyle.TEXT_PRIMARY,
                                   insertbackground=ModernStyle.TEXT_PRIMARY,
                                   relief=tk.FLAT,
                                   font=ModernStyle.FONT_BODY,
                                   highlightthickness=1,
                                   highlightbackground=ModernStyle.BORDER,
                                   highlightcolor=ModernStyle.ACCENT)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        
        self.browse_input_btn = RoundedButton(input_entry_frame, text="Upload",
                                            command=self.browse_input,
                                            width=100, height=36,
                                            bg=ModernStyle.BG_TERTIARY,
                                            hover_bg=ModernStyle.BORDER)
        self.browse_input_btn.pack(side=tk.RIGHT)
        
        # Output file
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(output_frame, text="Output File / Batch Folder").pack(anchor=tk.W)
        
        output_entry_frame = ttk.Frame(output_frame)
        output_entry_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.output_entry = tk.Entry(output_entry_frame,
                                    textvariable=self.output_file,
                                    bg=ModernStyle.BG_TERTIARY,
                                    fg=ModernStyle.TEXT_PRIMARY,
                                    insertbackground=ModernStyle.TEXT_PRIMARY,
                                    relief=tk.FLAT,
                                    font=ModernStyle.FONT_BODY,
                                    highlightthickness=1,
                                    highlightbackground=ModernStyle.BORDER,
                                    highlightcolor=ModernStyle.ACCENT)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        
        self.browse_output_btn = RoundedButton(output_entry_frame, text="Browse",
                                             command=self.browse_output,
                                             width=100, height=36,
                                             bg=ModernStyle.BG_TERTIARY,
                                             hover_bg=ModernStyle.BORDER)
        self.browse_output_btn.pack(side=tk.RIGHT)

        prefix_frame = ttk.Frame(output_frame)
        prefix_frame.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(prefix_frame, text="Batch Name Prefix").pack(side=tk.LEFT, padx=(0, 10))

        self.image_output_prefix_entry = tk.Entry(
            prefix_frame,
            textvariable=self.image_output_prefix,
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            insertbackground=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            font=ModernStyle.FONT_BODY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
            highlightcolor=ModernStyle.ACCENT,
        )
        self.image_output_prefix_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)

        output_actions = ttk.Frame(output_frame)
        output_actions.pack(fill=tk.X, pady=(8, 0))

        self.use_output_as_input_btn = RoundedButton(
            output_actions,
            text="Use output as input",
            command=self.use_output_as_input,
            width=160,
            height=34,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.use_output_as_input_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.save_output_copy_btn = RoundedButton(
            output_actions,
            text="Save output as…",
            command=self.save_output_as_copy,
            width=150,
            height=34,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.save_output_copy_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.reveal_output_btn = RoundedButton(
            output_actions,
            text="Reveal in Finder",
            command=self.reveal_output_in_finder,
            width=150,
            height=34,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        if sys.platform == "darwin":
            self.reveal_output_btn.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(
            output_frame,
            textvariable=self.last_saved_output_hint,
            style="Small.TLabel",
        ).pack(anchor=tk.W, pady=(6, 0))

        ttk.Label(
            output_frame,
            text="Single images save to the Output File. Multiple images save in order to the batch folder as prefix_1.png, prefix_2.png, and so on.",
            style="Small.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        sprite_output_frame = ttk.Frame(file_frame)
        sprite_output_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(sprite_output_frame, text="Sprite Output Folder").pack(anchor=tk.W)

        sprite_output_entry_frame = ttk.Frame(sprite_output_frame)
        sprite_output_entry_frame.pack(fill=tk.X, pady=(5, 0))

        self.sprite_output_entry = tk.Entry(
            sprite_output_entry_frame,
            textvariable=self.sprite_output_dir,
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            insertbackground=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            font=ModernStyle.FONT_BODY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
            highlightcolor=ModernStyle.ACCENT,
        )
        self.sprite_output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))

        self.browse_sprite_output_btn = RoundedButton(
            sprite_output_entry_frame,
            text="Browse",
            command=self.browse_sprite_output,
            width=100,
            height=36,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.browse_sprite_output_btn.pack(side=tk.RIGHT)
        
        # Options section
        options_frame = ttk.Frame(left_col)
        options_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(options_frame, text="AI Model").pack(anchor=tk.W)
        
        model_frame = ttk.Frame(options_frame)
        model_frame.pack(fill=tk.X, pady=(8, 15))
        
        models = [
            ("u2net", "General (Recommended)"),
            ("u2netp", "Fast"),
            ("u2net_human_seg", "People & Portraits")
        ]
        
        for value, text in models:
            rb = tk.Radiobutton(model_frame, text=text, variable=self.model_choice, value=value,
                               bg=ModernStyle.BG_PRIMARY,
                               fg=ModernStyle.TEXT_PRIMARY,
                               selectcolor=ModernStyle.BG_TERTIARY,
                               activebackground=ModernStyle.BG_PRIMARY,
                               activeforeground=ModernStyle.ACCENT,
                               indicatoron=True,
                               padx=5,
                               font=ModernStyle.FONT_BODY)
            rb.pack(side=tk.LEFT, padx=(0, 20))
        
        alpha_check = tk.Checkbutton(options_frame, text="Enable Alpha Matting (higher quality edges)",
                                    variable=self.alpha_matting,
                                    bg=ModernStyle.BG_PRIMARY,
                                    fg=ModernStyle.TEXT_PRIMARY,
                                    selectcolor=ModernStyle.BG_TERTIARY,
                                    activebackground=ModernStyle.BG_PRIMARY,
                                    activeforeground=ModernStyle.ACCENT,
                                    padx=5,
                                    font=ModernStyle.FONT_BODY)
        alpha_check.pack(anchor=tk.W)

        crop_check = tk.Checkbutton(options_frame, text="Auto Crop To Content",
                                   variable=self.auto_crop_output,
                                   bg=ModernStyle.BG_PRIMARY,
                                   fg=ModernStyle.TEXT_PRIMARY,
                                   selectcolor=ModernStyle.BG_TERTIARY,
                                   activebackground=ModernStyle.BG_PRIMARY,
                                   activeforeground=ModernStyle.ACCENT,
                                   padx=5,
                                   font=ModernStyle.FONT_BODY)
        crop_check.pack(anchor=tk.W, pady=(8, 0))

        image_cleanup_frame = ttk.Frame(right_col)
        image_cleanup_frame.pack(fill=tk.X, pady=(15, 0))

        image_cleanup_header = ttk.Frame(image_cleanup_frame)
        image_cleanup_header.pack(fill=tk.X)
        ttk.Label(image_cleanup_header, text="Final Color Cleanup").pack(side=tk.LEFT)
        self.image_cleanup_threshold_value_label = ttk.Label(
            image_cleanup_header,
            text=str(self.image_cleanup_threshold.get()),
            style='Small.TLabel',
        )
        self.image_cleanup_threshold_value_label.pack(side=tk.RIGHT)

        self.image_cleanup_threshold_scale = tk.Scale(
            image_cleanup_frame,
            from_=COLOR_CLEANUP_THRESHOLD_MIN,
            to=COLOR_CLEANUP_THRESHOLD_MAX,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.image_cleanup_threshold,
            command=self._on_image_cleanup_threshold_change,
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_PRIMARY,
            troughcolor=ModernStyle.BG_TERTIARY,
            activebackground=ModernStyle.ACCENT,
            highlightthickness=0,
            font=ModernStyle.FONT_SMALL,
        )
        self.image_cleanup_threshold_scale.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(
            image_cleanup_frame,
            text="Cycle Sample to pick from input or output preview (when output exists), or add a color manually. Matching pixels become transparent on save.",
            style='Small.TLabel',
        ).pack(anchor=tk.W, pady=(4, 0))

        image_cleanup_button_row = ttk.Frame(image_cleanup_frame)
        image_cleanup_button_row.pack(fill=tk.X, pady=(10, 8))

        self.pick_image_cleanup_color_btn = RoundedButton(
            image_cleanup_button_row,
            text="Sample: off (click to cycle)",
            command=self._toggle_image_preview_color_pick,
            width=180,
            height=38,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.pick_image_cleanup_color_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.add_image_cleanup_color_btn = RoundedButton(
            image_cleanup_button_row,
            text="Add Color...",
            command=self._choose_image_cleanup_color,
            width=140,
            height=38,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.add_image_cleanup_color_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_image_cleanup_colors_btn = RoundedButton(
            image_cleanup_button_row,
            text="Clear Colors",
            command=self._clear_image_cleanup_colors,
            width=130,
            height=38,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.clear_image_cleanup_colors_btn.pack(side=tk.LEFT)

        image_cleanup_summary_row = ttk.Frame(image_cleanup_frame)
        image_cleanup_summary_row.pack(fill=tk.X, pady=(6, 0))

        self.image_cleanup_swatches_frame = tk.Frame(
            image_cleanup_summary_row,
            bg=ModernStyle.BG_PRIMARY,
        )
        self.image_cleanup_swatches_frame.pack(side=tk.LEFT)

        self.image_cleanup_colors_label = ttk.Label(
            image_cleanup_summary_row,
            text="No cleanup colors selected.",
            style='Small.TLabel',
        )
        self.image_cleanup_colors_label.pack(side=tk.LEFT, padx=(8, 0))

        # ── Image Protected Colors ────────────────────────────────
        image_protect_frame = ttk.Frame(right_col)
        image_protect_frame.pack(fill=tk.X, pady=(12, 0))

        image_protect_header = ttk.Frame(image_protect_frame)
        image_protect_header.pack(fill=tk.X)
        tk.Label(
            image_protect_header, text="Protected Colors",
            bg=ModernStyle.BG_PRIMARY, fg="#FF9500",
            font=("SF Pro Text", 13, "bold"),
        ).pack(side=tk.LEFT)
        self.image_protected_threshold_value_label = ttk.Label(
            image_protect_header,
            text=str(self.image_protected_threshold.get()),
            style='Small.TLabel',
        )
        self.image_protected_threshold_value_label.pack(side=tk.RIGHT)

        self.image_protected_threshold_scale = tk.Scale(
            image_protect_frame,
            from_=COLOR_PROTECT_THRESHOLD_MIN,
            to=COLOR_PROTECT_THRESHOLD_MAX,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.image_protected_threshold,
            command=self._on_image_protected_threshold_change,
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_PRIMARY,
            troughcolor=ModernStyle.BG_TERTIARY,
            activebackground="#FF9500",
            highlightthickness=0,
            font=ModernStyle.FONT_SMALL,
        )
        self.image_protected_threshold_scale.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(
            image_protect_frame,
            text="\u26a0 Protected colours override cleanup when ranges overlap (safety-first).",
            style='Small.TLabel',
        ).pack(anchor=tk.W, pady=(4, 0))

        image_protect_button_row = ttk.Frame(image_protect_frame)
        image_protect_button_row.pack(fill=tk.X, pady=(10, 8))

        self.pick_image_protect_color_btn = RoundedButton(
            image_protect_button_row,
            text="Sample: off (click to cycle)",
            command=self._toggle_image_protect_color_pick,
            width=180,
            height=38,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.pick_image_protect_color_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.add_image_protect_color_btn = RoundedButton(
            image_protect_button_row,
            text="Add Color...",
            command=self._choose_image_protected_color,
            width=140,
            height=38,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.add_image_protect_color_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_image_protect_colors_btn = RoundedButton(
            image_protect_button_row,
            text="Clear Colors",
            command=self._clear_image_protected_colors,
            width=130,
            height=38,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.clear_image_protect_colors_btn.pack(side=tk.LEFT)

        image_protect_summary_row = ttk.Frame(image_protect_frame)
        image_protect_summary_row.pack(fill=tk.X, pady=(6, 0))

        self.image_protected_swatches_frame = tk.Frame(
            image_protect_summary_row,
            bg=ModernStyle.BG_PRIMARY,
        )
        self.image_protected_swatches_frame.pack(side=tk.LEFT)

        self.image_protected_colors_label = ttk.Label(
            image_protect_summary_row,
            text="No protected colors selected.",
            style='Small.TLabel',
        )
        self.image_protected_colors_label.pack(side=tk.LEFT, padx=(8, 0))

        process_row = ttk.Frame(right_col)
        process_row.pack(pady=(10, 10))

        self.process_btn = RoundedButton(
            process_row,
            text="Remove Background",
            command=self.process_image,
            width=280,
            height=50,
            radius=15,
            bg=ModernStyle.ACCENT,
            hover_bg=ModernStyle.ACCENT_HOVER,
            font=ModernStyle.FONT_BUTTON,
        )
        self.process_btn.pack(side=tk.LEFT, padx=(0, 12))

        self.apply_cleanup_save_btn = RoundedButton(
            process_row,
            text="Apply cleanup & save",
            command=self.apply_cleanup_and_save,
            width=220,
            height=50,
            radius=15,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_BUTTON,
        )
        self.apply_cleanup_save_btn.pack(side=tk.LEFT)

        self.sprite_process_btn = RoundedButton(
            right_col,
            text="Smart Sprite Kit Cut",
            command=self.process_sprite_kit,
            width=280,
            height=50,
            radius=15,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
            font=ModernStyle.FONT_BUTTON,
        )
        self.sprite_process_btn.pack(pady=(0, 20))
        
        # Progress bar
        self.progress = ttk.Progressbar(bottom_layout, mode='indeterminate', style='TProgressbar')
        self.progress.pack(fill=tk.X, pady=(0, 5))
        
        # Status label
        self.status_label = ttk.Label(bottom_layout, text="Ready", style='Small.TLabel')
        self.status_label.pack(pady=(0, 10))
        
        # Preview section (zoomable canvases)
        input_preview_frame = tk.Frame(left_col, bg=ModernStyle.BG_SECONDARY)
        input_preview_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        tk.Label(
            input_preview_frame,
            text="Input",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
        ).pack(pady=(10, 5))

        input_zoom_bar = tk.Frame(input_preview_frame, bg=ModernStyle.BG_SECONDARY)
        input_zoom_bar.pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Label(
            input_zoom_bar,
            text="Zoom",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.input_zoom_value_label = tk.Label(
            input_zoom_bar,
            text="100%",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_SMALL,
        )
        self.input_zoom_value_label.pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            input_zoom_bar,
            text="−",
            command=lambda: self._adjust_input_preview_zoom(1.0 / PREVIEW_ZOOM_STEP),
            width=3,
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            highlightthickness=0,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            input_zoom_bar,
            text="100%",
            command=self._reset_input_preview_zoom,
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            highlightthickness=0,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            input_zoom_bar,
            text="+",
            command=lambda: self._adjust_input_preview_zoom(PREVIEW_ZOOM_STEP),
            width=3,
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            highlightthickness=0,
        ).pack(side=tk.LEFT, padx=2)

        input_canvas_container = tk.Frame(input_preview_frame, bg=ModernStyle.BG_TERTIARY)
        input_canvas_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        input_canvas_container.grid_rowconfigure(0, weight=1)
        input_canvas_container.grid_columnconfigure(0, weight=1)

        self.input_preview_canvas = tk.Canvas(
            input_canvas_container,
            bg=ModernStyle.BG_TERTIARY,
            highlightthickness=0,
            bd=0,
        )
        self.input_preview_canvas.grid(row=0, column=0, sticky="nsew")

        input_vbar = ttk.Scrollbar(input_canvas_container, orient=tk.VERTICAL, command=self.input_preview_canvas.yview)
        input_vbar.grid(row=0, column=1, sticky="ns")
        input_hbar = ttk.Scrollbar(input_canvas_container, orient=tk.HORIZONTAL, command=self.input_preview_canvas.xview)
        input_hbar.grid(row=1, column=0, sticky="ew")
        self.input_preview_canvas.configure(xscrollcommand=input_hbar.set, yscrollcommand=input_vbar.set)
        self.input_preview_canvas.bind("<Button-1>", self._on_input_preview_canvas_click)
        self.input_preview_canvas.bind("<MouseWheel>", self._on_input_preview_mousewheel)
        self.input_preview_canvas.bind("<Button-4>", self._on_input_preview_mousewheel)
        self.input_preview_canvas.bind("<Button-5>", self._on_input_preview_mousewheel)
        self.input_preview_canvas.bind("<Configure>", self._on_input_preview_canvas_configure)

        output_preview_frame = tk.Frame(right_col, bg=ModernStyle.BG_SECONDARY)
        output_preview_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        tk.Label(
            output_preview_frame,
            text="Output",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
        ).pack(pady=(10, 5))

        output_zoom_bar = tk.Frame(output_preview_frame, bg=ModernStyle.BG_SECONDARY)
        output_zoom_bar.pack(fill=tk.X, padx=10, pady=(0, 4))
        tk.Label(
            output_zoom_bar,
            text="Zoom",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(0, 6))
        self.output_zoom_value_label = tk.Label(
            output_zoom_bar,
            text="100%",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_SMALL,
        )
        self.output_zoom_value_label.pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            output_zoom_bar,
            text="−",
            command=lambda: self._adjust_output_preview_zoom(1.0 / PREVIEW_ZOOM_STEP),
            width=3,
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            highlightthickness=0,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            output_zoom_bar,
            text="100%",
            command=self._reset_output_preview_zoom,
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            highlightthickness=0,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            output_zoom_bar,
            text="+",
            command=lambda: self._adjust_output_preview_zoom(PREVIEW_ZOOM_STEP),
            width=3,
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            highlightthickness=0,
        ).pack(side=tk.LEFT, padx=2)

        output_canvas_container = tk.Frame(output_preview_frame, bg=ModernStyle.BG_TERTIARY)
        output_canvas_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        output_canvas_container.grid_rowconfigure(0, weight=1)
        output_canvas_container.grid_columnconfigure(0, weight=1)

        self.output_preview_canvas = tk.Canvas(
            output_canvas_container,
            bg=ModernStyle.BG_TERTIARY,
            highlightthickness=0,
            bd=0,
        )
        self.output_preview_canvas.grid(row=0, column=0, sticky="nsew")

        output_vbar = ttk.Scrollbar(output_canvas_container, orient=tk.VERTICAL, command=self.output_preview_canvas.yview)
        output_vbar.grid(row=0, column=1, sticky="ns")
        output_hbar = ttk.Scrollbar(output_canvas_container, orient=tk.HORIZONTAL, command=self.output_preview_canvas.xview)
        output_hbar.grid(row=1, column=0, sticky="ew")
        self.output_preview_canvas.configure(xscrollcommand=output_hbar.set, yscrollcommand=output_vbar.set)
        self.output_preview_canvas.bind("<Button-1>", self._on_output_preview_canvas_click)
        self.output_preview_canvas.bind("<MouseWheel>", self._on_output_preview_mousewheel)
        self.output_preview_canvas.bind("<Button-4>", self._on_output_preview_mousewheel)
        self.output_preview_canvas.bind("<Button-5>", self._on_output_preview_mousewheel)
        self.output_preview_canvas.bind("<Configure>", self._on_output_preview_canvas_configure)

        self._refresh_image_cleanup_controls()
        self._refresh_output_action_buttons()

    def _image_cleanup_colors_summary(self):
        if not self.image_cleanup_colors:
            return "None selected"

        swatches = [format_rgb_color(color) for color in self.image_cleanup_colors[:4]]
        if len(self.image_cleanup_colors) > 4:
            swatches.append(f"+{len(self.image_cleanup_colors) - 4} more")
        return ", ".join(swatches)

    def _refresh_image_cleanup_controls(self):
        if getattr(self, "pick_image_cleanup_color_btn", None) is None:
            return

        if self.image_color_pick_mode == "input":
            self.pick_image_cleanup_color_btn.set_text("Click input preview to sample")
            self.input_preview_canvas.configure(cursor="crosshair")
            self.output_preview_canvas.configure(cursor="")
        elif self.image_color_pick_mode == "output":
            self.pick_image_cleanup_color_btn.set_text("Click output preview to sample")
            self.input_preview_canvas.configure(cursor="")
            self.output_preview_canvas.configure(cursor="crosshair")
        elif getattr(self, 'image_protect_color_pick_mode', 'off') != "off":
            # Protect-color pick is active; leave cleanup button in default.
            self.pick_image_cleanup_color_btn.set_text("Sample: off (click to cycle)")
            cursor_mode = getattr(self, 'image_protect_color_pick_mode', 'off')
            self.input_preview_canvas.configure(cursor="crosshair" if cursor_mode == "input" else "")
            self.output_preview_canvas.configure(cursor="crosshair" if cursor_mode == "output" else "")
        else:
            self.pick_image_cleanup_color_btn.set_text("Sample: off (click to cycle)")
            self.input_preview_canvas.configure(cursor="")
            self.output_preview_canvas.configure(cursor="")

        # Protect sample button text.
        if getattr(self, 'pick_image_protect_color_btn', None):
            pm = getattr(self, 'image_protect_color_pick_mode', 'off')
            if pm == "input":
                self.pick_image_protect_color_btn.set_text("Click input to sample")
            elif pm == "output":
                self.pick_image_protect_color_btn.set_text("Click output to sample")
            else:
                self.pick_image_protect_color_btn.set_text("Sample: off (click to cycle)")

        self.image_cleanup_colors_label.configure(
            text=(
                f"Cleanup colors: {self._image_cleanup_colors_summary()}. "
                f"Tolerance: {self.image_cleanup_threshold.get()}"
            )
        )
        self._rebuild_cleanup_swatches(self.image_cleanup_swatches_frame, self.image_cleanup_colors)

        # Protected color label and swatches.
        if getattr(self, 'image_protected_colors_label', None):
            self.image_protected_colors_label.configure(
                text=(
                    f"Protected: {self._cleanup_colors_summary(self.image_protected_colors)}. "
                    f"Tol: {self.image_protected_threshold.get()}"
                )
            )
            self._rebuild_cleanup_swatches(self.image_protected_swatches_frame, self.image_protected_colors)

        has_input = bool(self.input_file.get()) and os.path.exists(self.input_file.get())
        if self.processing:
            self.pick_image_cleanup_color_btn.configure_state("disabled")
            self.add_image_cleanup_color_btn.configure_state("disabled")
            self.clear_image_cleanup_colors_btn.configure_state("disabled")
            if getattr(self, "apply_cleanup_save_btn", None):
                self.apply_cleanup_save_btn.configure_state("disabled")
            if getattr(self, 'pick_image_protect_color_btn', None):
                self.pick_image_protect_color_btn.configure_state("disabled")
                self.add_image_protect_color_btn.configure_state("disabled")
                self.clear_image_protect_colors_btn.configure_state("disabled")
            return

        self.pick_image_cleanup_color_btn.configure_state("normal" if has_input else "disabled")
        self.add_image_cleanup_color_btn.configure_state("normal")
        self.clear_image_cleanup_colors_btn.configure_state("normal" if self.image_cleanup_colors else "disabled")
        if getattr(self, "apply_cleanup_save_btn", None):
            self.apply_cleanup_save_btn.configure_state(
                "normal" if has_input and not self._is_image_batch_mode() else "disabled"
            )
        if getattr(self, 'pick_image_protect_color_btn', None):
            self.pick_image_protect_color_btn.configure_state("normal" if has_input else "disabled")
            self.add_image_protect_color_btn.configure_state("normal")
            self.clear_image_protect_colors_btn.configure_state("normal" if self.image_protected_colors else "disabled")

    def _on_image_cleanup_threshold_change(self, _value=None):
        self.image_cleanup_threshold_value_label.configure(text=str(self.image_cleanup_threshold.get()))
        self._refresh_image_cleanup_controls()

    def _toggle_image_preview_color_pick(self):
        # Exit protect pick mode when entering cleanup pick.
        self.image_protect_color_pick_mode = "off"

        has_input = bool(self.input_file.get()) and os.path.isfile(self.input_file.get())
        out_path = self.output_file.get()
        has_output = bool(out_path) and os.path.isfile(out_path)

        if self.image_color_pick_mode == "off":
            if not has_input:
                messagebox.showerror("Error", "Select an input image before sampling a cleanup color.")
                return
            self.image_color_pick_mode = "input"
        elif self.image_color_pick_mode == "input":
            self.image_color_pick_mode = "output" if has_output else "off"
        else:
            self.image_color_pick_mode = "off"

        self._refresh_image_cleanup_controls()

    def _choose_image_cleanup_color(self):
        chosen, hex_color = colorchooser.askcolor(
            title="Choose Cleanup Color",
            parent=self.root,
        )
        if chosen is None or hex_color is None:
            return

        color = tuple(int(round(channel)) for channel in chosen[:3])
        self._add_image_cleanup_color(color)

    def _add_image_cleanup_color(self, color):
        normalized_color = tuple(max(0, min(255, int(channel))) for channel in color[:3])
        if normalized_color not in self.image_cleanup_colors:
            self.image_cleanup_colors.append(normalized_color)

        self.image_color_pick_mode = "off"
        self._refresh_image_cleanup_controls()

    def _clear_image_cleanup_colors(self):
        self.image_cleanup_colors = []
        self.image_color_pick_mode = "off"
        self._refresh_image_cleanup_controls()

    # ── Image protected-color handlers ─────────────────────────────

    def _on_image_protected_threshold_change(self, _value=None):
        """Update the protect-tolerance label."""
        if getattr(self, 'image_protected_threshold_value_label', None):
            self.image_protected_threshold_value_label.configure(
                text=str(self.image_protected_threshold.get())
            )
        self._refresh_image_cleanup_controls()

    def _toggle_image_protect_color_pick(self):
        """Cycle the image protect-color sample mode: off -> input -> output -> off."""
        # Exit cleanup pick mode when entering protect pick.
        self.image_color_pick_mode = "off"

        has_input = bool(self.input_file.get()) and os.path.isfile(self.input_file.get())
        out_path = self.output_file.get()
        has_output = bool(out_path) and os.path.isfile(out_path)

        mode = getattr(self, 'image_protect_color_pick_mode', 'off')
        if mode == "off":
            if not has_input:
                messagebox.showerror("Error", "Select an input image before sampling a protected color.")
                return
            self.image_protect_color_pick_mode = "input"
        elif mode == "input":
            self.image_protect_color_pick_mode = "output" if has_output else "off"
        else:
            self.image_protect_color_pick_mode = "off"

        self._refresh_image_cleanup_controls()

    def _choose_image_protected_color(self):
        """Open the system color chooser for a protected color (image tab)."""
        chosen, hex_color = colorchooser.askcolor(
            title="Choose Protected Color",
            parent=self.root,
        )
        if chosen is None or hex_color is None:
            return
        color = tuple(int(round(ch)) for ch in chosen[:3])
        self._add_image_protected_color(color)

    def _add_image_protected_color(self, color):
        """Add a normalized RGB colour to the image protected-colors list."""
        normalized = tuple(max(0, min(255, int(ch))) for ch in color[:3])
        if normalized not in self.image_protected_colors:
            self.image_protected_colors.append(normalized)
        self.image_protect_color_pick_mode = "off"
        self._refresh_image_cleanup_controls()

    def _clear_image_protected_colors(self):
        """Clear all image protected colors."""
        self.image_protected_colors = []
        self.image_protect_color_pick_mode = "off"
        self._refresh_image_cleanup_controls()

    def _rebuild_cleanup_swatches(self, parent_frame, colors):
        """Draw small color circles for the cleanup color list."""
        if parent_frame is None:
            return
        for child in parent_frame.winfo_children():
            child.destroy()
        for rgb in colors[:COLOR_SWATCH_MAX_VISIBLE]:
            swatch = tk.Canvas(
                parent_frame,
                width=COLOR_SWATCH_SIZE,
                height=COLOR_SWATCH_SIZE,
                bg=ModernStyle.BG_PRIMARY,
                highlightthickness=1,
                highlightbackground=ModernStyle.BORDER,
                highlightcolor=ModernStyle.BORDER,
            )
            pad = 2
            swatch.create_oval(
                pad,
                pad,
                COLOR_SWATCH_SIZE - pad,
                COLOR_SWATCH_SIZE - pad,
                fill=tk_rgb_from_color_tuple(rgb),
                outline=ModernStyle.BORDER,
            )
            swatch.pack(side=tk.LEFT, padx=(0, 2))
        if len(colors) > COLOR_SWATCH_MAX_VISIBLE:
            ttk.Label(
                parent_frame,
                text=f"+{len(colors) - COLOR_SWATCH_MAX_VISIBLE}",
                style="Small.TLabel",
            ).pack(side=tk.LEFT, padx=(2, 0))

    def _clamp_preview_zoom(self, value):
        return max(PREVIEW_ZOOM_MIN, min(PREVIEW_ZOOM_MAX, float(value)))

    def _update_input_zoom_label(self):
        if getattr(self, "input_zoom_value_label", None):
            pct = int(round(self.input_preview_zoom.get() * 100))
            self.input_zoom_value_label.configure(text=f"{pct}%")

    def _update_output_zoom_label(self):
        if getattr(self, "output_zoom_value_label", None):
            pct = int(round(self.output_preview_zoom.get() * 100))
            self.output_zoom_value_label.configure(text=f"{pct}%")

    def _open_image_for_preview(self, filepath):
        """Load an image with EXIF orientation applied for accurate previews and sampling."""
        img = Image.open(filepath)
        try:
            from PIL import ImageOps

            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        return img

    def _build_scaled_preview(self, filepath, zoom):
        """Return (display PIL image, source width, source height) for preview drawing."""
        img = self._open_image_for_preview(filepath)
        try:
            source_w, source_h = img.size
            z = self._clamp_preview_zoom(zoom)
            max_w = min(int(PREVIEW_BASE_WIDTH * z), PREVIEW_MAX_DISPLAY_SIDE)
            max_h = min(int(PREVIEW_BASE_HEIGHT * z), PREVIEW_MAX_DISPLAY_SIDE)
            thumb = img.copy()
            thumb.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            return thumb, source_w, source_h
        finally:
            img.close()

    def _draw_preview_canvas_placeholder(self, canvas, message):
        canvas.delete("all")
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 120)
        ch = max(canvas.winfo_height(), 80)
        canvas.create_text(
            cw // 2,
            ch // 2,
            text=message,
            fill=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
            anchor=tk.CENTER,
        )

    def _redraw_input_preview(self):
        if getattr(self, "input_preview_canvas", None) is None:
            return
        canvas = self.input_preview_canvas
        path = self.input_file.get()
        canvas.delete("all")
        self.input_preview_image_id = None
        self._input_preview_draw = None
        self._input_source_size = None
        self.input_photo = None

        if not path or not os.path.isfile(path):
            self._draw_preview_canvas_placeholder(canvas, "No image selected")
            self._refresh_image_cleanup_controls()
            return

        try:
            thumb, sw, sh = self._build_scaled_preview(path, self.input_preview_zoom.get())
            self._input_source_size = (sw, sh)
            dw, dh = thumb.size
            cw = max(canvas.winfo_width(), 1)
            ch = max(canvas.winfo_height(), 1)
            x0 = max(0, (cw - dw) // 2)
            y0 = max(0, (ch - dh) // 2)
            self.input_photo = ImageTk.PhotoImage(thumb)
            self.input_preview_image_id = canvas.create_image(x0, y0, image=self.input_photo, anchor=tk.NW)
            canvas.config(scrollregion=(0, 0, max(cw, dw), max(ch, dh)))
            self._input_preview_draw = {
                "x0": x0,
                "y0": y0,
                "dw": dw,
                "dh": dh,
                "sw": sw,
                "sh": sh,
                "path": path,
            }
        except Exception as exc:
            self._draw_preview_canvas_placeholder(canvas, f"Error: {str(exc)[:40]}")

        self._update_input_zoom_label()
        self._refresh_image_cleanup_controls()

    def _redraw_output_preview(self):
        if getattr(self, "output_preview_canvas", None) is None:
            return
        canvas = self.output_preview_canvas
        path = self.output_preview_display_path
        if path is None:
            path = self.output_file.get()
        canvas.delete("all")
        self.output_preview_image_id = None
        self._output_preview_draw = None
        self._output_source_size = None
        self.output_photo = None

        if not path or not os.path.isfile(path):
            self._draw_preview_canvas_placeholder(canvas, "No output yet")
            self._update_output_zoom_label()
            return

        try:
            thumb, sw, sh = self._build_scaled_preview(path, self.output_preview_zoom.get())
            self._output_source_size = (sw, sh)
            dw, dh = thumb.size
            cw = max(canvas.winfo_width(), 1)
            ch = max(canvas.winfo_height(), 1)
            x0 = max(0, (cw - dw) // 2)
            y0 = max(0, (ch - dh) // 2)
            self.output_photo = ImageTk.PhotoImage(thumb)
            self.output_preview_image_id = canvas.create_image(x0, y0, image=self.output_photo, anchor=tk.NW)
            canvas.config(scrollregion=(0, 0, max(cw, dw), max(ch, dh)))
            self._output_preview_draw = {
                "x0": x0,
                "y0": y0,
                "dw": dw,
                "dh": dh,
                "sw": sw,
                "sh": sh,
                "path": path,
            }
        except Exception as exc:
            self._draw_preview_canvas_placeholder(canvas, f"Error: {str(exc)[:40]}")

        self._update_output_zoom_label()

    def _on_input_preview_canvas_configure(self, event):
        if (event.width, event.height) == self._input_preview_canvas_size:
            return
        self._input_preview_canvas_size = (event.width, event.height)
        self._redraw_input_preview()

    def _on_output_preview_canvas_configure(self, event):
        if (event.width, event.height) == self._output_preview_canvas_size:
            return
        self._output_preview_canvas_size = (event.width, event.height)
        self._redraw_output_preview()

    def _adjust_input_preview_zoom(self, factor):
        new_z = self._clamp_preview_zoom(self.input_preview_zoom.get() * factor)
        self.input_preview_zoom.set(new_z)
        self._redraw_input_preview()

    def _adjust_output_preview_zoom(self, factor):
        new_z = self._clamp_preview_zoom(self.output_preview_zoom.get() * factor)
        self.output_preview_zoom.set(new_z)
        self._redraw_output_preview()

    def _reset_input_preview_zoom(self):
        self.input_preview_zoom.set(1.0)
        self._redraw_input_preview()

    def _reset_output_preview_zoom(self):
        self.output_preview_zoom.set(1.0)
        self._redraw_output_preview()

    def _on_input_preview_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            self._adjust_input_preview_zoom(PREVIEW_ZOOM_STEP)
        elif getattr(event, "num", None) == 5:
            self._adjust_input_preview_zoom(1.0 / PREVIEW_ZOOM_STEP)
        else:
            delta = getattr(event, "delta", 0)
            if delta > 0:
                self._adjust_input_preview_zoom(PREVIEW_ZOOM_STEP)
            elif delta < 0:
                self._adjust_input_preview_zoom(1.0 / PREVIEW_ZOOM_STEP)
        return "break"

    def _on_output_preview_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            self._adjust_output_preview_zoom(PREVIEW_ZOOM_STEP)
        elif getattr(event, "num", None) == 5:
            self._adjust_output_preview_zoom(1.0 / PREVIEW_ZOOM_STEP)
        else:
            delta = getattr(event, "delta", 0)
            if delta > 0:
                self._adjust_output_preview_zoom(PREVIEW_ZOOM_STEP)
            elif delta < 0:
                self._adjust_output_preview_zoom(1.0 / PREVIEW_ZOOM_STEP)
        return "break"

    def _sample_rgb_from_draw(self, draw_info, event_x, event_y):
        """Map a canvas click to an RGB tuple using draw_info from _redraw_*_preview."""
        if not draw_info:
            return None
        ix = event_x
        iy = event_y
        x0 = draw_info["x0"]
        y0 = draw_info["y0"]
        dw = draw_info["dw"]
        dh = draw_info["dh"]
        sw = draw_info["sw"]
        sh = draw_info["sh"]
        path = draw_info["path"]
        if ix < x0 or iy < y0 or ix >= x0 + dw or iy >= y0 + dh:
            return None
        local_x = ix - x0
        local_y = iy - y0
        src_x = min(sw - 1, max(0, int(local_x * sw / max(dw, 1))))
        src_y = min(sh - 1, max(0, int(local_y * sh / max(dh, 1))))
        img = self._open_image_for_preview(path)
        try:
            rgb_img = img.convert("RGB")
            return rgb_img.getpixel((src_x, src_y))
        finally:
            img.close()

    def _on_input_preview_canvas_click(self, event):
        protect_mode = getattr(self, 'image_protect_color_pick_mode', 'off')
        if self.image_color_pick_mode != "input" and protect_mode != "input":
            return
        cx = event.widget.canvasx(event.x)
        cy = event.widget.canvasy(event.y)
        sampled = self._sample_rgb_from_draw(self._input_preview_draw, cx, cy)
        if sampled is None:
            return
        if protect_mode == "input":
            self._add_image_protected_color(sampled)
            self.status_label.configure(
                text=f"Added protected color {format_rgb_color(sampled)}.",
                foreground="#FF9500",
            )
        else:
            self._add_image_cleanup_color(sampled)
            self.status_label.configure(
                text=f"Added cleanup color {format_rgb_color(sampled)}. Matching pixels will be removed before save.",
                foreground=ModernStyle.SUCCESS,
            )

    def _on_output_preview_canvas_click(self, event):
        protect_mode = getattr(self, 'image_protect_color_pick_mode', 'off')
        if self.image_color_pick_mode != "output" and protect_mode != "output":
            return
        cx = event.widget.canvasx(event.x)
        cy = event.widget.canvasy(event.y)
        sampled = self._sample_rgb_from_draw(self._output_preview_draw, cx, cy)
        if sampled is None:
            return
        if protect_mode == "output":
            self._add_image_protected_color(sampled)
            self.status_label.configure(
                text=f"Added protected color {format_rgb_color(sampled)} from output preview.",
                foreground="#FF9500",
            )
        else:
            self._add_image_cleanup_color(sampled)
            self.status_label.configure(
                text=f"Added cleanup color {format_rgb_color(sampled)} from output preview.",
                foreground=ModernStyle.SUCCESS,
            )

    def _set_last_saved_output_hint(self, path):
        if path:
            self.last_saved_output_hint.set(f"Last saved: {path}")
        else:
            self.last_saved_output_hint.set("")

    def _selected_image_input_paths(self):
        """Return the currently selected image paths, preserving upload order."""
        single_path = (self.input_file.get() or "").strip()
        paths = [
            path for path in getattr(self, "image_input_files", [])
            if path and os.path.isfile(path)
        ]
        if paths:
            if single_path and single_path not in paths and os.path.isfile(single_path):
                return [single_path]
            return paths

        if single_path and os.path.isfile(single_path):
            return [single_path]
        return []

    def _is_image_batch_mode(self):
        return len(self._selected_image_input_paths()) > 1

    def _default_image_output_prefix(self):
        input_paths = self._selected_image_input_paths()
        first_path = input_paths[0] if input_paths else ""
        return os.path.splitext(os.path.basename(first_path))[0] or "image"

    def _resolved_image_output_prefix(self):
        return resolve_output_prefix(
            self.image_output_prefix.get(),
            self._default_image_output_prefix(),
        )

    def _refresh_output_action_buttons(self):
        if getattr(self, "use_output_as_input_btn", None) is None:
            return
        out_path = self.output_file.get()
        has_output_file = bool(out_path) and os.path.isfile(out_path)
        has_output_location = bool(out_path) and (os.path.isfile(out_path) or os.path.isdir(out_path))
        busy = self.processing
        self.use_output_as_input_btn.configure_state("normal" if has_output_file and not busy else "disabled")
        self.save_output_copy_btn.configure_state("normal" if has_output_file and not busy else "disabled")
        if sys.platform == "darwin" and getattr(self, "reveal_output_btn", None):
            self.reveal_output_btn.configure_state("normal" if has_output_location and not busy else "disabled")

    def use_output_as_input(self):
        """Set the current output file as the input image for cleanup refinement."""
        out_path = (self.output_file.get() or "").strip()
        if not out_path or not os.path.isfile(out_path):
            messagebox.showerror("Error", "No saved output file found. Run Remove Background or Apply cleanup & save first.")
            return
        self.image_input_files = [out_path]
        self.input_file.set(out_path)
        self.load_input_preview(out_path)
        self._refresh_output_action_buttons()

    def save_output_as_copy(self):
        """Copy the current output file to a path the user chooses."""
        src = (self.output_file.get() or "").strip()
        if not src or not os.path.isfile(src):
            messagebox.showerror("Error", "There is no output file to save yet.")
            return
        dest = filedialog.asksaveasfilename(
            title="Save Output Copy As",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if not dest:
            return
        try:
            shutil.copy2(src, dest)
            self.output_file.set(dest)
            self._set_last_saved_output_hint(dest)
            self.load_output_preview()
            self.status_label.configure(
                text=f"Output copied to {dest}",
                foreground=ModernStyle.SUCCESS,
            )
            self._refresh_output_action_buttons()
        except OSError as exc:
            messagebox.showerror("Error", f"Could not save copy:\n{exc}")

    def reveal_output_in_finder(self):
        """Show the output file in Finder (macOS)."""
        path = (self.output_file.get() or "").strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "No output file to reveal.")
            return
        try:
            command = ["open", path] if os.path.isdir(path) else ["open", "-R", path]
            subprocess.run(command, check=False)
        except OSError as exc:
            messagebox.showerror("Error", f"Could not reveal file:\n{exc}")

    def apply_cleanup_and_save(self):
        """Apply color cleanup to the current input (RGBA) and save without running the AI model."""
        in_path = (self.input_file.get() or "").strip()
        out_path = (self.output_file.get() or "").strip()
        if not in_path or not os.path.isfile(in_path):
            messagebox.showerror("Error", "Select an input image file first.")
            return
        if self._is_image_batch_mode():
            messagebox.showerror("Error", "Apply cleanup & save is only available for one image at a time.")
            return
        if not out_path:
            messagebox.showerror("Error", "Specify an output file path.")
            return
        if self.processing:
            return
        self.processing = True
        self.process_btn.configure_state("disabled")
        self.sprite_process_btn.configure_state("disabled")
        self.apply_cleanup_save_btn.configure_state("disabled")
        self._refresh_image_cleanup_controls()
        self._refresh_output_action_buttons()
        self.progress.start(10)
        self.status_label.configure(text="Applying color cleanup…", foreground=ModernStyle.TEXT_SECONDARY)
        thread = threading.Thread(target=self._apply_cleanup_only_thread, daemon=True)
        thread.start()

    def _apply_cleanup_only_thread(self):
        try:
            img = Image.open(self.input_file.get())
            try:
                from PIL import ImageOps

                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            try:
                rgba = img.convert("RGBA")
                cutout = apply_color_cleanup(
                    rgba, self.image_cleanup_colors,
                    self.image_cleanup_threshold.get(),
                    protected_colors=self.image_protected_colors,
                    protected_threshold=self.image_protected_threshold.get(),
                )
                if self.auto_crop_output.get():
                    cutout = crop_to_visible_bounds(cutout)
                cutout.save(self.output_file.get(), "PNG")
            finally:
                img.close()
            self.root.after(0, self._on_cleanup_only_success)
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.root.after(0, lambda: self._on_cleanup_only_error(str(e)))

    def _on_cleanup_only_success(self):
        self.processing = False
        self.progress.stop()
        self.process_btn.configure_state("normal")
        self.sprite_process_btn.configure_state("normal")
        self.apply_cleanup_save_btn.configure_state("normal")
        self._refresh_image_cleanup_controls()
        self._refresh_output_action_buttons()
        self._set_last_saved_output_hint(self.output_file.get())
        self.load_output_preview()
        self.status_label.configure(
            text="Cleanup applied and saved.",
            foreground=ModernStyle.SUCCESS,
        )

    def _on_cleanup_only_error(self, error_msg):
        self.processing = False
        self.progress.stop()
        self.process_btn.configure_state("normal")
        self.sprite_process_btn.configure_state("normal")
        self.apply_cleanup_save_btn.configure_state("normal")
        self._refresh_image_cleanup_controls()
        self._refresh_output_action_buttons()
        self.status_label.configure(text="Cleanup save failed.", foreground=ModernStyle.ERROR)
        messagebox.showerror("Error", f"Apply cleanup failed:\n\n{error_msg}")
    
    def browse_input(self):
        """Open file dialog for input image"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.heic *.heif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("HEIC files", "*.heic *.heif"),
            ("All files", "*.*")
        ]
        
        filenames = filedialog.askopenfilenames(
            title="Select Input Image(s)",
            filetypes=filetypes
        )
        
        if filenames:
            paths = list(filenames)
            self.image_input_files = paths
            first_path = paths[0]
            self.input_file.set(first_path)
            self.load_input_preview(first_path)

            base, _ext = os.path.splitext(first_path)
            prefix = resolve_output_prefix(os.path.basename(base), "image")
            self.image_output_prefix.set(prefix)

            if len(paths) == 1:
                self.output_file.set(f"{base}_no_bg.png")
                self.status_label.configure(
                    text="1 image selected.",
                    foreground=ModernStyle.TEXT_SECONDARY,
                )
            else:
                self.output_file.set(os.path.join(os.path.expanduser("~"), "Downloads"))
                self.status_label.configure(
                    text=f"{len(paths)} images selected. They will process sequentially.",
                    foreground=ModernStyle.TEXT_SECONDARY,
                )
            self.output_preview_display_path = None
            self.load_output_preview()

    def browse_output(self):
        """Open file dialog for output location"""
        filetypes = [
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        if self._is_image_batch_mode():
            directory = filedialog.askdirectory(
                title="Choose Folder For Batch Images",
                initialdir=self.output_file.get() or os.path.join(os.path.expanduser("~"), "Downloads"),
            )

            if directory:
                self.output_file.set(directory)
                self.output_preview_display_path = None
                self.load_output_preview()
            return

        filename = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=".png",
            filetypes=filetypes
        )

        if filename:
            self.output_file.set(filename)
            self.output_preview_display_path = None
            self.load_output_preview()

    def browse_sprite_output(self):
        """Open file dialog for sprite-kit output folder."""
        directory = filedialog.askdirectory(
            title="Select Sprite Output Folder",
            initialdir=self.sprite_output_dir.get() or os.path.join(os.path.expanduser("~"), "Downloads"),
        )

        if directory:
            self.sprite_output_dir.set(directory)
    
    def load_input_preview(self, filepath):
        """Load and display input image preview on the zoomable canvas."""
        self.input_file.set(filepath)
        if not getattr(self, "image_input_files", None):
            self.image_input_files = [filepath]
        self._redraw_input_preview()
        self._refresh_output_action_buttons()

    def load_output_preview(self, filepath=None):
        """Redraw the output preview. Pass a path to preview that file without changing Output File (e.g. sprite). Pass None to show the current Output File path."""
        if filepath is not None:
            self.output_preview_display_path = filepath
        else:
            self.output_preview_display_path = None
        self._redraw_output_preview()
        self._refresh_output_action_buttons()
    
    def process_image(self):
        """Start image processing"""
        input_paths = self._selected_image_input_paths()
        if not input_paths:
            messagebox.showerror("Error", "Please select at least one input image.")
            return
        
        if not self.output_file.get():
            messagebox.showerror("Error", "Please specify an output file or batch folder.")
            return

        is_batch = len(input_paths) > 1
        if is_batch:
            target_dir = self.output_file.get()
            try:
                os.makedirs(target_dir, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("Error", f"Could not create output folder:\n{exc}")
                return
            if not self.image_output_prefix.get().strip():
                self.image_output_prefix.set(self._default_image_output_prefix())
        else:
            parent_dir = os.path.dirname(self.output_file.get())
            if parent_dir:
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                except OSError as exc:
                    messagebox.showerror("Error", f"Could not create output folder:\n{exc}")
                    return
        
        if self.processing:
            return
        
        self.processing = True
        self.process_btn.configure_state("disabled")
        self.sprite_process_btn.configure_state("disabled")
        if getattr(self, "apply_cleanup_save_btn", None):
            self.apply_cleanup_save_btn.configure_state("disabled")
        self.progress.start(10)
        if is_batch:
            self.status_label.configure(text=f"Processing 1/{len(input_paths)}... Please wait.")
        else:
            self.status_label.configure(text="Processing... Please wait.")
        self._refresh_image_cleanup_controls()
        self._refresh_output_action_buttons()

        thread = threading.Thread(target=self._process_thread, args=(input_paths,))
        thread.daemon = True
        thread.start()

    def process_sprite_kit(self):
        """Start smart sprite-kit export for the selected image."""
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input image.")
            return

        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Input file does not exist.")
            return

        if self.processing:
            return

        target_dir = self.sprite_output_dir.get() or os.path.join(os.path.expanduser("~"), "Downloads")
        os.makedirs(target_dir, exist_ok=True)

        self.processing = True
        self.process_btn.configure_state("disabled")
        self.sprite_process_btn.configure_state("disabled")
        if getattr(self, "apply_cleanup_save_btn", None):
            self.apply_cleanup_save_btn.configure_state("disabled")
        self._refresh_output_action_buttons()
        self.progress.start(10)
        self.status_label.configure(text="Building smart sprite kit... Please wait.")

        thread = threading.Thread(target=self._process_sprite_kit_thread, args=(target_dir,))
        thread.daemon = True
        thread.start()
    
    def _load_model(self, model_name):
        """Load a U2Net model with caching"""
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        
        from backgroundremover.u2net import u2net
        from backgroundremover import github
        
        if model_name == "u2netp":
            net = u2net.U2NETP(3, 1)
        else:
            net = u2net.U2NET(3, 1)
        
        path = os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth"))
        
        if not os.path.exists(path):
            github.download_files_from_github(path, model_name)
        
        if DEVICE.type == "cuda":
            net.load_state_dict(torch.load(path))
            net.to(torch.device("cuda"))
        elif DEVICE.type == "mps":
            net.load_state_dict(torch.load(path, map_location="mps"))
            net.to(torch.device("mps"))
        else:
            net.load_state_dict(torch.load(path, map_location="cpu"))
        
        net.eval()
        self.model_cache[model_name] = net
        return net
    
    def _predict(self, net, image_array):
        """Run inference on an image"""
        from torchvision import transforms
        from backgroundremover.u2net import data_loader
        
        label_3 = np.zeros(image_array.shape)
        label = np.zeros(label_3.shape[0:2])
        
        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3
        
        if 3 == len(image_array.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image_array.shape) and 2 == len(label.shape):
            image_array = image_array[:, :, np.newaxis]
            label = label[:, :, np.newaxis]
        
        transform = transforms.Compose(
            [data_loader.RescaleT(320), data_loader.ToTensorLab(flag=0)]
        )
        sample = transform({"imidx": np.array([0]), "image": image_array, "label": label})
        
        with torch.no_grad():
            if DEVICE.type == "cuda":
                inputs_test = torch.cuda.FloatTensor(sample["image"].unsqueeze(0).cuda().float())
            elif DEVICE.type == "mps":
                inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float()).to(DEVICE)
            else:
                inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())
            
            d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
            
            pred = d1[:, 0, :, :]
            ma = torch.max(pred)
            mi = torch.min(pred)
            predict = (pred - mi) / (ma - mi)
            
            predict = predict.squeeze()
            predict_np = predict.cpu().detach().numpy()
            mask = Image.fromarray(predict_np * 255).convert("L")
            
            del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test, sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return mask
    
    def _alpha_matting_cutout(self, img, mask, fg_threshold=240, bg_threshold=10, erode_size=10, base_size=1000):
        """Apply alpha matting for better edge quality"""
        from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
        from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
        from pymatting.util.util import stack_images
        from scipy.ndimage import binary_erosion
        
        size = img.size
        img.thumbnail((base_size, base_size), Image.Resampling.LANCZOS)
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)
        
        img_arr = np.asarray(img)
        mask_arr = np.asarray(mask)
        
        is_foreground = mask_arr > fg_threshold
        is_background = mask_arr < bg_threshold
        
        structure = None
        if erode_size > 0:
            structure = np.ones((erode_size, erode_size), dtype=np.int64)
        
        is_foreground = binary_erosion(is_foreground, structure=structure)
        is_background = binary_erosion(is_background, structure=structure, border_value=1)
        
        trimap = np.full(mask_arr.shape, dtype=np.uint8, fill_value=128)
        trimap[is_foreground] = 255
        trimap[is_background] = 0
        
        img_normalized = img_arr / 255.0
        trimap_normalized = trimap / 255.0
        
        alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
        foreground = estimate_foreground_ml(img_normalized, alpha)
        cutout = stack_images(foreground, alpha)
        
        cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
        cutout = Image.fromarray(cutout)
        cutout = cutout.resize(size, Image.Resampling.LANCZOS)
        
        return cutout
    
    def _naive_cutout(self, img, mask):
        """Simple cutout without alpha matting"""
        empty = Image.new("RGBA", img.size, 0)
        cutout = Image.composite(img, empty, mask.resize(img.size, Image.Resampling.LANCZOS))
        return cutout

    def _create_cutout_for_image(self, img, net, alpha_matting):
        """Run the existing background-removal pipeline for one PIL image."""
        mask = self._predict(net, np.array(img))
        if alpha_matting:
            return self._alpha_matting_cutout(img, mask)
        return self._naive_cutout(img, mask)

    def _resolved_video_output_prefix(self):
        """Resolve the current video export prefix with a fallback to the video basename."""
        fallback_name = os.path.splitext(os.path.basename(self.video_file.get()))[0] or "video"
        return resolve_output_prefix(self.video_output_prefix.get(), fallback_name)
    
    def _process_thread(self, input_paths=None):
        """Background processing thread"""
        try:
            input_paths = input_paths or self._selected_image_input_paths()
            if not input_paths:
                raise ValueError("No input images selected.")

            net = self._load_model(self.model_choice.get())
            saved_paths = []
            total = len(input_paths)
            output_prefix = self._resolved_image_output_prefix()

            for position, input_path in enumerate(input_paths, start=1):
                self.root.after(
                    0,
                    lambda current=position, t=total: self.status_label.configure(
                        text=f"Processing image {current}/{t}...",
                        foreground=ModernStyle.TEXT_SECONDARY,
                    ),
                )

                img = Image.open(input_path)
                try:
                    from PIL import ImageOps
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass

                try:
                    rgb_img = img.convert("RGB")
                    cutout = self._create_cutout_for_image(rgb_img, net, self.alpha_matting.get())
                    cutout = apply_color_cleanup(
                        cutout,
                        self.image_cleanup_colors,
                        self.image_cleanup_threshold.get(),
                        protected_colors=self.image_protected_colors,
                        protected_threshold=self.image_protected_threshold.get(),
                    )
                    if self.auto_crop_output.get():
                        cutout = crop_to_visible_bounds(cutout)

                    if total == 1:
                        destination = self.output_file.get()
                    else:
                        destination = os.path.join(
                            self.output_file.get(),
                            build_export_filename(output_prefix, position),
                        )

                    cutout.save(destination, "PNG")
                    saved_paths.append(destination)
                finally:
                    img.close()

            self.root.after(0, lambda: self._on_success(saved_paths))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._on_error(str(e)))

    def _process_sprite_kit_thread(self, target_dir):
        """Background processing thread for smart sprite-kit export."""
        try:
            prefix = resolve_output_prefix(
                os.path.splitext(os.path.basename(self.input_file.get()))[0],
                "sprite",
            )
            result = bg.create_sprite_kit(
                self.input_file.get(),
                destination_dir=target_dir,
                prefix=prefix,
                model_name=self.model_choice.get(),
                alpha_matting=self.alpha_matting.get(),
            )
            self.root.after(0, lambda: self._on_sprite_kit_success(result, target_dir))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._on_error(str(e)))
    
    def _on_success(self, saved_paths=None):
        """Called on successful processing"""
        saved_paths = saved_paths or [self.output_file.get()]
        self.processing = False
        self.progress.stop()
        self.process_btn.configure_state("normal")
        self.sprite_process_btn.configure_state("normal")
        if getattr(self, "apply_cleanup_save_btn", None):
            self.apply_cleanup_save_btn.configure_state("normal")
        self._refresh_image_cleanup_controls()
        self._refresh_output_action_buttons()

        if len(saved_paths) == 1:
            self.status_label.configure(text="✓ Background removed successfully!", foreground=ModernStyle.SUCCESS)
            self.output_file.set(saved_paths[0])
            self._set_last_saved_output_hint(saved_paths[0])
            self.load_output_preview()
            messagebox.showinfo("Success", f"Background removed!\n\nSaved to:\n{saved_paths[0]}")
            return

        target_dir = os.path.dirname(saved_paths[0]) if saved_paths else self.output_file.get()
        self.status_label.configure(
            text=f"✓ Backgrounds removed for {len(saved_paths)} image(s)!",
            foreground=ModernStyle.SUCCESS,
        )
        self._set_last_saved_output_hint(target_dir)
        if saved_paths:
            self.load_output_preview(saved_paths[0])
        messagebox.showinfo(
            "Success",
            f"Background removed for {len(saved_paths)} image(s)!\n\nSaved to:\n{target_dir}",
        )

    def _on_sprite_kit_success(self, result, target_dir):
        """Called on successful sprite-kit export."""
        self.processing = False
        self.progress.stop()
        self.process_btn.configure_state("normal")
        self.sprite_process_btn.configure_state("normal")
        self._refresh_image_cleanup_controls()
        self._refresh_output_action_buttons()
        sprite_count = int(result.get("sprite_count", 0))
        self.status_label.configure(
            text=f"✓ Smart sprite kit exported {sprite_count} sprite(s)!",
            foreground=ModernStyle.SUCCESS,
        )

        sprites = result.get("sprites", [])
        if sprites:
            first_sprite_path = os.path.join(target_dir, sprites[0]["filename"])
            self.load_output_preview(first_sprite_path)
        else:
            self.output_preview_display_path = None
            self._redraw_output_preview()

        manifest_path = result.get("manifest_path", "")
        messagebox.showinfo(
            "Success",
            f"Smart sprite kit exported {sprite_count} sprite(s) to:\n\n{target_dir}\n\nManifest:\n{manifest_path}",
        )
    
    def _on_error(self, error_msg):
        """Called on processing error"""
        self.processing = False
        self.progress.stop()
        self.process_btn.configure_state("normal")
        self.sprite_process_btn.configure_state("normal")
        if getattr(self, "apply_cleanup_save_btn", None):
            self.apply_cleanup_save_btn.configure_state("normal")
        self._refresh_image_cleanup_controls()
        self._refresh_output_action_buttons()
        self.status_label.configure(text="✗ Processing failed", foreground=ModernStyle.ERROR)

        messagebox.showerror("Error", f"Failed to process image:\n\n{error_msg}")
