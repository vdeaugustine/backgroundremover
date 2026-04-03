#!/usr/bin/env python3
"""
Background Remover - macOS Application
A modern, user-friendly GUI for removing backgrounds from images
and extracting frames from videos.
Optimized for Apple Silicon (M-series) with MPS acceleration.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import os
import sys
import shutil
import tempfile
from PIL import Image, ImageTk
import threading
import io
import numpy as np

DUPLICATE_FRAME_SIGNATURE_SIZE = (24, 24)
DUPLICATE_FRAME_MAX_MEAN_DIFFERENCE = 0.015
DEDUPLICATION_THRESHOLD_MIN = 0.001
DEDUPLICATION_THRESHOLD_MAX = 0.050
COLOR_CLEANUP_THRESHOLD_MIN = 0
COLOR_CLEANUP_THRESHOLD_MAX = 120
COLOR_CLEANUP_THRESHOLD_DEFAULT = 15

# Handle running from app bundle
if getattr(sys, 'frozen', False):
    # Running in a bundle
    bundle_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    os.chdir(bundle_dir)
    sys.path.insert(0, bundle_dir)

# Import torch first to detect device
import torch

# Detect device
try:
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
        DEVICE_NAME = "NVIDIA GPU"
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        DEVICE_NAME = "Apple Silicon GPU"
    else:
        DEVICE = torch.device('cpu')
        DEVICE_NAME = "CPU"
except Exception:
    DEVICE = torch.device('cpu')
    DEVICE_NAME = "CPU"


def build_frame_similarity_signature(image, size=DUPLICATE_FRAME_SIGNATURE_SIZE):
    """Create a small normalized grayscale representation for frame comparison."""
    grayscale = image.convert("L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(grayscale, dtype=np.float32) / 255.0


def dedupe_frame_items(frame_items, threshold=DUPLICATE_FRAME_MAX_MEAN_DIFFERENCE):
    """Keep only frames that differ meaningfully from the most recent unique frame."""
    if not frame_items:
        return []

    unique_items = [frame_items[0]]
    previous_signature = frame_items[0]["compare_array"]

    for item in frame_items[1:]:
        current_signature = item["compare_array"]
        mean_difference = float(np.mean(np.abs(current_signature - previous_signature)))
        if mean_difference > threshold:
            unique_items.append(item)
            previous_signature = current_signature

    return unique_items


def resolve_output_prefix(prefix, fallback_name):
    """Resolve a user-provided output prefix with a simple safe fallback."""
    cleaned = (prefix or "").strip()
    if not cleaned:
        cleaned = fallback_name or "video"
    return cleaned.replace(os.sep, "_")


def build_export_filename(prefix, sequence_number, suffix=""):
    """Build sequential export filenames for frame batch operations."""
    return f"{prefix}_{sequence_number}{suffix}.png"


def crop_to_visible_bounds(image):
    """Crop transparent outer bounds from an RGBA image, if any visible pixels exist."""
    bbox = image.getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)


def format_rgb_color(color):
    """Format an RGB tuple as a hex color string."""
    red, green, blue = (max(0, min(255, int(channel))) for channel in color)
    return f"#{red:02X}{green:02X}{blue:02X}"


def apply_color_cleanup(image, cleanup_colors, threshold=0):
    """Make pixels transparent when they match any selected cleanup color within tolerance."""
    rgba_image = image.convert("RGBA")
    if not cleanup_colors:
        return rgba_image

    pixel_array = np.array(rgba_image, dtype=np.uint8)
    if pixel_array.size == 0:
        return rgba_image

    opaque_mask = pixel_array[:, :, 3] > 0
    if not np.any(opaque_mask):
        return rgba_image

    threshold = max(0, int(threshold))
    rgb_values = pixel_array[:, :, :3].astype(np.int16)
    matched_mask = np.zeros(opaque_mask.shape, dtype=bool)

    for color in cleanup_colors:
        target = np.array([int(channel) for channel in color[:3]], dtype=np.int16)
        channel_difference = np.max(np.abs(rgb_values - target), axis=2)
        matched_mask |= channel_difference <= threshold

    matched_mask &= opaque_mask
    if not np.any(matched_mask):
        return rgba_image

    pixel_array[matched_mask, 3] = 0
    return Image.fromarray(pixel_array)


class ModernStyle:
    """Modern macOS-inspired styling"""
    # Colors
    BG_PRIMARY = "#000000"     # Pure black for true dark mode
    BG_SECONDARY = "#1c1c1e"   # macOS dark secondary
    BG_TERTIARY = "#2c2c2e"    # macOS dark tertiary
    ACCENT = "#2ecc71"         # Peter Pan Green
    ACCENT_HOVER = "#27ae60"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#a1a1a6"
    SUCCESS = "#32d74b"
    ERROR = "#ff453a"
    BORDER = "#3a3a3c"
    
    # Fonts
    FONT_TITLE = ("SF Pro Display", 28, "bold")
    FONT_SUBTITLE = ("SF Pro Display", 15)
    FONT_BODY = ("SF Pro Text", 13)
    FONT_SMALL = ("SF Pro Text", 11)
    FONT_BUTTON = ("SF Pro Text", 13, "bold")


class RoundedButton(tk.Canvas):
    """A custom rounded button using tk.Canvas for better macOS support"""
    def __init__(self, parent, text, command=None, width=120, height=35, radius=10, bg=None, fg="white", hover_bg=None, font=None, **kwargs):
        # Try to get parent's background color, fallback to ModernStyle.BG_PRIMARY
        parent_bg = ModernStyle.BG_PRIMARY
        try:
            if "bg" in parent.keys():
                parent_bg = parent["bg"]
            elif "background" in parent.keys():
                parent_bg = parent["background"]
        except Exception:
            pass
            
        super().__init__(parent, width=width, height=height, bg=parent_bg, highlightthickness=0, **kwargs)
        self.command = command
        self.radius = radius
        self.text_str = text
        self.bg_color = bg or ModernStyle.ACCENT
        self.hover_bg = hover_bg or ModernStyle.ACCENT_HOVER
        self.fg_color = fg
        self.font = font or ModernStyle.FONT_BUTTON
        self._disabled = False
        
        self.rect = self._draw_rounded_rect(0, 0, width, height, radius, fill=self.bg_color)
        self.text = self.create_text(width/2, height/2, text=text, fill=fg, font=self.font)
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_enter(self, event):
        if not self._disabled:
            self.itemconfig(self.rect, fill=self.hover_bg)

    def _on_leave(self, event):
        if not self._disabled:
            self.itemconfig(self.rect, fill=self.bg_color)

    def _on_click(self, event):
        if not self._disabled:
            self.move(self.rect, 1, 1)
            self.move(self.text, 1, 1)

    def _on_release(self, event):
        if not self._disabled:
            self.move(self.rect, -1, -1)
            self.move(self.text, -1, -1)
            if self.command:
                self.command()

    def configure_state(self, state):
        if state == "disabled" or state == tk.DISABLED:
            self._disabled = True
            self.itemconfig(self.rect, fill=ModernStyle.BG_TERTIARY)
            self.itemconfig(self.text, fill=ModernStyle.TEXT_SECONDARY)
        else:
            self._disabled = False
            self.itemconfig(self.rect, fill=self.bg_color)
            self.itemconfig(self.text, fill=self.fg_color)

    def set_text(self, text):
        self.itemconfig(self.text, text=text)


class BackgroundRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Background Remover")
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)
        self.root.configure(bg=ModernStyle.BG_PRIMARY)
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.model_choice = tk.StringVar(value="u2net")
        self.alpha_matting = tk.BooleanVar(value=False)
        self.processing = False
        self.video_processing = False
        
        # Model cache
        self.model_cache = {}
        
        # Image references (prevent garbage collection)
        self.input_photo = None
        self.output_photo = None
        self.frame_preview_photo = None

        # Video extraction state
        self.video_file = tk.StringVar()
        self.frame_output_dir = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Downloads"))
        self.use_custom_frame_output = tk.BooleanVar(value=False)
        self.frame_items = []
        self.current_frame_index = None
        self.frame_temp_dir = None
        self.video_clip_metadata = {}
        self.full_frame_count = 0
        self.all_extracted_frame_items = []
        self.video_bg_model_choice = tk.StringVar(value="u2net")
        self.video_bg_alpha_matting = tk.BooleanVar(value=False)
        self.video_output_prefix = tk.StringVar()
        self.deduplication_threshold = tk.DoubleVar(value=DUPLICATE_FRAME_MAX_MEAN_DIFFERENCE)
        self.video_cleanup_threshold = tk.IntVar(value=COLOR_CLEANUP_THRESHOLD_DEFAULT)
        self.video_cleanup_colors = []
        self.preview_color_pick_active = False
        
        # Configure styles
        self.setup_styles()
        
        # Create UI
        self.create_widgets()

        # Clean up temp frame exports on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Center window
        self.center_window()
    
    def setup_styles(self):
        """Configure ttk styles for modern look"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=ModernStyle.BG_PRIMARY)
        style.configure('TLabel', 
                       background=ModernStyle.BG_PRIMARY,
                       foreground=ModernStyle.TEXT_PRIMARY,
                       font=ModernStyle.FONT_BODY)
        style.configure('Title.TLabel',
                       background=ModernStyle.BG_PRIMARY,
                       foreground=ModernStyle.TEXT_PRIMARY,
                       font=ModernStyle.FONT_TITLE)
        style.configure('Subtitle.TLabel',
                       background=ModernStyle.BG_PRIMARY,
                       foreground=ModernStyle.TEXT_SECONDARY,
                       font=ModernStyle.FONT_SUBTITLE)
        style.configure('Small.TLabel',
                       background=ModernStyle.BG_PRIMARY,
                       foreground=ModernStyle.TEXT_SECONDARY,
                       font=ModernStyle.FONT_SMALL)
        style.configure('Success.TLabel',
                       background=ModernStyle.BG_PRIMARY,
                       foreground=ModernStyle.SUCCESS,
                       font=ModernStyle.FONT_BODY)
        style.configure('TProgressbar',
                       background=ModernStyle.ACCENT,
                       troughcolor=ModernStyle.BG_TERTIARY,
                       borderwidth=0,
                       thickness=10)
    
    def create_widgets(self):
        """Create the main UI"""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        image_tab = ttk.Frame(notebook, padding="20")
        video_tab = ttk.Frame(notebook, padding="20")
        notebook.add(image_tab, text="Background Remover")
        notebook.add(video_tab, text="Video Frames")

        self.create_image_tab(image_tab)
        self.create_video_tab(video_tab)

    def create_image_tab(self, parent):
        """Create the image background removal tab"""
        title_frame = ttk.Frame(parent)
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
        file_frame = ttk.Frame(parent)
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Input file
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Input Image").pack(anchor=tk.W)
        
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
        
        self.browse_input_btn = RoundedButton(input_entry_frame, text="Browse",
                                            command=self.browse_input,
                                            width=100, height=36,
                                            bg=ModernStyle.BG_TERTIARY,
                                            hover_bg=ModernStyle.BORDER)
        self.browse_input_btn.pack(side=tk.RIGHT)
        
        # Output file
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(output_frame, text="Output File").pack(anchor=tk.W)
        
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
        
        # Options section
        options_frame = ttk.Frame(parent)
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
        
        # Process button
        self.process_btn = RoundedButton(parent, text="Remove Background",
                                        command=self.process_image,
                                        width=280, height=50,
                                        radius=15,
                                        bg=ModernStyle.ACCENT,
                                        hover_bg=ModernStyle.ACCENT_HOVER,
                                        font=ModernStyle.FONT_BUTTON)
        self.process_btn.pack(pady=(10, 20))
        
        # Progress bar
        self.progress = ttk.Progressbar(parent, mode='indeterminate', style='TProgressbar')
        self.progress.pack(fill=tk.X, pady=(0, 5))
        
        # Status label
        self.status_label = ttk.Label(parent, text="Ready", style='Small.TLabel')
        self.status_label.pack(pady=(0, 20))
        
        # Preview section
        preview_frame = ttk.Frame(parent)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input preview
        input_preview_frame = tk.Frame(preview_frame, bg=ModernStyle.BG_SECONDARY)
        input_preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        tk.Label(input_preview_frame, text="Input", 
                bg=ModernStyle.BG_SECONDARY, fg=ModernStyle.TEXT_SECONDARY,
                font=ModernStyle.FONT_SMALL).pack(pady=(10, 5))
        
        self.input_preview = tk.Label(input_preview_frame, 
                                     text="No image selected",
                                     bg=ModernStyle.BG_TERTIARY,
                                     fg=ModernStyle.TEXT_SECONDARY,
                                     font=ModernStyle.FONT_SMALL)
        self.input_preview.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Output preview
        output_preview_frame = tk.Frame(preview_frame, bg=ModernStyle.BG_SECONDARY)
        output_preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        tk.Label(output_preview_frame, text="Output",
                bg=ModernStyle.BG_SECONDARY, fg=ModernStyle.TEXT_SECONDARY,
                font=ModernStyle.FONT_SMALL).pack(pady=(10, 5))
        
        self.output_preview = tk.Label(output_preview_frame,
                                      text="No output yet",
                                      bg=ModernStyle.BG_TERTIARY,
                                      fg=ModernStyle.TEXT_SECONDARY,
                                      font=ModernStyle.FONT_SMALL)
        self.output_preview.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def create_video_tab(self, parent):
        """Create the video frame extraction tab"""
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(title_frame, text="Video Frames", style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(
            title_frame,
            text="Extract every frame from a video, review them in a sidebar, and save only the ones you want",
            style='Subtitle.TLabel',
        ).pack(anchor=tk.W, pady=(5, 0))

        upload_frame = tk.Frame(
            parent,
            bg=ModernStyle.BG_SECONDARY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
        )
        upload_frame.pack(fill=tk.X, pady=(0, 15))
        upload_frame.bind("<Button-1>", lambda _event: self.browse_video())

        upload_label = tk.Label(
            upload_frame,
            text="Click to choose a video for frame extraction",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_SUBTITLE,
            pady=20,
            cursor="hand2",
        )
        upload_label.pack()
        upload_label.bind("<Button-1>", lambda _event: self.browse_video())

        video_path_frame = ttk.Frame(parent)
        video_path_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(video_path_frame, text="Video File").pack(anchor=tk.W)

        video_entry_row = ttk.Frame(video_path_frame)
        video_entry_row.pack(fill=tk.X, pady=(5, 0))

        self.video_entry = tk.Entry(
            video_entry_row,
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
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))

        self.browse_video_btn = RoundedButton(
            video_entry_row,
            text="Upload",
            command=self.browse_video,
            width=100,
            height=36,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.browse_video_btn.pack(side=tk.RIGHT)

        save_frame = ttk.Frame(parent)
        save_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(save_frame, text="Save Location").pack(anchor=tk.W)

        save_options_row = ttk.Frame(save_frame)
        save_options_row.pack(fill=tk.X, pady=(8, 0))

        tk.Checkbutton(
            save_options_row,
            text="Choose custom folder",
            variable=self.use_custom_frame_output,
            command=self._toggle_frame_output_mode,
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_PRIMARY,
            selectcolor=ModernStyle.BG_TERTIARY,
            activebackground=ModernStyle.BG_PRIMARY,
            activeforeground=ModernStyle.ACCENT,
            font=ModernStyle.FONT_BODY,
        ).pack(side=tk.LEFT, padx=(0, 12))

        self.frame_output_entry = tk.Entry(
            save_options_row,
            textvariable=self.frame_output_dir,
            state="disabled",
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            disabledbackground=ModernStyle.BG_TERTIARY,
            disabledforeground=ModernStyle.TEXT_SECONDARY,
            insertbackground=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            font=ModernStyle.FONT_BODY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
            highlightcolor=ModernStyle.ACCENT,
        )
        self.frame_output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))

        self.browse_frame_output_btn = RoundedButton(
            save_options_row,
            text="Browse",
            command=self.browse_frame_output,
            width=100,
            height=36,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.browse_frame_output_btn.pack(side=tk.RIGHT)
        self.browse_frame_output_btn.configure_state("disabled")

        prefix_frame = ttk.Frame(parent)
        prefix_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(prefix_frame, text="Output Name Prefix").pack(anchor=tk.W)

        self.video_output_prefix_entry = tk.Entry(
            prefix_frame,
            textvariable=self.video_output_prefix,
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_PRIMARY,
            insertbackground=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT,
            font=ModernStyle.FONT_BODY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
            highlightcolor=ModernStyle.ACCENT,
        )
        self.video_output_prefix_entry.pack(fill=tk.X, pady=(5, 0), ipady=8)

        dedupe_frame = ttk.Frame(parent)
        dedupe_frame.pack(fill=tk.X, pady=(0, 15))

        threshold_header = ttk.Frame(dedupe_frame)
        threshold_header.pack(fill=tk.X)
        ttk.Label(threshold_header, text="Duplicate Threshold").pack(side=tk.LEFT)
        self.deduplication_threshold_value_label = ttk.Label(
            threshold_header,
            text=f"{self.deduplication_threshold.get():.3f}",
            style='Small.TLabel',
        )
        self.deduplication_threshold_value_label.pack(side=tk.RIGHT)

        self.deduplication_threshold_scale = tk.Scale(
            dedupe_frame,
            from_=DEDUPLICATION_THRESHOLD_MIN,
            to=DEDUPLICATION_THRESHOLD_MAX,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=self.deduplication_threshold,
            command=self._on_deduplication_threshold_change,
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_PRIMARY,
            troughcolor=ModernStyle.BG_TERTIARY,
            activebackground=ModernStyle.ACCENT,
            highlightthickness=0,
            font=ModernStyle.FONT_SMALL,
        )
        self.deduplication_threshold_scale.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(
            dedupe_frame,
            text="Lower keeps more frames. Higher removes more near-identical frames.",
            style='Small.TLabel',
        ).pack(anchor=tk.W, pady=(4, 0))

        cleanup_frame = ttk.Frame(parent)
        cleanup_frame.pack(fill=tk.X, pady=(0, 15))

        cleanup_header = ttk.Frame(cleanup_frame)
        cleanup_header.pack(fill=tk.X)
        ttk.Label(cleanup_header, text="Final Color Cleanup").pack(side=tk.LEFT)
        self.video_cleanup_threshold_value_label = ttk.Label(
            cleanup_header,
            text=str(self.video_cleanup_threshold.get()),
            style='Small.TLabel',
        )
        self.video_cleanup_threshold_value_label.pack(side=tk.RIGHT)

        self.video_cleanup_threshold_scale = tk.Scale(
            cleanup_frame,
            from_=COLOR_CLEANUP_THRESHOLD_MIN,
            to=COLOR_CLEANUP_THRESHOLD_MAX,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.video_cleanup_threshold,
            command=self._on_video_cleanup_threshold_change,
            bg=ModernStyle.BG_PRIMARY,
            fg=ModernStyle.TEXT_PRIMARY,
            troughcolor=ModernStyle.BG_TERTIARY,
            activebackground=ModernStyle.ACCENT,
            highlightthickness=0,
            font=ModernStyle.FONT_SMALL,
        )
        self.video_cleanup_threshold_scale.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(
            cleanup_frame,
            text="Sample leftover chroma colors from the selected-frame preview or add one manually. Matching pixels become transparent after AI removal and before save.",
            style='Small.TLabel',
        ).pack(anchor=tk.W, pady=(4, 0))

        cleanup_button_row = ttk.Frame(cleanup_frame)
        cleanup_button_row.pack(fill=tk.X, pady=(10, 8))

        self.pick_cleanup_color_btn = RoundedButton(
            cleanup_button_row,
            text="Sample From Preview",
            command=self._toggle_preview_color_pick,
            width=180,
            height=38,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.pick_cleanup_color_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.add_cleanup_color_btn = RoundedButton(
            cleanup_button_row,
            text="Add Color...",
            command=self._choose_video_cleanup_color,
            width=140,
            height=38,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.add_cleanup_color_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_cleanup_colors_btn = RoundedButton(
            cleanup_button_row,
            text="Clear Colors",
            command=self._clear_video_cleanup_colors,
            width=130,
            height=38,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.clear_cleanup_colors_btn.pack(side=tk.LEFT)

        self.video_cleanup_colors_label = ttk.Label(
            cleanup_frame,
            text="No cleanup colors selected.",
            style='Small.TLabel',
        )
        self.video_cleanup_colors_label.pack(anchor=tk.W)

        controls_row = ttk.Frame(parent)
        controls_row.pack(fill=tk.X, pady=(0, 15))

        self.extract_frames_btn = RoundedButton(
            controls_row,
            text="Extract Frames",
            command=self.extract_frames,
            width=170,
            height=42,
            bg=ModernStyle.ACCENT,
            hover_bg=ModernStyle.ACCENT_HOVER,
        )
        self.extract_frames_btn.pack(side=tk.LEFT, padx=(0, 12))

        self.select_all_btn = RoundedButton(
            controls_row,
            text="Select All",
            command=self.select_all_frames,
            width=120,
            height=42,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.select_all_btn.pack(side=tk.LEFT, padx=(0, 12))
        self.select_all_btn.configure_state("disabled")

        self.remove_duplicates_btn = RoundedButton(
            controls_row,
            text="Remove Duplicates",
            command=self.remove_duplicate_frames,
            width=170,
            height=42,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.remove_duplicates_btn.pack(side=tk.LEFT, padx=(0, 12))
        self.remove_duplicates_btn.configure_state("disabled")

        self.clear_selection_btn = RoundedButton(
            controls_row,
            text="Select None",
            command=self.clear_frame_selection,
            width=120,
            height=42,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.clear_selection_btn.pack(side=tk.LEFT, padx=(0, 12))
        self.clear_selection_btn.configure_state("disabled")

        self.save_frames_btn = RoundedButton(
            controls_row,
            text="Save Selected",
            command=self.save_selected_frames,
            width=150,
            height=42,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.save_frames_btn.pack(side=tk.LEFT, padx=(0, 12))
        self.save_frames_btn.configure_state("disabled")

        self.remove_bg_frames_btn = RoundedButton(
            controls_row,
            text="Remove Background + Save",
            command=self.remove_background_and_save_selected_frames,
            width=230,
            height=42,
            bg=ModernStyle.ACCENT,
            hover_bg=ModernStyle.ACCENT_HOVER,
        )
        self.remove_bg_frames_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.remove_bg_frames_btn.configure_state("disabled")

        self.remove_bg_frames_options_btn = RoundedButton(
            controls_row,
            text="v",
            command=self.open_video_background_options,
            width=42,
            height=42,
            bg=ModernStyle.BG_TERTIARY,
            hover_bg=ModernStyle.BORDER,
        )
        self.remove_bg_frames_options_btn.pack(side=tk.LEFT)
        self.remove_bg_frames_options_btn.configure_state("disabled")

        self.video_progress = ttk.Progressbar(parent, mode='indeterminate', style='TProgressbar')
        self.video_progress.pack(fill=tk.X, pady=(0, 5))

        self.video_status_label = ttk.Label(
            parent,
            text=f"Ready. Saved frames default to {self.frame_output_dir.get()}",
            style='Small.TLabel',
        )
        self.video_status_label.pack(anchor=tk.W, pady=(0, 15))

        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True)

        sidebar_container = tk.Frame(
            content_frame,
            bg=ModernStyle.BG_SECONDARY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
        )
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar_container.configure(width=280)
        sidebar_container.pack_propagate(False)

        tk.Label(
            sidebar_container,
            text="Frames",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
        ).pack(anchor=tk.W, padx=12, pady=(12, 8))

        self.frame_list_canvas = tk.Canvas(
            sidebar_container,
            bg=ModernStyle.BG_SECONDARY,
            highlightthickness=0,
            bd=0,
            width=280,
        )
        self.frame_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 8))

        frame_scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=self.frame_list_canvas.yview)
        frame_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=(0, 8))
        self.frame_list_canvas.configure(yscrollcommand=frame_scrollbar.set)

        self.frame_list_inner = tk.Frame(self.frame_list_canvas, bg=ModernStyle.BG_SECONDARY)
        self.frame_list_canvas_window = self.frame_list_canvas.create_window(
            (0, 0),
            window=self.frame_list_inner,
            anchor="nw",
        )
        self.frame_list_inner.bind("<Configure>", self._on_frame_list_configure)
        self.frame_list_canvas.bind("<Configure>", self._on_frame_list_canvas_resize)
        self._bind_frame_list_scrolling()

        preview_container = tk.Frame(
            content_frame,
            bg=ModernStyle.BG_SECONDARY,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER,
        )
        preview_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            preview_container,
            text="Selected Frame Preview",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
        ).pack(anchor=tk.W, padx=14, pady=(12, 8))

        self.frame_preview = tk.Label(
            preview_container,
            text="Extract frames to start reviewing them",
            bg=ModernStyle.BG_TERTIARY,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_BODY,
        )
        self.frame_preview.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))
        self.frame_preview.bind("<Button-1>", self._on_frame_preview_click)

        self.frame_preview_meta = tk.Label(
            preview_container,
            text="",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_BODY,
            anchor="w",
            justify=tk.LEFT,
        )
        self.frame_preview_meta.pack(fill=tk.X, padx=14, pady=(0, 14))
        self._refresh_video_cleanup_controls()
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')

    def on_close(self):
        """Clean up temporary files before closing the app"""
        self._cleanup_frame_temp_dir()
        self.root.destroy()

    def _toggle_frame_output_mode(self):
        """Enable or disable custom frame output selection"""
        if self.use_custom_frame_output.get():
            self.frame_output_entry.configure(state="normal")
            self.browse_frame_output_btn.configure_state("normal")
            self.video_status_label.configure(
                text=f"Custom frame save folder: {self.frame_output_dir.get()}",
                foreground=ModernStyle.TEXT_SECONDARY,
            )
        else:
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            self.frame_output_dir.set(downloads_dir)
            self.frame_output_entry.configure(state="disabled")
            self.browse_frame_output_btn.configure_state("disabled")
            self.video_status_label.configure(
                text=f"Ready. Saved frames default to {downloads_dir}",
                foreground=ModernStyle.TEXT_SECONDARY,
            )

    def _on_deduplication_threshold_change(self, _value=None):
        """Refresh the threshold value label when the dedupe slider moves."""
        self.deduplication_threshold_value_label.configure(
            text=f"{self.deduplication_threshold.get():.3f}"
        )

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
        self.frame_preview.configure(image="", text="Extract frames to start reviewing them")
        self.frame_preview_meta.configure(text="")
        self.select_all_btn.configure_state("disabled")
        self.remove_duplicates_btn.configure_state("disabled")
        self.clear_selection_btn.configure_state("disabled")
        self.save_frames_btn.configure_state("disabled")
        self.remove_bg_frames_btn.configure_state("disabled")
        self.remove_bg_frames_options_btn.configure_state("disabled")
        self.preview_color_pick_active = False
        if clear_cleanup_colors:
            self.video_cleanup_colors = []
        self._refresh_video_cleanup_controls()
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

        self._rebuild_frame_list(frame_items)
        self.all_extracted_frame_items = [dict(item) for item in frame_items]

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

    def _rebuild_frame_list(self, frame_items):
        """Replace the sidebar with a new set of frame items."""
        for child in self.frame_list_inner.winfo_children():
            child.destroy()

        self.frame_items = []
        self.current_frame_index = None
        self.frame_preview_photo = None
        self.frame_preview.configure(image="", text="Extract frames to start reviewing them")
        self.frame_preview_meta.configure(text="")
        self.preview_color_pick_active = False
        self._refresh_video_cleanup_controls()

        for item in frame_items:
            frame_item = dict(item)
            frame_item["selected_var"] = tk.BooleanVar(value=False)
            frame_item["widget"] = None
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
            self.clear_selection_btn.configure_state("disabled")
            self.save_frames_btn.configure_state("disabled")
            self.remove_bg_frames_btn.configure_state("disabled")
            self.remove_bg_frames_options_btn.configure_state("disabled")
            self.pick_cleanup_color_btn.configure_state("disabled")
            self.add_cleanup_color_btn.configure_state("disabled")
            self.clear_cleanup_colors_btn.configure_state("disabled")
            return

        self.extract_frames_btn.configure_state("normal")
        has_frames = bool(self.frame_items)
        has_selected_frames = has_frames and self._selected_frame_count() > 0
        self.select_all_btn.configure_state("normal" if has_frames else "disabled")
        self.remove_duplicates_btn.configure_state("normal" if len(self.all_extracted_frame_items) > 1 else "disabled")
        self.clear_selection_btn.configure_state("normal" if has_frames else "disabled")
        self.save_frames_btn.configure_state("normal" if has_selected_frames else "disabled")
        self.remove_bg_frames_btn.configure_state("normal" if has_selected_frames else "disabled")
        self.remove_bg_frames_options_btn.configure_state("normal" if has_selected_frames else "disabled")
        self.pick_cleanup_color_btn.configure_state("normal" if self.current_frame_index is not None else "disabled")
        self.add_cleanup_color_btn.configure_state("normal")
        self.clear_cleanup_colors_btn.configure_state("normal" if self.video_cleanup_colors else "disabled")

    def _on_video_cleanup_threshold_change(self, _value):
        self.video_cleanup_threshold_value_label.configure(text=str(self.video_cleanup_threshold.get()))
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)
        else:
            self._refresh_video_cleanup_controls()

    def _cleanup_colors_summary(self):
        if not self.video_cleanup_colors:
            return "None selected"

        swatches = [format_rgb_color(color) for color in self.video_cleanup_colors[:4]]
        if len(self.video_cleanup_colors) > 4:
            swatches.append(f"+{len(self.video_cleanup_colors) - 4} more")
        return ", ".join(swatches)

    def _refresh_video_cleanup_controls(self):
        if self.preview_color_pick_active:
            self.pick_cleanup_color_btn.set_text("Click Preview To Sample")
            self.frame_preview.configure(cursor="crosshair")
        else:
            self.pick_cleanup_color_btn.set_text("Sample From Preview")
            self.frame_preview.configure(cursor="")

        self.video_cleanup_colors_label.configure(
            text=f"Cleanup colors: {self._cleanup_colors_summary()}. Tolerance: {self.video_cleanup_threshold.get()}"
        )

        if self.video_processing:
            self.pick_cleanup_color_btn.configure_state("disabled")
            self.add_cleanup_color_btn.configure_state("disabled")
            self.clear_cleanup_colors_btn.configure_state("disabled")
            return

        self.pick_cleanup_color_btn.configure_state("normal" if self.current_frame_index is not None else "disabled")
        self.add_cleanup_color_btn.configure_state("normal")
        self.clear_cleanup_colors_btn.configure_state("normal" if self.video_cleanup_colors else "disabled")

    def _toggle_preview_color_pick(self):
        if self.current_frame_index is None:
            messagebox.showerror("Error", "Select a frame preview before sampling a cleanup color.")
            return

        self.preview_color_pick_active = not self.preview_color_pick_active
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

    def _add_video_cleanup_color(self, color):
        normalized_color = tuple(max(0, min(255, int(channel))) for channel in color[:3])
        if normalized_color not in self.video_cleanup_colors:
            self.video_cleanup_colors.append(normalized_color)

        self.preview_color_pick_active = False
        self._refresh_video_cleanup_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _clear_video_cleanup_colors(self):
        self.video_cleanup_colors = []
        self.preview_color_pick_active = False
        self._refresh_video_cleanup_controls()
        if self.current_frame_index is not None:
            self._show_frame_preview(self.current_frame_index)

    def _current_frame_item(self):
        return next((item for item in self.frame_items if item["index"] == self.current_frame_index), None)

    def _on_frame_preview_click(self, event):
        if not self.preview_color_pick_active or self.current_frame_index is None or self.frame_preview_photo is None:
            return

        current_item = self._current_frame_item()
        if current_item is None:
            return

        display_width = self.frame_preview_photo.width()
        display_height = self.frame_preview_photo.height()
        widget_width = max(self.frame_preview.winfo_width(), display_width)
        widget_height = max(self.frame_preview.winfo_height(), display_height)
        x_offset = max((widget_width - display_width) // 2, 0)
        y_offset = max((widget_height - display_height) // 2, 0)

        local_x = event.x - x_offset
        local_y = event.y - y_offset
        if local_x < 0 or local_y < 0 or local_x >= display_width or local_y >= display_height:
            return

        with Image.open(current_item["path"]) as opened_image:
            source_image = opened_image.convert("RGB")
            source_x = min(source_image.width - 1, max(0, int(local_x * source_image.width / display_width)))
            source_y = min(source_image.height - 1, max(0, int(local_y * source_image.height / display_height)))
            sampled_color = source_image.getpixel((source_x, source_y))

        self._add_video_cleanup_color(sampled_color)
        self.video_status_label.configure(
            text=f"Added cleanup color {format_rgb_color(sampled_color)}. Matching pixels will be removed on the final save pass.",
            foreground=ModernStyle.SUCCESS,
        )

    def remove_duplicate_frames(self):
        """Start duplicate-frame removal in a background thread."""
        if len(self.all_extracted_frame_items) <= 1 or self.video_processing:
            return

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

        self._rebuild_frame_list(unique_items)
        self.video_status_label.configure(
            text=(
                f"{self.full_frame_count} frames extracted, {len(self.frame_items)} unique frames kept "
                f"at threshold {self.deduplication_threshold.get():.3f}."
            ),
            foreground=ModernStyle.SUCCESS,
        )

    def _on_duplicate_removal_error(self, error_msg):
        """Handle duplicate-removal failure without losing extracted frames."""
        self.video_processing = False
        self.video_progress.stop()
        self._set_video_action_states(is_busy=False)
        self.video_status_label.configure(
            text="Removing duplicate frames failed.",
            foreground=ModernStyle.ERROR,
        )
        messagebox.showerror("Error", f"Failed to remove duplicate frames:\n\n{error_msg}")

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
        """Display the selected frame larger on the right side"""
        matching = next((item for item in self.frame_items if item["index"] == index), None)
        if matching is None:
            return

        self.current_frame_index = index
        with Image.open(matching["path"]) as opened_image:
            image = opened_image.copy()
        image.thumbnail((700, 520), Image.Resampling.LANCZOS)
        self.frame_preview_photo = ImageTk.PhotoImage(image)
        self.frame_preview.configure(image=self.frame_preview_photo, text="")

        selected_count = self._selected_frame_count()
        width, height = matching["size"]
        meta_lines = [
            f"{matching['name']}",
            f"Resolution: {width} x {height}",
            f"Selected for saving: {'Yes' if matching['selected_var'].get() else 'No'}",
            f"Total selected: {selected_count}",
            f"Cleanup colors: {self._cleanup_colors_summary()}",
            f"Cleanup tolerance: {self.video_cleanup_threshold.get()}",
        ]
        if self.preview_color_pick_active:
            meta_lines.append("Sampling mode: click this preview to add one cleanup color.")
        self.frame_preview_meta.configure(text="\n".join(meta_lines))
        self._refresh_frame_highlight()
        self._refresh_video_cleanup_controls()

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

        thread = threading.Thread(
            target=self._save_selected_frames_thread,
            args=(selected_items, target_dir),
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
        ).pack(anchor=tk.W, pady=(0, 20))

        button_row = ttk.Frame(container)
        button_row.pack(fill=tk.X)

        def start_with_settings():
            self.video_bg_model_choice.set(model_var.get())
            self.video_bg_alpha_matting.set(alpha_var.get())
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

    def remove_background_and_save_selected_frames(self):
        """Apply background removal to selected frames and save the results."""
        if not self.frame_items:
            messagebox.showerror("Error", "Extract frames before trying to save them.")
            return

        selected_items = [item for item in self.frame_items if item["selected_var"].get()]
        if not selected_items:
            messagebox.showerror("Error", "Select at least one frame to process.")
            return

        if self.video_processing:
            return

        target_dir = self.frame_output_dir.get() or os.path.join(os.path.expanduser("~"), "Downloads")
        os.makedirs(target_dir, exist_ok=True)
        output_prefix = self._resolved_video_output_prefix()

        self.video_processing = True
        self.video_progress.start(10)
        self._set_video_action_states(is_busy=True)
        self.video_status_label.configure(
            text=(
                f"Removing backgrounds from {len(selected_items)} selected frame(s)"
                f"{' with final color cleanup' if self.video_cleanup_colors else ''}..."
            ),
            foreground=ModernStyle.TEXT_SECONDARY,
        )

        thread = threading.Thread(
            target=self._remove_background_and_save_selected_frames_thread,
            args=(
                selected_items,
                target_dir,
                output_prefix,
                self.video_bg_model_choice.get(),
                self.video_bg_alpha_matting.get(),
                list(self.video_cleanup_colors),
                self.video_cleanup_threshold.get(),
            ),
            daemon=True,
        )
        thread.start()

    def _remove_background_and_save_selected_frames_thread(self, selected_items, target_dir, output_prefix, model_name, alpha_matting, cleanup_colors=None, cleanup_threshold=0):
        """Batch-remove backgrounds for selected frame images and save them as PNGs."""
        try:
            saved_paths = []
            net = self._load_model(model_name)

            for position, item in enumerate(selected_items, start=1):
                self.root.after(
                    0,
                    lambda current=position, total=len(selected_items): self.video_status_label.configure(
                        text=f"Removing backgrounds... {current}/{total}",
                        foreground=ModernStyle.TEXT_SECONDARY,
                    ),
                )
                with Image.open(item["path"]) as opened_image:
                    img = opened_image.convert("RGB")
                cutout = self._create_cutout_for_image(img, net, alpha_matting)
                cutout = apply_color_cleanup(cutout, cleanup_colors or [], cleanup_threshold)
                cutout = crop_to_visible_bounds(cutout)
                destination = os.path.join(
                    target_dir,
                    build_export_filename(output_prefix, position, suffix="_no_bg"),
                )
                cutout.save(destination, "PNG")
                saved_paths.append(destination)

            self.root.after(0, lambda: self._on_background_frames_saved(saved_paths, target_dir))
        except Exception as e:
            self.root.after(0, lambda: self._on_background_frames_save_error(str(e)))

    def _save_selected_frames_thread(self, selected_items, target_dir):
        """Copy selected frame PNGs to the final output folder"""
        try:
            saved_paths = []
            output_prefix = self._resolved_video_output_prefix()

            for position, item in enumerate(selected_items, start=1):
                destination = os.path.join(
                    target_dir,
                    build_export_filename(output_prefix, position),
                )
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
        """Handle frame export failure"""
        self.video_processing = False
        self.video_progress.stop()
        self._set_video_action_states(is_busy=False)
        self.video_status_label.configure(
            text="Saving selected frames failed.",
            foreground=ModernStyle.ERROR,
        )
        messagebox.showerror("Error", f"Failed to save selected frames:\n\n{error_msg}")

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
    
    def browse_input(self):
        """Open file dialog for input image"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.heic *.heif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("HEIC files", "*.heic *.heif"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=filetypes
        )
        
        if filename:
            self.input_file.set(filename)
            self.load_input_preview(filename)
            
            # Always update output filename when input changes
            base, ext = os.path.splitext(filename)
            self.output_file.set(f"{base}_no_bg.png")
    
    def browse_output(self):
        """Open file dialog for output location"""
        filetypes = [
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=".png",
            filetypes=filetypes
        )
        
        if filename:
            self.output_file.set(filename)
    
    def load_input_preview(self, filepath):
        """Load and display input image preview"""
        try:
            image = Image.open(filepath)
            image.thumbnail((250, 200), Image.Resampling.LANCZOS)
            
            self.input_photo = ImageTk.PhotoImage(image)
            self.input_preview.configure(image=self.input_photo, text="")
        except Exception as e:
            self.input_preview.configure(image="", text=f"Error: {str(e)[:30]}...")
    
    def load_output_preview(self, filepath):
        """Load and display output image preview"""
        try:
            if os.path.exists(filepath):
                image = Image.open(filepath)
                image.thumbnail((250, 200), Image.Resampling.LANCZOS)
                
                self.output_photo = ImageTk.PhotoImage(image)
                self.output_preview.configure(image=self.output_photo, text="")
        except Exception as e:
            self.output_preview.configure(image="", text=f"Error: {str(e)[:30]}...")
    
    def process_image(self):
        """Start image processing"""
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input image.")
            return
        
        if not self.output_file.get():
            messagebox.showerror("Error", "Please specify an output file.")
            return
        
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Input file does not exist.")
            return
        
        if self.processing:
            return
        
        self.processing = True
        self.process_btn.configure_state("disabled")
        self.progress.start(10)
        self.status_label.configure(text="Processing... Please wait.")
        
        thread = threading.Thread(target=self._process_thread)
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
    
    def _process_thread(self):
        """Background processing thread"""
        try:
            # Load image
            img = Image.open(self.input_file.get())
            try:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            img = img.convert("RGB")
            
            # Load model
            net = self._load_model(self.model_choice.get())
            
            cutout = self._create_cutout_for_image(img, net, self.alpha_matting.get())
            
            # Save
            cutout.save(self.output_file.get(), "PNG")
            
            self.root.after(0, self._on_success)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._on_error(str(e)))
    
    def _on_success(self):
        """Called on successful processing"""
        self.processing = False
        self.progress.stop()
        self.process_btn.configure_state("normal")
        self.status_label.configure(text="✓ Background removed successfully!", foreground=ModernStyle.SUCCESS)
        
        self.load_output_preview(self.output_file.get())
        
        messagebox.showinfo("Success", f"Background removed!\n\nSaved to:\n{self.output_file.get()}")
    
    def _on_error(self, error_msg):
        """Called on processing error"""
        self.processing = False
        self.progress.stop()
        self.process_btn.configure_state("normal")
        self.status_label.configure(text="✗ Processing failed", foreground=ModernStyle.ERROR)
        
        messagebox.showerror("Error", f"Failed to process image:\n\n{error_msg}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = BackgroundRemoverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
