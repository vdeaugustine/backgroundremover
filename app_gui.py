#!/usr/bin/env python3
"""
Background Remover - macOS Application
A modern, user-friendly GUI for removing backgrounds from images
and extracting frames from videos.
Optimized for Apple Silicon (M-series) with MPS acceleration.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
import sys
import shutil
import tempfile
from PIL import Image, ImageTk
import threading
import io
import json
import subprocess
import numpy as np
from backgroundremover import bg

DUPLICATE_FRAME_SIGNATURE_SIZE = (24, 24)
DUPLICATE_FRAME_MAX_MEAN_DIFFERENCE = 0.015
DEDUPLICATION_THRESHOLD_MIN = 0.001
DEDUPLICATION_THRESHOLD_MAX = 0.050
COLOR_CLEANUP_THRESHOLD_MIN = 0
COLOR_CLEANUP_THRESHOLD_MAX = 120
COLOR_CLEANUP_THRESHOLD_DEFAULT = 15
MAX_COLOR_CLEANUP_WORKERS = 4
MAX_PENDING_COLOR_CLEANUP_TASKS_PER_WORKER = 2

PREVIEW_BASE_WIDTH = 250
PREVIEW_BASE_HEIGHT = 200
PREVIEW_ZOOM_MIN = 0.25
PREVIEW_ZOOM_MAX = 4.0
PREVIEW_ZOOM_STEP = 1.15
PREVIEW_MAX_DISPLAY_SIDE = 1800
COLOR_SWATCH_SIZE = 18
COLOR_SWATCH_MAX_VISIBLE = 12

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


def _resolve_window_icon_png_path():
    """Path to app_window_icon.png in dev tree or inside a py2app bundle Resources folder."""
    if getattr(sys, "frozen", False):
        contents_dir = os.path.dirname(os.path.dirname(sys.executable))
        bundled = os.path.join(contents_dir, "Resources", "app_window_icon.png")
        if os.path.isfile(bundled):
            return bundled
    dev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_window_icon.png")
    return dev_path if os.path.isfile(dev_path) else None


def apply_window_icon(root):
    """
    Set the Tk window icon where supported. The Dock and Finder icon for the distributed
    .app comes from BackgroundRemover.icns via py2app; this PNG keeps the title bar consistent.
    """
    path = _resolve_window_icon_png_path()
    if not path:
        return
    try:
        icon_image = Image.open(path)
        photo = ImageTk.PhotoImage(icon_image)
        root.iconphoto(True, photo)
        root._wm_icon_photo_ref = photo
    except Exception:
        pass


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


def crop_to_visible_bounds(image, padding=0):
    """Crop transparent outer bounds from an RGBA image, if any visible pixels exist."""
    bbox = image.getbbox()
    if bbox is None:
        return image
    
    if padding > 0:
        left, top, right, bottom = bbox
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(image.width, right + padding)
        bottom = min(image.height, bottom + padding)
        bbox = (left, top, right, bottom)
        
    return image.crop(bbox)


def format_rgb_color(color):
    """Format an RGB tuple as a hex color string."""
    red, green, blue = (max(0, min(255, int(channel))) for channel in color)
    return f"#{red:02X}{green:02X}{blue:02X}"


def tk_rgb_from_color_tuple(color):
    """Tk color string from an RGB tuple (handles luminance for near-white swatch borders)."""
    red, green, blue = (max(0, min(255, int(channel))) for channel in color[:3])
    return f"#{red:02x}{green:02x}{blue:02x}"


def apply_color_cleanup(image, cleanup_colors, threshold=0, exact_colors=None):
    """Make pixels transparent when they match any selected cleanup color within tolerance, or exact match."""
    rgba_image = image.convert("RGBA")
    if not cleanup_colors and not exact_colors:
        return rgba_image

    pixel_array = np.array(rgba_image, dtype=np.uint8)
    if pixel_array.size == 0:
        return rgba_image

    opaque_mask = pixel_array[:, :, 3] > 0
    if not np.any(opaque_mask):
        return rgba_image

    rgb_values = pixel_array[:, :, :3].astype(np.int16)
    matched_mask = np.zeros(opaque_mask.shape, dtype=bool)

    if cleanup_colors:
        threshold = max(0, int(threshold))
        for color in cleanup_colors:
            target = np.array([int(channel) for channel in color[:3]], dtype=np.int16)
            channel_difference = np.max(np.abs(rgb_values - target), axis=2)
            matched_mask |= channel_difference <= threshold

    if exact_colors:
        for color in exact_colors:
            target = np.array([int(channel) for channel in color[:3]], dtype=np.int16)
            channel_difference = np.max(np.abs(rgb_values - target), axis=2)
            matched_mask |= channel_difference == 0

    matched_mask &= opaque_mask
    if not np.any(matched_mask):
        return rgba_image

    pixel_array[matched_mask, 3] = 0
    return Image.fromarray(pixel_array)


def resolve_color_cleanup_worker_count(frame_count):
    """Choose a bounded worker count for asynchronous color cleanup and saves."""
    available_cpus = os.cpu_count() or 1
    return max(1, min(MAX_COLOR_CLEANUP_WORKERS, available_cpus, int(frame_count or 1)))


def finalize_processed_cutout(cutout, destination, cleanup_colors, cleanup_threshold, exact_colors=None, auto_crop=True, target_size=None, crop_padding=0):
    """Run the final cleanup pass, crop, resize and save one processed frame."""
    finalized = apply_color_cleanup(cutout, cleanup_colors or [], cleanup_threshold, exact_colors or [])
    if auto_crop:
        finalized = crop_to_visible_bounds(finalized, padding=crop_padding)
    if target_size and target_size[0] > 0 and target_size[1] > 0:
        target_width, target_height = target_size
        img_width, img_height = finalized.size
        left = (img_width - target_width) // 2
        top = (img_height - target_height) // 2
        right = (img_width + target_width) // 2
        bottom = (img_height + target_height) // 2
        finalized = finalized.crop((left, top, right, bottom))
    finalized.save(destination, "PNG")
    return destination


# ---------------------------------------------------------------------------
# Preferences persistence
# ---------------------------------------------------------------------------

_PREFS_DIR = os.path.join(os.path.expanduser("~"), ".config", "backgroundremover")
_PREFS_FILE = os.path.join(_PREFS_DIR, "preferences.json")

_PREF_KEYS = (
    "image_cleanup_threshold",
    "video_cleanup_threshold",
    "deduplication_threshold",
)


def load_preferences():
    """Return a dict of saved user preferences, or an empty dict if none exist."""
    try:
        with open(_PREFS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_preferences(prefs: dict):
    """Persist a dict of user preferences to disk."""
    try:
        os.makedirs(_PREFS_DIR, exist_ok=True)
        with open(_PREFS_FILE, "w", encoding="utf-8") as fh:
            json.dump(prefs, fh, indent=2)
    except Exception:
        pass


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
        self.sprite_output_dir = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Downloads"))
        self.model_choice = tk.StringVar(value="u2net")
        self.alpha_matting = tk.BooleanVar(value=False)
        self.auto_crop_output = tk.BooleanVar(value=True)
        self.image_cleanup_threshold = tk.IntVar(value=COLOR_CLEANUP_THRESHOLD_DEFAULT)
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
        self._video_edit_history = []
        self.video_bg_model_choice = tk.StringVar(value="u2net")
        self.video_bg_alpha_matting = tk.BooleanVar(value=False)
        self.video_bg_auto_crop = tk.BooleanVar(value=True)
        self.video_bg_crop_padding = tk.IntVar(value=0)
        self.video_export_width = tk.StringVar(value="")
        self.video_export_height = tk.StringVar(value="")

        self.video_output_prefix = tk.StringVar()
        self.deduplication_threshold = tk.DoubleVar(value=DUPLICATE_FRAME_MAX_MEAN_DIFFERENCE)
        self.video_cleanup_threshold = tk.IntVar(value=COLOR_CLEANUP_THRESHOLD_DEFAULT)
        self.video_cleanup_colors = []
        self.video_exact_cleanup_colors = []
        self.preview_color_pick_active = False
        self.image_cleanup_colors = []
        self.image_exact_cleanup_colors = []
        self.image_color_pick_mode = "off"
        self.input_preview_zoom = tk.DoubleVar(value=1.0)
        self.output_preview_zoom = tk.DoubleVar(value=1.0)
        self.last_saved_output_hint = tk.StringVar(value="")
        self.input_preview_image_id = None
        self.output_preview_image_id = None
        self._input_preview_canvas_size = (0, 0)
        self._output_preview_canvas_size = (0, 0)
        self._input_preview_draw = None
        self._output_preview_draw = None
        self._input_source_size = None
        self._output_source_size = None
        self.output_preview_display_path = None

        # Apply any previously-saved user preferences before building the UI.
        self._apply_saved_preferences()

        # Configure styles
        self.setup_styles()
        
        # Create UI
        self.create_widgets()

        # Clean up temp frame exports on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Center window
        self.center_window()

        # Sync slider label text with restored values
        self._on_image_cleanup_threshold_change()
        self._on_deduplication_threshold_change()
        self._on_video_cleanup_threshold_change()
    
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
        self.app_shell = tk.Canvas(
            self.root,
            bg=ModernStyle.BG_PRIMARY,
            highlightthickness=0,
            bd=0,
        )
        self.app_shell.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.app_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.app_shell.yview)
        self.app_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.app_shell.configure(yscrollcommand=self.app_scrollbar.set)

        self.app_content = ttk.Frame(self.app_shell, padding="20")
        self.app_shell_window = self.app_shell.create_window((0, 0), window=self.app_content, anchor="nw")
        self.app_content.bind("<Configure>", self._on_app_content_configure)
        self.app_shell.bind("<Configure>", self._on_app_shell_resize)
        self.app_shell.bind_all("<MouseWheel>", self._on_app_mousewheel, add="+")
        self.app_shell.bind_all("<Button-4>", self._on_app_mousewheel, add="+")
        self.app_shell.bind_all("<Button-5>", self._on_app_mousewheel, add="+")

        self.notebook = ttk.Notebook(self.app_content)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        image_tab = ttk.Frame(self.notebook, padding="20")
        self.video_tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(image_tab, text="Background Remover")
        self.notebook.add(self.video_tab, text="Video Frames")

        self.create_image_tab(image_tab)
        self.create_video_tab(self.video_tab)

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
            text="Remove Background writes to the Output File path above. Use Save output as… to copy elsewhere.",
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

        # — Save group (right-aligned) —
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

        # ── Initialize control states ──────────────────────────────
        # Create hidden references for export settings used by other methods
        self.frame_output_entry = None
        self.browse_frame_output_btn = None
        self.video_output_prefix_entry = None

        self._refresh_video_cleanup_controls()
        self._bind_frame_navigation()
    
    def center_window(self):
        """Center the window on screen and adapt to full height"""
        self.root.update_idletasks()
        width = 1180
        # Give some padding to the screen height (e.g., 80px)
        screen_height = self.root.winfo_screenheight()
        height = screen_height - 80
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = max(0, (screen_height // 2) - (height // 2) - 30) # slight upward shift
        self.root.geometry(f'{width}x{height}+{x}+{y}')

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

    def _on_app_content_configure(self, _event=None):
        self.app_shell.configure(scrollregion=self.app_shell.bbox("all"))

    def _on_app_shell_resize(self, event):
        # Update width so content stretches horizontally
        self.app_shell.itemconfigure(self.app_shell_window, width=event.width)
        # Also update height: content fills the canvas unless it naturally needs more space.
        # This eliminates the dead black space below when the window is tall.
        min_height = self.app_content.winfo_reqheight()
        fill_height = max(event.height, min_height)
        self.app_shell.itemconfigure(self.app_shell_window, height=fill_height)
        self.app_shell.configure(scrollregion=(0, 0, event.width, fill_height))


    def _on_app_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return None
            step = -1 * int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)

        self.app_shell.yview_scroll(step, "units")
        return "break"

    def on_close(self):
        """Save user preferences, clean up temporary files, then close."""
        save_preferences(self._collect_preferences())
        self._cleanup_frame_temp_dir()
        self.root.destroy()

    def _collect_preferences(self) -> dict:
        """Build a dict of the threshold values that should persist between launches."""
        return {
            "image_cleanup_threshold": self.image_cleanup_threshold.get(),
            "video_cleanup_threshold": self.video_cleanup_threshold.get(),
            "deduplication_threshold": round(self.deduplication_threshold.get(), 4),
        }

    def _apply_saved_preferences(self):
        """Load persisted preferences and apply them to the relevant tk variables."""
        prefs = load_preferences()
        if not prefs:
            return

        if "image_cleanup_threshold" in prefs:
            val = int(prefs["image_cleanup_threshold"])
            val = max(COLOR_CLEANUP_THRESHOLD_MIN, min(COLOR_CLEANUP_THRESHOLD_MAX, val))
            self.image_cleanup_threshold.set(val)

        if "video_cleanup_threshold" in prefs:
            val = int(prefs["video_cleanup_threshold"])
            val = max(COLOR_CLEANUP_THRESHOLD_MIN, min(COLOR_CLEANUP_THRESHOLD_MAX, val))
            self.video_cleanup_threshold.set(val)

        if "deduplication_threshold" in prefs:
            val = float(prefs["deduplication_threshold"])
            val = max(DEDUPLICATION_THRESHOLD_MIN, min(DEDUPLICATION_THRESHOLD_MAX, val))
            self.deduplication_threshold.set(val)

    def _open_export_settings_dialog(self):
        """Open a dialog for export settings: save location, prefix, resize."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Settings")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.configure(bg=ModernStyle.BG_PRIMARY)

        container = ttk.Frame(dialog, padding="20")
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Export Settings", style='Title.TLabel').pack(
            anchor=tk.W, pady=(0, 10),
        )

        # Save location
        ttk.Label(container, text="Save Location").pack(anchor=tk.W, pady=(0, 4))

        loc_row = ttk.Frame(container)
        loc_row.pack(fill=tk.X, pady=(0, 8))

        tk.Checkbutton(
            loc_row, text="Custom folder",
            variable=self.use_custom_frame_output,
            command=lambda: self._toggle_frame_output_mode_dialog(dialog),
            bg=ModernStyle.BG_PRIMARY, fg=ModernStyle.TEXT_PRIMARY,
            selectcolor=ModernStyle.BG_TERTIARY,
            activebackground=ModernStyle.BG_PRIMARY,
            activeforeground=ModernStyle.ACCENT,
            font=ModernStyle.FONT_BODY,
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.frame_output_entry = tk.Entry(
            loc_row, textvariable=self.frame_output_dir,
            state="normal" if self.use_custom_frame_output.get() else "disabled",
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_PRIMARY,
            disabledbackground=ModernStyle.BG_TERTIARY,
            disabledforeground=ModernStyle.TEXT_SECONDARY,
            insertbackground=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT, font=ModernStyle.FONT_BODY,
            highlightthickness=1, highlightbackground=ModernStyle.BORDER,
            highlightcolor=ModernStyle.ACCENT,
        )
        self.frame_output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 8))

        self.browse_frame_output_btn = RoundedButton(
            loc_row, text="Browse", command=self.browse_frame_output,
            width=80, height=32,
            bg=ModernStyle.BG_TERTIARY, hover_bg=ModernStyle.BORDER,
        )
        self.browse_frame_output_btn.pack(side=tk.RIGHT)
        if not self.use_custom_frame_output.get():
            self.browse_frame_output_btn.configure_state("disabled")

        # Output prefix
        ttk.Label(container, text="Output Name Prefix").pack(anchor=tk.W, pady=(8, 4))

        self.video_output_prefix_entry = tk.Entry(
            container, textvariable=self.video_output_prefix,
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_PRIMARY,
            insertbackground=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT, font=ModernStyle.FONT_BODY,
            highlightthickness=1, highlightbackground=ModernStyle.BORDER,
            highlightcolor=ModernStyle.ACCENT,
        )
        self.video_output_prefix_entry.pack(fill=tk.X, pady=(0, 8), ipady=6)

        # Export resize
        ttk.Label(container, text="Export Resize (optional)").pack(anchor=tk.W, pady=(8, 4))

        resize_row = ttk.Frame(container)
        resize_row.pack(fill=tk.X, pady=(0, 16))

        ttk.Label(resize_row, text="Width:").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            resize_row, textvariable=self.video_export_width, width=8,
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT, font=ModernStyle.FONT_BODY,
        ).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(resize_row, text="Height:").pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            resize_row, textvariable=self.video_export_height, width=8,
            bg=ModernStyle.BG_TERTIARY, fg=ModernStyle.TEXT_PRIMARY,
            relief=tk.FLAT, font=ModernStyle.FONT_BODY,
        ).pack(side=tk.LEFT)

        # Close button
        RoundedButton(
            container, text="Done", command=dialog.destroy,
            width=100, height=36,
            bg=ModernStyle.ACCENT, hover_bg=ModernStyle.ACCENT_HOVER,
        ).pack(anchor=tk.E)

        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_rooty() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.grab_set()
        dialog.focus_set()

    def _toggle_settings_panel(self):
        """Collapse or expand the settings panel below the preview canvas."""
        if self._settings_visible.get():
            self._settings_body.pack_forget()
            self._settings_visible.set(False)
            self._settings_toggle_btn.configure(text="▶  Settings")
        else:
            self._settings_body.pack(fill=tk.X, padx=12, pady=(0, 4))
            self._settings_visible.set(True)
            self._settings_toggle_btn.configure(text="▼  Settings")

    def _toggle_frame_output_mode_dialog(self, dialog=None):
        """Enable or disable custom frame output selection (dialog-safe)."""
        if self.use_custom_frame_output.get():
            if self.frame_output_entry:
                self.frame_output_entry.configure(state="normal")
            if self.browse_frame_output_btn:
                self.browse_frame_output_btn.configure_state("normal")
            self.video_status_label.configure(
                text=f"Custom save folder: {self.frame_output_dir.get()}",
                foreground=ModernStyle.TEXT_SECONDARY,
            )
        else:
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            self.frame_output_dir.set(downloads_dir)
            if self.frame_output_entry:
                self.frame_output_entry.configure(state="disabled")
            if self.browse_frame_output_btn:
                self.browse_frame_output_btn.configure_state("disabled")
            self.video_status_label.configure(
                text=f"Ready. Frames save to {downloads_dir}",
                foreground=ModernStyle.TEXT_SECONDARY,
            )

    def _toggle_frame_output_mode(self):
        """Legacy wrapper for frame output mode toggle."""
        self._toggle_frame_output_mode_dialog()

    def _on_deduplication_threshold_change(self, _value=None):
        """Refresh the threshold value label when the dedupe slider moves."""
        if not hasattr(self, 'deduplication_threshold_value_label') or self.deduplication_threshold_value_label is None:
            return
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
        for child in self.frame_list_inner.winfo_children():
            child.destroy()

        self.frame_items = []
        self.current_frame_index = None
        self.frame_preview_photo = None
        self._clear_frame_preview_canvas("Extract frames to start reviewing them")
        self.frame_preview_meta.configure(text="")
        self.preview_color_pick_active = False
        self._refresh_video_cleanup_controls()

        for item in frame_items:
            frame_item = dict(item)
            # Preserve existing selection when the caller asks for it
            # (e.g. after color cleanup or background removal).
            existing_var = item.get("selected_var")
            was_selected = existing_var.get() if existing_var is not None else bool(item.get("_selected", False))
            initial_value = was_selected if preserve_selection else False
            frame_item["selected_var"] = tk.BooleanVar(value=initial_value)
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
            self.undo_remove_duplicates_btn.configure_state("disabled")
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
        self.pick_cleanup_color_btn.configure_state("normal" if self.current_frame_index is not None else "disabled")
        self.add_cleanup_color_btn.configure_state("normal")
        self.clear_cleanup_colors_btn.configure_state("normal" if self.video_cleanup_colors else "disabled")

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
        else:
            self.pick_cleanup_color_btn.set_text("Sample From Preview")
            self.frame_preview.configure(cursor="")

        self.video_cleanup_colors_label.configure(
            text=f"Cleanup colors: {self._cleanup_colors_summary(self.video_cleanup_colors)}. Tolerance: {self.video_cleanup_threshold.get()}"
        )
        self._rebuild_cleanup_swatches(self.video_cleanup_swatches_frame, self.video_cleanup_colors)

        if hasattr(self, 'video_exact_cleanup_colors_label'):
            self.video_exact_cleanup_colors_label.configure(
                text=f"Exact match colors: {self._cleanup_colors_summary(self.video_exact_cleanup_colors)}"
            )
            self._rebuild_cleanup_swatches(self.video_exact_cleanup_swatches_frame, self.video_exact_cleanup_colors)

        if self.video_processing:
            self.pick_cleanup_color_btn.configure_state("disabled")
            self.add_cleanup_color_btn.configure_state("disabled")
            self.clear_cleanup_colors_btn.configure_state("disabled")
            if hasattr(self, 'pick_exact_cleanup_color_btn'):
                self.pick_exact_cleanup_color_btn.configure_state("disabled")
                self.add_exact_cleanup_color_btn.configure_state("disabled")
                self.clear_exact_cleanup_colors_btn.configure_state("disabled")
            return

        self.pick_cleanup_color_btn.configure_state("normal" if self.current_frame_index is not None else "disabled")
        self.add_cleanup_color_btn.configure_state("normal")
        self.clear_cleanup_colors_btn.configure_state("normal" if self.video_cleanup_colors else "disabled")
        
        if hasattr(self, 'pick_exact_cleanup_color_btn'):
            self.pick_exact_cleanup_color_btn.configure_state("normal" if self.current_frame_index is not None else "disabled")
            self.add_exact_cleanup_color_btn.configure_state("normal")
            self.clear_exact_cleanup_colors_btn.configure_state("normal" if getattr(self, 'video_exact_cleanup_colors', []) else "disabled")

    def _toggle_preview_color_pick(self):
        if self.current_frame_index is None:
            messagebox.showerror("Error", "Select a frame preview before sampling a cleanup color.")
            return

        self.preview_color_pick_active = not self.preview_color_pick_active
        if self.preview_color_pick_active:
            self.exact_preview_color_pick_active = False
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

    def _current_frame_item(self):
        return next((item for item in self.frame_items if item["index"] == self.current_frame_index), None)

    def _snapshot_video_frame_items(self):
        snapshot = []
        for item in self.frame_items:
            snapshot_item = {
                key: value
                for key, value in item.items()
                if key not in {"selected_var", "widget", "thumbnail_photo"}
            }
            selected_var = item.get("selected_var")
            snapshot_item["_selected"] = selected_var.get() if selected_var is not None else bool(item.get("_selected", False))
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
        
        if not (is_threshold_pick or is_exact_pick) or self.current_frame_index is None or self.frame_preview_photo is None:
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

        if is_exact_pick:
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
        """Display the selected frame larger on the right side"""
        matching = next((item for item in self.frame_items if item["index"] == index), None)
        if matching is None:
            return

        self.current_frame_index = index
        with Image.open(matching["path"]) as opened_image:
            image = opened_image.copy()
        # Keep it somewhat constrained but allow scroll scrolling
        image.thumbnail((1200, 900), Image.Resampling.LANCZOS)
        self.frame_preview_photo = ImageTk.PhotoImage(image)
        
        canvas = self.frame_preview_canvas
        canvas.delete("all")
        dw, dh = image.size
        cw = max(canvas.winfo_width(), 1)
        ch = max(canvas.winfo_height(), 1)
        x0 = max(0, (cw - dw) // 2)
        y0 = max(0, (ch - dh) // 2)
        self._frame_preview_x0 = x0
        self._frame_preview_y0 = y0
        
        canvas.create_image(x0, y0, image=self.frame_preview_photo, anchor=tk.NW)
        canvas.config(scrollregion=(0, 0, max(cw, dw), max(ch, dh)))

        selected_count = self._selected_frame_count()
        width, height = matching["size"]
        sel_icon = "\u2713" if matching['selected_var'].get() else "\u2717"
        meta_text = f"{matching['name']}  \u00b7  {width}\u00d7{height}  \u00b7  {sel_icon} {'Selected' if matching['selected_var'].get() else 'Not selected'}  \u00b7  {selected_count} of {len(self.frame_items)} selected"
        if self.preview_color_pick_active or getattr(self, 'exact_preview_color_pick_active', False):
            meta_text += "  \u00b7  Click preview to sample a cleanup color"
        self.frame_preview_meta.configure(text=meta_text)
        self._update_sidebar_counts()
        if hasattr(self, '_preview_info_label'):
            self._preview_info_label.configure(text=f"{matching['name']}  \u00b7  {width}\u00d7{height}")
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
                self.video_bg_crop_padding.get()
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

    def _process_frames_inline_thread(self, selected_items, action, model_name, alpha_matting, cleanup_colors, cleanup_threshold, exact_colors, auto_crop, crop_padding):
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

                    future = executor.submit(
                        finalize_processed_cutout,
                        cutout,
                        destination,
                        cols,
                        cleanup_threshold,
                        exacts,
                        auto_crop if action == "remove_background" else False,
                        None,  # Resize happens on "save selected"
                        crop_padding if action == "remove_background" else 0
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
        """
        self.video_processing = False
        self.video_progress.stop()
        self._rebuild_frame_list(self.frame_items, preserve_selection=True)
        self._set_video_action_states(is_busy=False)
        self.video_status_label.configure(
            text="Processed frames updated in preview.",
            foreground=ModernStyle.SUCCESS,
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
        """Handle frame export failure"""
        self.video_processing = False
        self.video_progress.stop()
        self._set_video_action_states(is_busy=False)
        self.video_status_label.configure(
            text="Saving selected frames failed.",
            foreground=ModernStyle.ERROR,
        )
        messagebox.showerror("Error", f"Failed to save selected frames:\n\n{error_msg}")
    def remove_background_and_save_selected_frames(self):
        """Batch remove backgrounds from selected frames and save them to output directory."""
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

        # Get target size if specified
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
            text=f"Removing backgrounds and saving {len(selected_items)} frame(s)...",
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
                self.video_bg_crop_padding.get()
            ),
            daemon=True
        )
        thread.start()

    def _remove_background_and_save_selected_frames_thread(self, selected_items, target_dir, model_name, alpha_matting, target_size, auto_crop, crop_padding):
        """Worker thread for batch background removal and saving."""
        try:
            from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
            net = self._load_model(model_name)
            
            # Warm up MPS
            if DEVICE.type == "mps":
                _dummy = Image.new("RGB", (64, 64), (128, 128, 128))
                _ = self._predict(net, np.array(_dummy))
                if hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
                del _dummy

            worker_count = resolve_color_cleanup_worker_count(len(selected_items))
            total = len(selected_items)
            output_prefix = self._resolved_video_output_prefix()

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = []
                for position, item in enumerate(selected_items, start=1):
                    self.root.after(0, lambda p=position, t=total: self.video_status_label.configure(text=f"Removing background {p}/{t}..."))
                    
                    with Image.open(item["path"]) as opened_image:
                        img = opened_image.convert("RGB")
                    
                    # AI removal mask
                    mask_array = self._predict(net, np.array(img))
                    mask = Image.fromarray(mask_array).convert("L")
                    # Using global _mask_to_cutout
                    cutout = _mask_to_cutout(img, mask, alpha_matting=alpha_matting)
                    
                    destination = os.path.join(target_dir, build_export_filename(output_prefix, position))
                    
                    # Finalize (includes crop)
                    future = executor.submit(
                        finalize_processed_cutout,
                        cutout,
                        destination,
                        [], # cleanup_colors
                        0,  # cleanup_threshold
                        [], # exact_colors
                        auto_crop,
                        target_size,
                        crop_padding
                    )
                    futures.append(future)

                wait(futures, return_when=ALL_COMPLETED)
                saved_paths = [f.result() for f in futures]

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
        else:
            self.pick_image_cleanup_color_btn.set_text("Sample: off (click to cycle)")
            self.input_preview_canvas.configure(cursor="")
            self.output_preview_canvas.configure(cursor="")

        self.image_cleanup_colors_label.configure(
            text=(
                f"Cleanup colors: {self._image_cleanup_colors_summary()}. "
                f"Tolerance: {self.image_cleanup_threshold.get()}"
            )
        )
        self._rebuild_cleanup_swatches(self.image_cleanup_swatches_frame, self.image_cleanup_colors)

        has_input = bool(self.input_file.get()) and os.path.exists(self.input_file.get())
        if self.processing:
            self.pick_image_cleanup_color_btn.configure_state("disabled")
            self.add_image_cleanup_color_btn.configure_state("disabled")
            self.clear_image_cleanup_colors_btn.configure_state("disabled")
            if getattr(self, "apply_cleanup_save_btn", None):
                self.apply_cleanup_save_btn.configure_state("disabled")
            return

        self.pick_image_cleanup_color_btn.configure_state("normal" if has_input else "disabled")
        self.add_image_cleanup_color_btn.configure_state("normal")
        self.clear_image_cleanup_colors_btn.configure_state("normal" if self.image_cleanup_colors else "disabled")
        if getattr(self, "apply_cleanup_save_btn", None):
            self.apply_cleanup_save_btn.configure_state("normal" if has_input else "disabled")

    def _on_image_cleanup_threshold_change(self, _value=None):
        self.image_cleanup_threshold_value_label.configure(text=str(self.image_cleanup_threshold.get()))
        self._refresh_image_cleanup_controls()

    def _toggle_image_preview_color_pick(self):
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
        if self.image_color_pick_mode != "input":
            return
        cx = event.widget.canvasx(event.x)
        cy = event.widget.canvasy(event.y)
        sampled = self._sample_rgb_from_draw(self._input_preview_draw, cx, cy)
        if sampled is None:
            return
        self._add_image_cleanup_color(sampled)
        self.status_label.configure(
            text=f"Added cleanup color {format_rgb_color(sampled)}. Matching pixels will be removed before save.",
            foreground=ModernStyle.SUCCESS,
        )

    def _on_output_preview_canvas_click(self, event):
        if self.image_color_pick_mode != "output":
            return
        cx = event.widget.canvasx(event.x)
        cy = event.widget.canvasy(event.y)
        sampled = self._sample_rgb_from_draw(self._output_preview_draw, cx, cy)
        if sampled is None:
            return
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

    def _refresh_output_action_buttons(self):
        if getattr(self, "use_output_as_input_btn", None) is None:
            return
        out_path = self.output_file.get()
        has_output_file = bool(out_path) and os.path.isfile(out_path)
        busy = self.processing
        self.use_output_as_input_btn.configure_state("normal" if has_output_file and not busy else "disabled")
        self.save_output_copy_btn.configure_state("normal" if has_output_file and not busy else "disabled")
        if sys.platform == "darwin" and getattr(self, "reveal_output_btn", None):
            self.reveal_output_btn.configure_state("normal" if has_output_file and not busy else "disabled")

    def use_output_as_input(self):
        """Set the current output file as the input image for cleanup refinement."""
        out_path = (self.output_file.get() or "").strip()
        if not out_path or not os.path.isfile(out_path):
            messagebox.showerror("Error", "No saved output file found. Run Remove Background or Apply cleanup & save first.")
            return
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
            subprocess.run(["open", "-R", path], check=False)
        except OSError as exc:
            messagebox.showerror("Error", f"Could not reveal file:\n{exc}")

    def apply_cleanup_and_save(self):
        """Apply color cleanup to the current input (RGBA) and save without running the AI model."""
        in_path = (self.input_file.get() or "").strip()
        out_path = (self.output_file.get() or "").strip()
        if not in_path or not os.path.isfile(in_path):
            messagebox.showerror("Error", "Select an input image file first.")
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
                cutout = apply_color_cleanup(rgba, self.image_cleanup_colors, self.image_cleanup_threshold.get())
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
            self.output_preview_display_path = None
            self.load_output_preview()

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
        self.sprite_process_btn.configure_state("disabled")
        self.progress.start(10)
        self.status_label.configure(text="Processing... Please wait.")
        self._refresh_image_cleanup_controls()
        self._refresh_output_action_buttons()

        thread = threading.Thread(target=self._process_thread)
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
            cutout = apply_color_cleanup(
                cutout,
                self.image_cleanup_colors,
                self.image_cleanup_threshold.get(),
            )
            if self.auto_crop_output.get():
                cutout = crop_to_visible_bounds(cutout)
            
            # Save
            cutout.save(self.output_file.get(), "PNG")
            
            self.root.after(0, self._on_success)
            
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
    
    def _on_success(self):
        """Called on successful processing"""
        self.processing = False
        self.progress.stop()
        self.process_btn.configure_state("normal")
        self.sprite_process_btn.configure_state("normal")
        self._refresh_image_cleanup_controls()
        self.status_label.configure(text="✓ Background removed successfully!", foreground=ModernStyle.SUCCESS)

        self._set_last_saved_output_hint(self.output_file.get())
        self.load_output_preview()

        messagebox.showinfo("Success", f"Background removed!\n\nSaved to:\n{self.output_file.get()}")

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
        self._refresh_image_cleanup_controls()
        self._refresh_output_action_buttons()
        self.status_label.configure(text="✗ Processing failed", foreground=ModernStyle.ERROR)

        messagebox.showerror("Error", f"Failed to process image:\n\n{error_msg}")


def main():
    """Main entry point"""
    root = tk.Tk()
    apply_window_icon(root)
    app = BackgroundRemoverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
