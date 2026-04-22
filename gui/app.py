"""Tk application class: shell + image tab + video tab mixins."""

import os

import tkinter as tk

from gui.constants import (
    COLOR_CLEANUP_THRESHOLD_DEFAULT,
    COLOR_PROTECT_THRESHOLD_DEFAULT,
    DUPLICATE_FRAME_MAX_MEAN_DIFFERENCE,
    PROTECT_BRUSH_RADIUS_DEFAULT,
)
from gui.widgets import ModernStyle
from gui.mixins.image_tab import ImageTabMixin
from gui.mixins.shell import ShellMixin
from gui.mixins.video_tab import VideoTabMixin


class BackgroundRemoverApp(ShellMixin, ImageTabMixin, VideoTabMixin):
    """macOS Background Remover GUI."""

    def __init__(self, root):
        self.root = root
        self.root.title("Background Remover")
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)
        self.root.configure(bg=ModernStyle.BG_PRIMARY)

        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.image_input_files = []
        self.image_output_prefix = tk.StringVar()
        self.sprite_output_dir = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Downloads"))
        self.model_choice = tk.StringVar(value="u2net")
        self.alpha_matting = tk.BooleanVar(value=False)
        self.auto_crop_output = tk.BooleanVar(value=True)
        self.image_cleanup_threshold = tk.IntVar(value=COLOR_CLEANUP_THRESHOLD_DEFAULT)
        self.processing = False
        self.video_processing = False

        self.model_cache = {}

        self.input_photo = None
        self.output_photo = None
        self.frame_preview_photo = None

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
        self.video_protected_colors = []
        self.video_protected_threshold = tk.IntVar(value=COLOR_PROTECT_THRESHOLD_DEFAULT)
        self.preview_color_pick_active = False
        self.exact_preview_color_pick_active = False
        self.video_protect_color_pick_active = False
        self.image_cleanup_colors = []
        self.image_exact_cleanup_colors = []
        self.image_protected_colors = []
        self.image_protected_threshold = tk.IntVar(value=COLOR_PROTECT_THRESHOLD_DEFAULT)
        self.image_protect_color_pick_mode = "off"
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

        self._frame_preview_interaction_mode = "off"
        self.protect_brush_radius = tk.IntVar(value=PROTECT_BRUSH_RADIUS_DEFAULT)
        self._protect_brush_dragging = False

        self._apply_saved_preferences()

        self.setup_styles()

        self.create_widgets()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.center_window()

        self._on_image_cleanup_threshold_change()
        self._on_deduplication_threshold_change()
        self._on_video_cleanup_threshold_change()


def main():
    from gui.window_icon import apply_window_icon

    root = tk.Tk()
    apply_window_icon(root)
    BackgroundRemoverApp(root)
    root.mainloop()
