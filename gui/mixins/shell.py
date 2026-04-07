"""Main window shell: ttk styles, notebook host, scrolling, preferences."""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

from gui.constants import (
    COLOR_CLEANUP_THRESHOLD_MAX,
    COLOR_CLEANUP_THRESHOLD_MIN,
    DEDUPLICATION_THRESHOLD_MAX,
    DEDUPLICATION_THRESHOLD_MIN,
)
from gui.preferences import load_preferences, save_preferences
from gui.widgets import ModernStyle, RoundedButton


class ShellMixin:
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
