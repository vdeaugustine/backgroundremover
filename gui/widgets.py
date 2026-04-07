"""Tk styling tokens and a rounded canvas button."""

import tkinter as tk


class ModernStyle:
    """Modern macOS-inspired styling."""

    BG_PRIMARY = "#000000"
    BG_SECONDARY = "#1c1c1e"
    BG_TERTIARY = "#2c2c2e"
    ACCENT = "#2ecc71"
    ACCENT_HOVER = "#27ae60"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#a1a1a6"
    SUCCESS = "#32d74b"
    ERROR = "#ff453a"
    BORDER = "#3a3a3c"

    FONT_TITLE = ("SF Pro Display", 28, "bold")
    FONT_SUBTITLE = ("SF Pro Display", 15)
    FONT_BODY = ("SF Pro Text", 13)
    FONT_SMALL = ("SF Pro Text", 11)
    FONT_BUTTON = ("SF Pro Text", 13, "bold")


class RoundedButton(tk.Canvas):
    """A custom rounded button using tk.Canvas for better macOS support."""

    def __init__(
        self,
        parent,
        text,
        command=None,
        width=120,
        height=35,
        radius=10,
        bg=None,
        fg="white",
        hover_bg=None,
        font=None,
        **kwargs,
    ):
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
        self.text = self.create_text(width / 2, height / 2, text=text, fill=fg, font=self.font)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r,
            y1,
            x1 + r,
            y1,
            x2 - r,
            y1,
            x2 - r,
            y1,
            x2,
            y1,
            x2,
            y1 + r,
            x2,
            y1 + r,
            x2,
            y2 - r,
            x2,
            y2 - r,
            x2,
            y2,
            x2 - r,
            y2,
            x2 - r,
            y2,
            x1 + r,
            y2,
            x1 + r,
            y2,
            x1,
            y2,
            x1,
            y2 - r,
            x1,
            y2 - r,
            x1,
            y1 + r,
            x1,
            y1 + r,
            x1,
            y1,
        ]
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
