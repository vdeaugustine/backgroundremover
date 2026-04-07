"""RGB color formatting helpers for Tk and labels."""


def format_rgb_color(color):
    """Format an RGB tuple as a hex color string."""
    red, green, blue = (max(0, min(255, int(channel))) for channel in color)
    return f"#{red:02X}{green:02X}{blue:02X}"


def tk_rgb_from_color_tuple(color):
    """Tk color string from an RGB tuple (handles luminance for near-white swatch borders)."""
    red, green, blue = (max(0, min(255, int(channel))) for channel in color[:3])
    return f"#{red:02x}{green:02x}{blue:02x}"
