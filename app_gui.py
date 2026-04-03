#!/usr/bin/env python3
"""
Background Remover - macOS Application
A modern, user-friendly GUI for removing backgrounds from images.
Optimized for Apple Silicon (M-series) with MPS acceleration.

Note: This GUI is for IMAGE processing only. For video, use the CLI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from PIL import Image, ImageTk
import threading
import io
import numpy as np

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
        self.root.geometry("800x700")
        self.root.minsize(700, 600)
        self.root.configure(bg=ModernStyle.BG_PRIMARY)
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.model_choice = tk.StringVar(value="u2net")
        self.alpha_matting = tk.BooleanVar(value=False)
        self.processing = False
        
        # Model cache
        self.model_cache = {}
        
        # Image references (prevent garbage collection)
        self.input_photo = None
        self.output_photo = None
        
        # Configure styles
        self.setup_styles()
        
        # Create UI
        self.create_widgets()
        
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
        main_frame = ttk.Frame(self.root, padding="30")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title section
        title_frame = ttk.Frame(main_frame)
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
        file_frame = ttk.Frame(main_frame)
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
        options_frame = ttk.Frame(main_frame)
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
        self.process_btn = RoundedButton(main_frame, text="Remove Background",
                                        command=self.process_image,
                                        width=280, height=50,
                                        radius=15,
                                        bg=ModernStyle.ACCENT,
                                        hover_bg=ModernStyle.ACCENT_HOVER,
                                        font=ModernStyle.FONT_BUTTON)
        self.process_btn.pack(pady=(10, 20))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', style='TProgressbar')
        self.progress.pack(fill=tk.X, pady=(0, 5))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", style='Small.TLabel')
        self.status_label.pack(pady=(0, 20))
        
        # Preview section
        preview_frame = ttk.Frame(main_frame)
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
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
    
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
            
            # Get mask
            mask = self._predict(net, np.array(img))
            
            # Apply cutout
            if self.alpha_matting.get():
                cutout = self._alpha_matting_cutout(img, mask)
            else:
                cutout = self._naive_cutout(img, mask)
            
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
