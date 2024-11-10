import tkinter as tk
from tkinter import ttk, messagebox, colorchooser, filedialog
from PIL import Image, ImageGrab, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging
from datetime import datetime
import json

class EnhancedDigitRecognizer:
    def __init__(self, model_path='mnist_cnn_model.h5'):
        self.setup_logging()
        self.load_model(model_path)
        self.prediction_history = []
        self.load_settings()
        self.setup_ui()
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.current_brush_color = "#000000"
        self.strokes = []  # Store drawing strokes for undo
        
    def setup_logging(self):
        """Configure logging"""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        logging.basicConfig(
            filename=f'logs/digit_recognizer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = load_model(model_path)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            messagebox.showerror("Error", "Failed to load the model. Please check if the model file exists.")
            raise

    def load_settings(self):
        """Load user settings"""
        self.settings = {
            'brush_size': 15,
            'brush_color': '#000000',
            'canvas_color': '#FFFFFF'
        }
        
        try:
            if os.path.exists('settings.json'):
                with open('settings.json', 'r') as f:
                    self.settings.update(json.load(f))
        except Exception as e:
            logging.error(f"Failed to load settings: {str(e)}")

    def setup_ui(self):
        """Initialize the UI"""
        self.window = tk.Tk()
        self.window.title("Enhanced Digit Recognizer")
        self.window.geometry("800x600")
        
        # Configure styles
        style = ttk.Style()
        style.configure("TButton", padding=5)
        style.configure("TLabel", padding=3)
        
        # Main container with two panels
        main_container = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for drawing
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=2)
        
        # Right panel for history
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=1)
        
        self.setup_drawing_area(left_panel)
        self.setup_controls(left_panel)
        self.setup_history_panel(right_panel)
        
    def setup_drawing_area(self, parent):
        """Setup the drawing canvas"""
        # Canvas frame
        canvas_frame = ttk.LabelFrame(parent, text="Drawing Area", padding=10)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            width=280,
            height=280,
            bg=self.settings['canvas_color'],
            relief="ridge",
            bd=3
        )
        self.canvas.pack(padx=5, pady=5)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
    def setup_controls(self, parent):
        """Setup the control panel"""
        controls = ttk.LabelFrame(parent, text="Controls", padding=10)
        controls.pack(fill=tk.X, padx=5, pady=5)
        
        # Brush size control
        size_frame = ttk.Frame(controls)
        size_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(size_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        self.brush_size = tk.IntVar(value=self.settings['brush_size'])
        brush_scale = ttk.Scale(
            size_frame,
            from_=5,
            to=30,
            orient=tk.HORIZONTAL,
            variable=self.brush_size
        )
        brush_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Color picker
        color_btn = ttk.Button(
            size_frame,
            text="Brush Color",
            command=self.choose_color
        )
        color_btn.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        btn_frame = ttk.Frame(controls)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            btn_frame,
            text="Clear",
            command=self.clear_canvas
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Predict",
            command=self.predict_digit
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Undo",
            command=self.undo_last_stroke
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Save Drawing",
            command=self.save_drawing
        ).pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.result_label = ttk.Label(
            controls,
            text="Draw a digit and click Predict",
            font=('Helvetica', 14)
        )
        self.result_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(
            controls,
            text="",
            font=('Helvetica', 12)
        )
        self.confidence_label.pack()
        
    def setup_history_panel(self, parent):
        """Setup the prediction history panel"""
        history_frame = ttk.LabelFrame(parent, text="Prediction History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview
        columns = ('Time', 'Digit', 'Confidence')
        self.history_tree = ttk.Treeview(
            history_frame,
            columns=columns,
            show='headings',
            height=10
        )
        
        # Configure columns
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=80)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            history_frame,
            orient=tk.VERTICAL,
            command=self.history_tree.yview
        )
        
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack elements
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export button
        ttk.Button(
            history_frame,
            text="Export History",
            command=self.export_history
        ).pack(pady=5)

    def choose_color(self):
        """Open color picker"""
        color = colorchooser.askcolor(title="Choose Brush Color")[1]
        if color:
            self.current_brush_color = color
            self.settings['brush_color'] = color

    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        self.strokes = []
        self.result_label.config(text="Draw a digit and click Predict")
        self.confidence_label.config(text="")
        logging.info("Canvas cleared")

    def undo_last_stroke(self):
        """Undo the last drawing stroke"""
        if self.strokes:
            last_stroke = self.strokes.pop()
            self.canvas.delete(last_stroke)

    def start_drawing(self, event):
        """Handle drawing start"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        """Handle drawing motion"""
        if self.drawing:
            x, y = event.x, event.y
            size = self.brush_size.get()
            
            line = self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=size,
                fill=self.current_brush_color,
                capstyle=tk.ROUND,
                smooth=True
            )
            
            self.strokes.append(line)
            self.last_x = x
            self.last_y = y

    def stop_drawing(self, event):
        """Handle drawing stop"""
        self.drawing = False

    def save_drawing(self):
        """Save the current drawing"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                x = self.window.winfo_rootx() + self.canvas.winfo_x()
                y = self.window.winfo_rooty() + self.canvas.winfo_y()
                x1 = x + self.canvas.winfo_width()
                y1 = y + self.canvas.winfo_height()
                
                ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
                messagebox.showinfo("Success", "Drawing saved successfully!")
        except Exception as e:
            logging.error(f"Failed to save drawing: {str(e)}")
            messagebox.showerror("Error", "Failed to save drawing")

    def export_history(self):
        """Export prediction history"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w') as f:
                    f.write("Time,Digit,Confidence\n")
                    for record in self.prediction_history:
                        f.write(f"{record['time']},{record['digit']},{record['confidence']}\n")
                messagebox.showinfo("Success", "History exported successfully!")
        except Exception as e:
            logging.error(f"Failed to export history: {str(e)}")
            messagebox.showerror("Error", "Failed to export history")

    def preprocess_image(self):
        """Preprocess the canvas image"""
        try:
            x = self.window.winfo_rootx() + self.canvas.winfo_x()
            y = self.window.winfo_rooty() + self.canvas.winfo_y()
            x1 = x + self.canvas.winfo_width()
            y1 = y + self.canvas.winfo_height()
            
            img = ImageGrab.grab().crop((x, y, x1, y1))
            img = img.convert('L')
            img = ImageOps.invert(img)
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            
            img_array = np.array(img).astype('float32') / 255
            img_array = img_array.reshape(1, 28, 28, 1)
            
            return img_array
        except Exception as e:
            logging.error(f"Image preprocessing failed: {str(e)}")
            messagebox.showerror("Error", "Failed to process the image")
            return None

    def predict_digit(self):
        """Predict the drawn digit"""
        try:
            img_array = self.preprocess_image()
            if img_array is None:
                return
                
            prediction = self.model.predict(img_array, verbose=0)
            digit = np.argmax(prediction)
            confidence = prediction[0][digit] * 100
            
            # Update display
            self.result_label.config(
                text=f"Predicted Digit: {digit}",
                foreground="green" if confidence > 80 else "orange"
            )
            self.confidence_label.config(
                text=f"Confidence: {confidence:.2f}%"
            )
            
            # Update history
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.history_tree.insert(
                '',
                'end',
                values=(timestamp, digit, f"{confidence:.2f}%")
            )
            
            # Store prediction
            self.prediction_history.append({
                'time': timestamp,
                'digit': digit,
                'confidence': confidence
            })
            
            logging.info(f"Prediction made: digit={digit}, confidence={confidence:.2f}%")
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            messagebox.showerror("Error", "Failed to make prediction")

    def run(self):
        """Start the application"""
        try:
            self.window.mainloop()
        except Exception as e:
            logging.error(f"Application crashed: {str(e)}")
            messagebox.showerror("Error", "Application crashed. Check logs for details.")

if __name__ == "__main__":
    try:
        app = EnhancedDigitRecognizer()
        app.run()
    except Exception as e:
        logging.error(f"Failed to start application: {str(e)}")
        messagebox.showerror("Error", "Failed to start application")