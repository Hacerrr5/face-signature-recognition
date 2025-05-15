import tkinter as tk
from PIL import Image, ImageDraw
import os

class SignatureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Drawing Application")

        # Canvas for drawing the signature
        self.canvas = tk.Canvas(root, width=400, height=300, bg="white")
        self.canvas.pack()

        # Buttons for saving and clearing the canvas
        self.button_save = tk.Button(root, text="Save", command=self.save_signature)
        self.button_save.pack()

        self.button_clear = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        # Image object for drawing (used to save the signature)
        self.signature_image = Image.new("RGB", (400, 300), "white")
        self.draw = ImageDraw.Draw(self.signature_image)

        self.last_x = None
        self.last_y = None

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        """Draw lines on the canvas and image when mouse is moved"""
        if self.last_x and self.last_y:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=3, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, x, y], fill="black", width=3)
        self.last_x = event.x
        self.last_y = event.y

    def reset(self, event):
        """Reset the last mouse position when the mouse button is released"""
        self.last_x = None
        self.last_y = None

    def save_signature(self):
        """Save the drawn signature image to a folder"""
        save_folder = "signatures"  # Universal folder name, created if not exist

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Generate a filename based on existing files count
        file_count = len(os.listdir(save_folder))
        file_path = os.path.join(save_folder, f"signature_{file_count + 1}.png")

        # Resize the image and save it
        resized_image = self.signature_image.resize((300, 150), Image.LANCZOS)
        resized_image.save(file_path)
        print(f"Signature saved: {file_path}")

        # Clear the canvas after saving
        self.reset_canvas()

    def clear_canvas(self):
        """Clear the drawing area"""
        self.canvas.delete("all")
        self.signature_image = Image.new("RGB", (400, 300), "white")
        self.draw = ImageDraw.Draw(self.signature_image)
        print("Canvas cleared.")

    def reset_canvas(self):
        """Reset the drawing area after saving"""
        self.canvas.delete("all")
        self.signature_image = Image.new("RGB", (400, 300), "white")
        self.draw = ImageDraw.Draw(self.signature_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureApp(root)
    root.mainloop()
