from fpdf import FPDF
import os

class PDFQuizGenerator(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()

        # Add a Unicode font (like DejaVu Sans)
        font_path = "./DejaVuSans.ttf"
        if not os.path.exists(font_path):
            raise FileNotFoundError("DejaVuSans.ttf not found. Download it and place it in your project directory.")

        self.add_font("DejaVu", "", font_path, uni=True)
        self.set_font("DejaVu", "", 14)

    def add_quiz(self, text: str):
        self.multi_cell(0, 10, text)