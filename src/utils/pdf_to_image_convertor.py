from pdf2image import convert_from_path
import os

class PDF2ImageConvertor:
    
    @staticmethod
    def pdf_to_images_from_path(pdf_path):
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=500)

        return images