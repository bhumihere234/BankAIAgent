from transformers import TableTransformerForObjectDetection
import torch
from utils.transformations import MaxResize
from torchvision import transforms
from utils.pdf_to_image_convertor import PDF2ImageConvertor
from table_transformer.table_detector import TableDetector
from table_transformer.row_col_detector import RowColDetector
from ocr.ocr_model import TextExtract
import csv
import tempfile
import pandas as pd


class StatementParser:
    def __init__(
            self,
            table_detector:TableDetector = TableDetector(),
            row_col_detector: RowColDetector = RowColDetector(),
            ocr_model: TextExtract = TextExtract()
        ) -> None:
            self.table_detector = table_detector
            self.row_col_detector = row_col_detector
            self.ocr_model = ocr_model
        
    def bankstatement2csv(self, pdf, output_path='output.xlsx'):
        # Convert pdf to image
        table_data = []
        images = PDF2ImageConvertor.pdf_to_images_from_path(pdf)
        for image in images:
            # Detect tables in pdf
            tables = self.table_detector.detect(image)
            for table in tables:
                table_image = table['image'].convert("RGB")
                table_image_w, table_image_h = table_image.size
                # if the image height is beyond a threshold, split the image into 2
                if table_image_h>1000:
                    # This section needs refactoring. This is a hack.
                    dissected_tables = self.dissect_image_in_half(table_image)
                    for idx, table_image in enumerate(dissected_tables):
                        with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as temp_file:
                            temp_filename = temp_file.name
                            # Save the image to the temporary file
                            table_image.save(temp_filename)
                            result = self.ocr_model.run_ocr(temp_filename)
                        ocr_results =  self.ocr_model.get_ocr_text_and_bbox(result)
                        # Get all rows and columns via the RowColDetector
                        row_columns = self.row_col_detector.detect(table_image)
                        row_columns_cell_coordinates = self.row_col_detector.get_cell_coordinates_by_row(row_columns)
                        # Combine OCR output with cell coordinates to get text in row-col intersection(cell)
                        if idx == 1:
                            data2 = self.apply_ocr(row_columns_cell_coordinates, ocr_results)
                            # Collect changes in a separate dictionary
                            changes = {}
                            for k, v in list(data2.items()):  # Convert to list to avoid runtime errors
                                new = data_key_last_idx + list(data2.keys()).index(k) + 1
                                changes[new] = v

                            data1.update(changes)
                        else:
                            data1 = self.apply_ocr(row_columns_cell_coordinates, ocr_results)
                            if not data1:
                                break
                            data_key_last_idx = list(data1.keys())[-1]
                    table_data.append(data1)
                else:
                    # Perform ocr to get ocr text and bbox coordinates on cropped table
                    with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as temp_file:
                        temp_filename = temp_file.name
                        # Save the image to the temporary file
                        table_image.save(temp_filename)
                        result = self.ocr_model.run_ocr(temp_filename)
                    ocr_results =  self.ocr_model.get_ocr_text_and_bbox(result)
                    # Get all rows and columns via the RowColDetector
                    row_columns = self.row_col_detector.detect(table_image)
                    row_columns_cell_coordinates = self.row_col_detector.get_cell_coordinates_by_row(row_columns)
                    # Combine OCR output with cell coordinates to get text in row-col intersection(cell)
                    data = self.apply_ocr(row_columns_cell_coordinates, ocr_results)
                    table_data.append(data)
        #print(table_data)
        self.create_sheets_from_data(table_data, output_path)
    
    
    def dissect_image_in_half(self, image):
        # Get the image dimensions
        width, height = image.size

        # Calculate the midpoint of the height
        midpoint = height // 2

        # Crop the image into two halves
        upper_half = image.crop((0, 0, width, midpoint))
        lower_half = image.crop((0, midpoint, width, height))

        return [upper_half, lower_half]
    
    def create_csv_from_data(self, data, output_path):
        with open(output_path,'w') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            for row, row_text in data.items():
                wr.writerow(row_text)
                
    def create_sheets_from_data(self, data, output_path):
        # Create a Pandas Excel writer using openpyxl as the engine
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Iterate over the list of dictionaries
            for idx, sheet_data in enumerate(data):
                if sheet_data:
                    # Create a DataFrame from the dictionary
                    df = pd.DataFrame(sheet_data)
                    # Transpose the DataFrame so rows become columns and columns become rows
                    df = df.transpose()
                    # Write the DataFrame to a sheet
                    sheet_name = f"Sheet{idx + 1}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        
            
    def apply_ocr(self, cell_coordinates, ocr_results):
        # let's OCR row by row
        data = dict()
        max_num_columns = 0
        for idx, row in enumerate(cell_coordinates):
            row_text = []
            for cell in row["cells"]:
                line = self.words_in_cell(cell['cell'], ocr_results)
                if len(line)>0:
                    row_text.append(line)
                if len(row_text) > max_num_columns:
                    max_num_columns = len(row_text)

                data[idx] = row_text

        # pad rows which don't have max_num_columns elements
        # to make sure all rows have the same number of columns
        for row, row_data in data.copy().items():
            if len(row_data) != max_num_columns:
                row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
                data[row] = row_data

        return data

    
    
    def is_within(self, cell_bbox, word_bbox, padding = 5):
            """
            Check if a word bounding box is within a cell bounding box.
            
            Args:
            cell_bbox (tuple): (x1, y1, x2, y2) coordinates of the cell bounding box.
            word_bbox (tuple): (x1, y1, x2, y2) coordinates of the word bounding box.
            
            Returns:
            bool: True if word_bbox is within cell_bbox, False otherwise.
            """
            cell_x1, cell_y1, cell_x2, cell_y2 = cell_bbox
            cell_x1, cell_y1, cell_x2, cell_y2 = cell_x1 + padding, cell_y1 + padding, cell_x2 + padding, cell_y2 + padding
            word_x1, word_y1, word_x2, word_y2 = word_bbox

            return (cell_x1 <= word_x1 and cell_y1 <= word_y1 and 
                    cell_x2 >= word_x2 and cell_y2 >= word_y2)

    def words_in_cell(self, cell_bbox, word_dicts):
        """
        Get all word bounding boxes within a given cell bounding box and sort them by x and y axis.
        
        Args:
        cell_bbox (tuple): (x1, y1, x2, y2) coordinates of the cell bounding box.
        word_dicts (list): List of dictionaries, each containing a word and its bounding box.
        
        Returns:
        list: List of sorted dictionaries with word and bounding boxes within the cell bounding box.
        """
        
        # Filter word dictionaries that are within the cell bounding box
        words_in_cell = [word_dict for word_dict in word_dicts if self.is_within(cell_bbox, word_dict['bbox'])]
        
        # Sort the word dictionaries first by y-axis and then by x-axis
        words_in_cell_sorted = sorted(words_in_cell, key=lambda word: (word['bbox'][1], word['bbox'][0]))
        
        words = [word['word']  for word in words_in_cell_sorted]
        
        return " ".join(words)
    
    