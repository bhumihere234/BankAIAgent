from transformers import TableTransformerForObjectDetection
import torch
from torchvision import transforms
from utils.transformations import MaxResize
from table_transformer.detector import Detector


class RowColDetector(Detector):
    def __init__(
            self, 
            model_card='farhanishraq/table_tr-finetuned-bs-2.0', 
            device=None,
            row_threshold=0.5,
            col_threshold=0.5
        ) -> None:
            super().__init__()
            self.model = TableTransformerForObjectDetection.from_pretrained(model_card)
            self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.structure_transform = transforms.Compose([
                MaxResize(1000),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            # update id2label to include "no object"
            self.id2label = self.model.config.id2label
            self.id2label[len(self.model.config.id2label)] = "no object"
            self.row_threshold = row_threshold
            self.col_threshold = col_threshold

    def detect(self, cropped_table):
        pixel_values = self.structure_transform(cropped_table).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)
        
        # forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values)
        cells = self.outputs_to_objects(outputs, cropped_table.size, self.id2label, True)
        return cells
    
    def get_cell_coordinates_by_row(self, table_data):
        # Extract rows and columns
        rows = [entry for entry in table_data if entry['label'] == 'table row' and entry['score']>= self.row_threshold]
        columns = [entry for entry in table_data if entry['label'] == 'table column' and entry['score']>= self.col_threshold]

        # Sort rows and columns by their Y and X coordinates, respectively
        rows.sort(key=lambda x: x['bbox'][1])

        # row_processed = []
        # i=0
        # while i<len(rows)-1:
        #     row1, row2 = rows[i], rows[i+1]
        #     row, to_merge = merge_rows(row1, row2)
        #     row_processed.append(row)
        #     if to_merge:
        #         i+=2
        #     else:
        #         i+=1
            
        columns.sort(key=lambda x: x['bbox'][0])

        # Generate cell coordinates and count cells in each row
        cell_coordinates = []

        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = self.find_cell_coordinates(row, column)
                row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

            # Sort cells in the row by X coordinate
            row_cells.sort(key=lambda x: x['column'][0])

            # Append row information to cell_coordinates
            cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

        # Sort rows from top to bottom
        cell_coordinates.sort(key=lambda x: x['row'][1])

        return cell_coordinates
    
    # Function to find cell coordinates
    def find_cell_coordinates(self, row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox