from transformers import TableTransformerForObjectDetection
import torch
from torchvision import transforms
from src.utils.transformations import MaxResize
from src.table_transformer.detector import Detector


class RowColDetector(Detector):
    def __init__(self, model_card='farhanishraq/table_tr-finetuned-bs-2.0', device=None) -> None:
        super().__init__()
        self.model = TableTransformerForObjectDetection.from_pretrained(model_card)
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.structure_transform = transforms.Compose([
            MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def detect(self, cropped_table):
        pixel_values = self.structure_transform(cropped_table).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)
        
        # forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values)
        cells = self.outputs_to_objects(outputs, cropped_table.size, self.model.config.id2label, True)
        return cells