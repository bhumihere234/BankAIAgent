from torchvision import transforms
from transformers import AutoModelForObjectDetection
import torch
from src.utils.transformations import MaxResize
from src.table_transformer.detector import Detector


class TableDetector(Detector):
    def __init__(self, model_card='microsoft/table-transformer-detection', device=None) -> None:
        super().__init__()
        self.model = AutoModelForObjectDetection.from_pretrained(model_card, revision="no_timm")
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.detection_transform = transforms.Compose([
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
            # update id2label to include "no object"
        self.id2label = self.model.config.id2label
        self.id2label[len(self.model.config.id2label)] = "no object"
        self.detection_class_thresholds = {
            "table": 0.5,
            "table rotated": 0.5,
            "no object": 10
        }
        
    def detect(self, image):
        pixel_values = self.detection_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values)
        objects = self.outputs_to_objects(outputs, image.size, self.id2label)
        
        tokens = []
        cropped_tables = self.objects_to_crops(image, tokens, objects, self.detection_class_thresholds, padding=50)
        return cropped_tables
    

    
    def objects_to_crops(self, img, tokens, objects, class_thresholds, padding=50):
        """
        Process the bounding boxes produced by the table detection model into
        cropped table images and cropped tokens.
        """

        table_crops = []
        for obj in objects:
            if obj['score'] < class_thresholds[obj['label']]:
                continue

            cropped_table = {}

            bbox = obj['bbox']
            bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

            cropped_img = img.crop(bbox)

            table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
            for token in table_tokens:
                token['bbox'] = [token['bbox'][0]-bbox[0],
                                token['bbox'][1]-bbox[1],
                                token['bbox'][2]-bbox[0],
                                token['bbox'][3]-bbox[1]]

            # If table is predicted to be rotated, rotate cropped image and tokens/words:
            if obj['label'] == 'table rotated':
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token['bbox']
                    bbox = [cropped_img.size[0]-bbox[3]-1,
                            bbox[0],
                            cropped_img.size[0]-bbox[1]-1,
                            bbox[2]]
                    token['bbox'] = bbox

            cropped_table['image'] = cropped_img
            cropped_table['tokens'] = table_tokens

            table_crops.append(cropped_table)

        return table_crops