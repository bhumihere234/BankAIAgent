from doctr.io import DocumentFile, read_img_as_tensor
from doctr.models import ocr_predictor



class TextExtract:
    def __init__(self) -> None:
        self.model = ocr_predictor(pretrained=True)
    
    def run_ocr(self, image):
        doc = DocumentFile.from_images(image)
        result = self.model(doc)
        return result
    
    def get_ocr_text_and_bbox(self, result):
        lines = []
        for page in result.pages:
            height, width = page.dimensions

            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join([word.value for word in line.words])
                    for word in line.words:
                        word_dict = {
                            "word": word.value,
                            "bbox": self.rescale_bbox(word.geometry, width, height)
                        }
                        #print(word.geometry)
                        lines.append(word_dict)
        return lines
    
    def rescale_bbox(self, normalized_bbox, image_width, image_height):
        """
        Rescale a normalized bounding box to the original image dimensions.
        
        Args:
        normalized_bbox (tuple): (x1, y1, x2, y2) coordinates of the normalized bounding box.
        image_width (int): Width of the original image.
        image_height (int): Height of the original image.
        
        Returns:
        tuple: Rescaled bounding box in the original image dimensions.
        """
        normalized_bbox = normalized_bbox[0][0],normalized_bbox[0][1],normalized_bbox[1][0], normalized_bbox[1][1]
        x1, y1, x2, y2 = normalized_bbox
        
        rescaled_x1 = x1 * image_width
        rescaled_y1 = y1 * image_height
        rescaled_x2 = x2 * image_width
        rescaled_y2 = y2 * image_height
        
        return [rescaled_x1, rescaled_y1, rescaled_x2, rescaled_y2]