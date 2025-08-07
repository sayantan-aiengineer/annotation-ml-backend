import os
import io
import logging
import numpy as np
import torch
import cv2
import base64
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_single_tag_key, get_choice, is_skipped
from label_studio.core.utils.io import json_load
from label_studio_converter.brush import encode_rle

logger = logging.getLogger(__name__)

class PIDSegmentationModel(LabelStudioMLBase):
    """U-Net based semantic segmentation model for P&ID line classification"""
    
    def __init__(self, **kwargs):
        # Initialize the ML backend
        super(PIDSegmentationModel, self).__init__(**kwargs)
        
        # Model configuration
        self.model_path = os.environ.get('MODEL_PATH', '/data/models/best_pid_unet.pth')
        self.num_classes = int(os.environ.get('NUM_CLASSES', '8'))
        self.img_size = int(os.environ.get('IMG_SIZE', '1024'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class labels corresponding to your 8 classes
        self.class_labels = [
            '0',
            '1',
            '2', 
            '3',
            '4',
            '5',
            '6',
            '7'
        ]
        
        # Model and transforms will be loaded lazily
        self.model = None
        self.transform = None
        
        # Parse labeling config
        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')
        self.from_name = from_name
        self.to_name = to_name
        self.value = value
        
        logger.info(f"Initialized PID Segmentation Model with {self.num_classes} classes")

    def _lazy_init(self):
        """Initialize model and transforms on first use"""
        if self.model is None:
            logger.info("Loading U-Net model...")
            
            # Load the model
            self.model = smp.Unet(
                encoder_name='resnet34',
                encoder_weights=None,  # Don't load imagenet weights
                in_channels=3,
                classes=self.num_classes,
                activation=None
            )
            
            # Load trained weights
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info(f"Loaded model weights from {self.model_path}")
            else:
                logger.warning(f"Model weights not found at {self.model_path}")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
            logger.info("Model and transforms loaded successfully")

    def get_first_tag_occurence(self, control_type, object_type):
        """Get the first occurrence of a tag from labeling config"""
        from_name, to_name, value = None, None, None
        
        for tag_name, tag_info in self.parsed_label_config.items():
            if tag_info['type'] == control_type:
                from_name = tag_name
                to_name = tag_info['to_name'][0]
                value = tag_info['inputs'][0]['value']
                break
                
        return from_name, to_name, value

    def _load_image_from_url(self, url):
        """Load image from URL or local path"""
        try:
            # Handle data URLs (base64 encoded images)
            if url.startswith('data:image'):
                # Extract base64 data
                header, encoded = url.split(',', 1)
                data = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(data)).convert('RGB')
                return np.array(image)
            else:
                # Handle local file paths or URLs
                if url.startswith('/data/'):
                    # Local file in Label Studio
                    image = cv2.imread(url)
                    if image is None:
                        raise ValueError(f"Could not load image from {url}")
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # Handle URLs or other paths
                    image = Image.open(url).convert('RGB')
                    return np.array(image)
        except Exception as e:
            logger.error(f"Error loading image from {url}: {e}")
            raise

    def _predict_mask(self, image):
        """Run inference on a single image"""
        self._lazy_init()
        
        # Preprocess image
        original_h, original_w = image.shape[:2]
        
        # Apply transforms
        augmented = self.transform(image=image)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # Resize back to original size
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (original_w, original_h), 
                              interpolation=cv2.INTER_NEAREST)
        
        return pred_mask

    def _mask_to_rle(self, mask, class_idx):
        """Convert binary mask to RLE format for Label Studio"""
        # Create binary mask for specific class
        binary_mask = (mask == class_idx).astype(np.uint8)
        
        # Skip if no pixels of this class
        if binary_mask.sum() == 0:
            return None
            
        # Convert to RLE
        rle = encode_rle(binary_mask)
        return rle

    def predict(self, tasks, context=None, **kwargs):
        """Main prediction method called by Label Studio"""
        predictions = []
        
        logger.info(f"Processing {len(tasks)} tasks")
        
        for task in tasks:
            try:
                # Skip if task is already annotated
                if is_skipped(task):
                    predictions.append(ModelResponse(task=task, result=[]))
                    continue
                
                # Get image URL from task data
                image_url = task['data'][self.value]
                
                # Load and process image
                image = self._load_image_from_url(image_url)
                
                # Get prediction mask
                pred_mask = self._predict_mask(image)
                
                # Convert mask to Label Studio format
                result = []
                
                for class_idx in range(1, self.num_classes):  # Skip background (class 0)
                    rle = self._mask_to_rle(pred_mask, class_idx)
                    
                    if rle is not None:
                        # Create brush annotation
                        brush_result = {
                            'from_name': self.from_name,
                            'to_name': self.to_name,
                            'type': 'brushlabels',
                            'value': {
                                'brushlabels': [self.class_labels[class_idx]],
                                'rle': rle
                            }
                        }
                        result.append(brush_result)
                
                # Create model response
                model_response = ModelResponse(
                    task=task,
                    result=result,
                    score=0.85  # Confidence score
                )
                predictions.append(model_response)
                
                logger.info(f"Generated {len(result)} predictions for task {task.get('id', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Error processing task {task.get('id', 'unknown')}: {e}")
                # Return empty prediction on error
                predictions.append(ModelResponse(task=task, result=[]))
        
        return predictions

    def fit(self, event, data, **kwargs):
        """Training method - can be used for online learning or model updates"""
        logger.info(f"Fit method called with event: {event}")
        
        # For now, we'll just log the event
        # You could implement online learning or model fine-tuning here
        
        # Extract annotations from the data if needed
        if event in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            logger.info("New annotation available for potential model updating")
            
        return {'status': 'ok'}

# Optional: Add health check endpoint
def health_check():
    """Health check for the ML backend"""
    return {'status': 'healthy', 'model': 'PID Segmentation U-Net'}
