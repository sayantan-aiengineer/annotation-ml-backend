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
from label_studio_converter.brush import encode_rle, decode_rle

logger = logging.getLogger(__name__)

class PIDSegmentationModel(LabelStudioMLBase):
    """U-Net based semantic segmentation model for P&ID line classification using brush labels"""
    
    def __init__(self, **kwargs):
        super(PIDSegmentationModel, self).__init__(**kwargs)
        
        # Model configuration
        self.model_path = os.environ.get('MODEL_PATH', '/data/models/best_pid_unet.pth')
        self.num_classes = int(os.environ.get('NUM_CLASSES', '8'))
        self.img_size = int(os.environ.get('IMG_SIZE', '1024'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class labels (index 0 is background, 1-7 are your line types)
        self.class_labels = [
            '0',  # index 0 - not used in Label Studio
            '1',       # index 1
            '2',      # index 2
            '3',   # index 3
            '4',    # index 4
            '5',   # index 5
            '6',   # index 6
            '7'       # index 7
        ]
        
        # Model and transforms will be loaded lazily
        self.model = None
        self.transform = None
        
        # Parse labeling config for brush labels
        self._parse_config()
        
        logger.info(f"Initialized PID Segmentation Model with {self.num_classes} classes")

    def _parse_config(self):
        """Parse Label Studio config to get brush label information"""
        self.from_name = None
        self.to_name = None
        self.value = None
        
        for tag_name, tag_info in self.parsed_label_config.items():
            if tag_info['type'] == 'BrushLabels':
                self.from_name = tag_name
                self.to_name = tag_info['to_name'][0]
                self.value = tag_info['inputs']['value']
                break
        
        if not self.from_name:
            logger.warning("No BrushLabels found in labeling config")

    def _lazy_init(self):
        """Initialize model and transforms on first use"""
        if self.model is None:
            logger.info("Loading U-Net model...")
            
            # Load the model
            self.model = smp.Unet(
                encoder_name='resnet50',
                encoder_weights=None,
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

    def _load_image_from_url(self, url):
        """Load image from URL or local path"""
        try:
            # Use Label Studio's built-in method to handle all URL types
            local_path = self.get_local_path(url)
            image = cv2.imread(local_path)
            if image is None:
                raise ValueError(f"Could not load image from {local_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image from {url}: {e}")
            raise

    def _predict_mask(self, image):
        """Run inference on a single image"""
        self._lazy_init()
        
        # Store original dimensions
        original_h, original_w = image.shape[:2]
        
        # Apply transforms
        augmented = self.transform(image=image)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # Resize back to original size
        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8), 
            (original_w, original_h), 
            interpolation=cv2.INTER_NEAREST
        )
        
        return pred_mask

    def _mask_to_rle(self, mask, class_idx, original_width, original_height):
        """Convert binary mask to RLE format for Label Studio brush labels"""
        # Create binary mask for specific class
        binary_mask = (mask == class_idx).astype(np.uint8)
        
        # Skip if no pixels of this class
        if binary_mask.sum() == 0:
            return None
        
        # Convert to RLE using Label Studio's encoder
        try:
            rle = encode_rle(binary_mask)
            return rle
        except Exception as e:
            logger.error(f"Error encoding RLE for class {class_idx}: {e}")
            return None

    def predict(self, tasks, context=None, **kwargs):
        """Main prediction method called by Label Studio"""
        predictions = []
        
        logger.info(f"Processing {len(tasks)} tasks")
        
        for task in tasks:
            try:
                # Get image URL from task data
                if not self.value or self.value not in task['data']:
                    logger.error(f"No image data found in task")
                    predictions.append(ModelResponse(task=task, result=[]))
                    continue
                
                image_url = task['data'][self.value]
                
                # Load and process image
                image = self._load_image_from_url(image_url)
                original_height, original_width = image.shape[:2]
                
                # Get prediction mask
                pred_mask = self._predict_mask(image)
                
                # Convert mask to Label Studio brush format
                result = []
                
                # Process each class (skip background class 0)
                for class_idx in range(1, self.num_classes):
                    rle = self._mask_to_rle(pred_mask, class_idx, original_width, original_height)
                    
                    if rle is not None:
                        # Create brush annotation result
                        brush_result = {
                            'from_name': self.from_name,
                            'to_name': self.to_name,
                            'type': 'brushlabels',
                            'value': {
                                'brushlabels': [self.class_labels[class_idx]],
                                'rle': rle,
                                'format': 'rle'
                            },
                            'original_width': original_width,
                            'original_height': original_height,
                            'image_rotation': 0
                        }
                        result.append(brush_result)
                
                # Create model response
                predictions.append(ModelResponse(
                    task=task,
                    result=result,
                    score=0.85,  # Confidence score
                    model_version=self.get("model_version", "1.0.0")
                ))
                
                logger.info(f"Generated {len(result)} predictions for task {task.get('id', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Error processing task {task.get('id', 'unknown')}: {e}")
                predictions.append(ModelResponse(task=task, result=[]))
        
        return predictions

    def fit(self, event, data, **kwargs):
        """Training method - can be used for online learning"""
        logger.info(f"Fit method called with event: {event}")
        
        # You could implement online learning or model fine-tuning here
        if event in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            logger.info("New annotation available for potential model updating")
            
        return {'status': 'ok'}

