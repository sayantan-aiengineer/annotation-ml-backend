import os
import logging
import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)

class PIDSegmentationModel(LabelStudioMLBase):
    """U-Net based semantic segmentation backend for Label Studio using brush labels."""

    def __init__(self, **kwargs):
        super(PIDSegmentationModel, self).__init__(**kwargs)
        # Model configuration
        self.model_path = os.environ.get('MODEL_PATH', '/data/models/best_pid_unet.pth')
        self.num_classes = int(os.environ.get('NUM_CLASSES', '8'))
        self.img_size = int(os.environ.get('IMG_SIZE', '1024'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Class labels (index 0 unused)
        self.class_labels = [
            'Background', '1', '2', '3', '4', '5', '6', '7'
        ]

        # Placeholder for lazy initialization
        self.model = None
        self.transform = None

        # Parse Label Studio config
        self._parse_config()
        logger.info(f"Initialized PIDSegmentationModel with {self.num_classes} classes")

    def _parse_config(self):
        """Extract from_name, to_name, and data key from labeling config."""
        self.from_name = None
        self.to_name = None
        self.value = None

        for tag_name, tag_info in self.parsed_label_config.items():
            if tag_info.get('type') == 'BrushLabels':
                self.from_name = tag_name
                self.to_name = tag_info.get('to_name', [None])[0]
                inputs = tag_info.get('inputs', [])
                if isinstance(inputs, list) and inputs:
                    self.value = inputs[0].get('value')
                break

        if not all([self.from_name, self.to_name, self.value]):
            logger.warning(
                f"BrushLabels config incomplete: from_name={self.from_name}, "
                f"to_name={self.to_name}, value={self.value}"
            )

    def _lazy_init(self):
        """Load model weights and set up transforms."""
        if self.model is None:
            logger.info("Loading U-Net model and transforms...")
            # Initialize U-Net
            self.model = smp.Unet(
                encoder_name='resnet50',
                encoder_weights=None,
                in_channels=3,
                classes=self.num_classes,
                activation=None
            )
            # Load weights
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info(f"Loaded model weights from {self.model_path}")
            else:
                logger.warning(f"Model weights not found at {self.model_path}")
            self.model.to(self.device).eval()
            # Define transforms
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def _load_image(self, url, task_id):
        """Fetch image using Label Studio utility (handles uploads, local/cloud storage)."""
        local_path = self.get_local_path(url, task_id=task_id)
        image = cv2.imread(local_path)
        if image is None:
            raise ValueError(f"Failed to load image from {local_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _predict_mask(self, image: np.ndarray) -> np.ndarray:
        """Run inference on image and return class mask."""
        self._lazy_init()
        h, w = image.shape[:2]
        augmented = self.transform(image=image)
        inp = augmented['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(inp)
            mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # Resize mask to original dimensions
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def _mask_to_rle(self, mask: np.ndarray, class_idx: int):
        """Convert binary mask for one class to RLE (brushlabels format)."""
        bin_mask = (mask == class_idx).astype(np.uint8)
        if bin_mask.sum() == 0:
            return None
        # Column-major flatten
        pixels = bin_mask.T.flatten()
        runs = []
        pos = 0
        while pos < len(pixels):
            if pixels[pos]:
                start = pos
                while pos < len(pixels) and pixels[pos]:
                    pos += 1
                runs.extend([start + 1, pos - start])
            pos += 1
        return {'size': list(bin_mask.shape), 'counts': runs}

    def predict(self, tasks, context=None, **kwargs):
        """Generate predictions for Label Studio."""
        predictions = []
        for task in tasks:
            try:
                img_url = task['data'].get(self.value)
                img = self._load_image(img_url, task_id=task['id'])
                mask = self._predict_mask(img)
                h, w = img.shape[:2]
                results = []
                for cls in range(1, self.num_classes):
                    rle = self._mask_to_rle(mask, cls)
                    if rle:
                        results.append({
                            'from_name': self.from_name,
                            'to_name': self.to_name,
                            'type': 'brushlabels',
                            'value': {
                                'brushlabels': [self.class_labels[cls]],
                                'rle': rle,
                                'format': 'rle'
                            },
                            'original_width': w,
                            'original_height': h,
                            'image_rotation': 0
                        })
                predictions.append({
                    'task': task,
                    'result': results,
                    'score': 0.85,
                    'model_version': kwargs.get('model_version', '1.0.0')
                })
            except Exception as e:
                logger.error(f"Error processing task {task.get('id')}: {e}")
                predictions.append({
                    'task': task,
                    'result': [],
                    'score': 0.0,
                    'model_version': kwargs.get('model_version', '1.0.0')
                })
        return {'predictions': predictions}

    def fit(self, event, data, **kwargs):
        """Handle training events (optional)."""
        logger.info(f"Fit called with event: {event}")
        return {'status': 'ok'}
