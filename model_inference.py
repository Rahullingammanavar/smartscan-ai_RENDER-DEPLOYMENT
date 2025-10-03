import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import sys
from typing import Dict, Tuple
from datetime import datetime
import boto3
from botocore.config import Config

# --- Progress Tracker Class for Boto3 ---
class ProgressPercentage(object):
    def __init__(self, filename, filesize):
        self._filename = filename
        self._size = float(filesize)
        self._seen_so_far = 0

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        percentage = (self._seen_so_far / self._size) * 100
        sys.stdout.write(
            f"\r -> Downloading {self._filename}: {self._seen_so_far / (1024*1024):.2f} MB / {self._size / (1024*1024):.2f} MB ({percentage:.2f}%)"
        )
        sys.stdout.flush()

# --- Final Download Function ---
def download_model_from_r2() -> Tuple[str, str]:
    """
    Downloads model and class map from Cloudflare R2 if they don't exist locally.
    Includes longer timeouts and a progress tracker.
    """
    MODEL_FILENAME = os.environ.get('R2_MODEL_KEY', 'model_final.pt')
    CLASS_MAP_FILENAME = os.environ.get('R2_CLASS_MAP_KEY', 'class_map.json')
    
    # For local testing, create a cache directory in the user's home folder
    LOCAL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".skin_model_cache")
    LOCAL_MODEL_PATH = os.path.join(LOCAL_CACHE_DIR, MODEL_FILENAME)
    LOCAL_CLASS_MAP_PATH = os.path.join(LOCAL_CACHE_DIR, CLASS_MAP_FILENAME)

    # For deployment on Render, use the specific path on the persistent disk
    if "RENDER" in os.environ:
        print("Render environment detected. Using /var/data for model cache.")
        RENDER_CACHE_DIR = "/var/data"
        LOCAL_MODEL_PATH = os.path.join(RENDER_CACHE_DIR, MODEL_FILENAME)
        LOCAL_CLASS_MAP_PATH = os.path.join(RENDER_CACHE_DIR, CLASS_MAP_FILENAME)

    if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_CLASS_MAP_PATH):
        print(f"✅ Model and class map already exist in cache: {os.path.dirname(LOCAL_MODEL_PATH)}")
        return LOCAL_MODEL_PATH, LOCAL_CLASS_MAP_PATH

    print("🚀 Model not found in local cache. Starting download from Cloudflare R2...")

    try:
        endpoint_url = os.environ['R2_ENDPOINT_URL']
        access_key_id = os.environ['R2_ACCESS_KEY_ID']
        secret_access_key = os.environ['R2_SECRET_ACCESS_KEY']
        bucket_name = os.environ['R2_BUCKET_NAME']
    except KeyError as e:
        raise RuntimeError(f"❌ Missing required environment variable: {e}")

    # Increase the timeout settings to be more patient
    config = Config(
        connect_timeout=60,  # Increase connect timeout to 60 seconds
        read_timeout=300,    # Increase read timeout to 5 minutes for large files
        retries={'max_attempts': 5}
    )

    s3_client = boto3.client(
        service_name='s3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name='auto',
        config=config # Apply the new timeout settings
    )

    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

    try:
        model_meta = s3_client.head_object(Bucket=bucket_name, Key=MODEL_FILENAME)
        model_size = int(model_meta.get('ContentLength', 0))
        
        progress = ProgressPercentage(MODEL_FILENAME, model_size)
        s3_client.download_file(bucket_name, MODEL_FILENAME, LOCAL_MODEL_PATH, Callback=progress)
        sys.stdout.write("\n") # New line after progress bar is complete
        print(f"✅ Successfully downloaded model to {LOCAL_MODEL_PATH}")

    except Exception as e:
        raise RuntimeError(f"❌ Failed to download model: {e}")

    try:
        class_map_meta = s3_client.head_object(Bucket=bucket_name, Key=CLASS_MAP_FILENAME)
        class_map_size = int(class_map_meta.get('ContentLength', 0))

        progress = ProgressPercentage(CLASS_MAP_FILENAME, class_map_size)
        s3_client.download_file(bucket_name, CLASS_MAP_FILENAME, LOCAL_CLASS_MAP_PATH, Callback=progress)
        sys.stdout.write("\n") # New line after progress bar is complete
        print(f"✅ Successfully downloaded class map to {LOCAL_CLASS_MAP_PATH}")

    except Exception as e:
        raise RuntimeError(f"❌ Failed to download class map: {e}")
        
    return LOCAL_MODEL_PATH, LOCAL_CLASS_MAP_PATH


class SkinDiseaseClassifier:
    def __init__(self, model_path: str, class_map_path: str, device: str = 'auto'):
        self.device = self._get_device(device)
        self.model_path = model_path
        self.class_map_path = class_map_path
        self.class_map = self._load_class_map()
        self.num_classes = len(self.class_map)
        self.model = self._load_model()
        self.transform = self._get_transforms()
        self.medical_info = self._get_medical_info()
    
    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_class_map(self) -> Dict[int, str]:
        try:
            with open(self.class_map_path, 'r') as f:
                class_map = json.load(f)
            return {int(k): v for k, v in class_map.items()}
        except Exception as e:
            raise FileNotFoundError(f"Could not load class map from {self.class_map_path}: {e}")
    
    def _load_model(self) -> nn.Module:
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False) 
            model = models.efficientnet_b3(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)
            model.load_state_dict(checkpoint['model_state'])
            model.to(self.device)
            model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
            print(f"📊 Classes: {list(self.class_map.values())}")
            return model
        except Exception as e:
            raise RuntimeError(f"Could not load model from {self.model_path}: {e}")
            
    def _get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(330),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_medical_info(self) -> Dict[str, Dict]:
        return {
            "Melanocytic_Nevi": {"description": "Benign moles or birthmarks that are usually harmless", "risk_level": "Low", "recommendation": "Regular monitoring recommended", "warning": False},
            "Melanoma": {"description": "A serious form of skin cancer that develops in melanocytes", "risk_level": "High", "recommendation": "Immediate medical consultation strongly recommended", "warning": True},
            "Seborrheic_Keratoses": {"description": "Benign skin growths that appear as waxy, raised lesions", "risk_level": "Low", "recommendation": "Usually harmless, but monitor for changes", "warning": False}
        }
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
        except Exception as e:
            raise ValueError(f"Error preprocessing image {image_path}: {e}")
    
    def predict(self, image_path: str) -> Dict:
        try:
            image_tensor = self.preprocess_image(image_path)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                all_probs = probabilities.squeeze().cpu().numpy()
            
            predictions = []
            for i, (class_idx, class_name) in enumerate(self.class_map.items()):
                prob = all_probs[i]
                medical_info = self.medical_info.get(class_name, {})
                predictions.append({
                    "class": class_name, "confidence": float(prob), **medical_info
                })
            
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            top_prediction = predictions[0]
            
            return {
                "predictions": predictions,
                "top_prediction": top_prediction,
                "overall_risk": "High" if top_prediction.get('warning') else "Low",
                "model_confidence": confidence.item(),
                "model_info": {
                    "architecture": "EfficientNet-B3",
                    "accuracy": "94.04%",
                    "classes": list(self.class_map.values())
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")

# Global model instance
_model_instance = None

def get_model() -> SkinDiseaseClassifier:
    global _model_instance
    if _model_instance is None:
        model_path, class_map_path = download_model_from_r2()
        _model_instance = SkinDiseaseClassifier(model_path, class_map_path)
    return _model_instance

if __name__ == "__main__":
    print("To test this script directly, first set your R2 environment variables.")
    print("Example (PowerShell): $env:R2_ENDPOINT_URL='...'")
    try:
        get_model()
    except Exception as e:
        print(e)