import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
import timm
import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget


def _reshape_transform_swin(tensor):
    return tensor.transpose(2, 3).transpose(1, 2)


def _reshape_transform_vit(tensor, height=16, width=16):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


MODEL_KEYS = ["Swin", "ResNet", "EfficientNet", "ViT"]
ENSEMBLE_KEY = "Ensemble"

# Test 세트 평가 결과 (Dataset/Test, 2026-06-09) — model_metrics.json 으로도 자동 반영
MODEL_CONFIG = {
    "Swin": {
        "file": "best_swin_model.pth",
        "label": "Swin Transformer",
        "temperature": 2.5,
        "f1_score": 0.9208,
        "recall": 0.9605,
        "accuracy": 0.9167,
        "precision": 0.8842,
    },
    "ResNet": {
        "file": "best_resnet_model.pth",
        "label": "ResNet-50",
        "temperature": 2.5,
        "f1_score": 0.8457,
        "recall": 0.9592,
        "accuracy": 0.8237,
        "precision": 0.7561,
    },
    "EfficientNet": {
        "file": "best_effib4_model.pth",
        "label": "EfficientNet-B4",
        "temperature": 2.5,
        "f1_score": 0.9298,
        "recall": 0.9547,
        "accuracy": 0.9274,
        "precision": 0.9062,
    },
    "ViT": {
        "file": "best_vit_model.pth",
        "label": "ViT-Small",
        "temperature": 2.5,
        "f1_score": 0.9102,
        "recall": 0.9627,
        "accuracy": 0.9044,
        "precision": 0.8632,
    },
    ENSEMBLE_KEY: {
        "file": "ensemble (4 models)",
        "label": "앙상블 (Ensemble)",
        "temperature": 2.5,
    },
}


PERFORMANCE_METRICS = ("f1_score", "recall", "accuracy", "precision")
ENSEMBLE_SHARPNESS = 30.0
METRICS_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "model_metrics.json")


def load_ensemble_metrics_from_file(path: str = METRICS_JSON) -> bool:
    """results/model_metrics.json 의 Test/Val 평가 결과를 MODEL_CONFIG에 반영."""
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    for key in MODEL_KEYS:
        entry = report.get("models", {}).get(key)
        if not entry:
            continue
        for metric in PERFORMANCE_METRICS:
            if metric in entry:
                MODEL_CONFIG[key][metric] = float(entry[metric])
    return True


load_ensemble_metrics_from_file()


def get_model_metrics(model_key: str) -> dict[str, float]:
    cfg = MODEL_CONFIG[model_key]
    return {m: float(cfg.get(m, 0.0)) for m in PERFORMANCE_METRICS}


def get_model_composite_score(model_key: str) -> float:
    """F1 · Recall · Accuracy · Precision 4항목 산술 평균."""
    metrics = get_model_metrics(model_key)
    return sum(metrics.values()) / len(PERFORMANCE_METRICS)


def get_best_model_key() -> str:
    return max(MODEL_KEYS, key=get_model_composite_score)


def get_ensemble_weights() -> dict[str, float]:
    """
    최고 성능 모델 기준 상대 가중치.

    weight_i ∝ exp(SHARPNESS × (score_i − score_best))
    → 1~2%p 차이도 상위 모델에 가중치가 크게 몰리고,
      오류가 많은 하위 모델의 영향은 줄어듭니다.
    """
    scores = {k: get_model_composite_score(k) for k in MODEL_KEYS}
    best_key = max(scores, key=scores.get)
    best_score = scores[best_key]

    raw = {
        k: float(np.exp(ENSEMBLE_SHARPNESS * (scores[k] - best_score)))
        for k in MODEL_KEYS
    }
    total = sum(raw.values()) or 1.0
    return {k: raw[k] / total for k in MODEL_KEYS}


def get_ensemble_weight_info() -> dict:
    """UI/저장용 — 모델별 점수·가중치·기준 모델 정보."""
    scores = {k: get_model_composite_score(k) for k in MODEL_KEYS}
    best_key = get_best_model_key()
    weights = get_ensemble_weights()
    return {
        "best_model_key": best_key,
        "best_model_label": MODEL_CONFIG[best_key]["label"],
        "best_score": scores[best_key],
        "sharpness": ENSEMBLE_SHARPNESS,
        "scores": scores,
        "weights": weights,
    }


def temperature_scale(logit: float, temperature: float = 1.0) -> tuple[float, float]:
    t = max(float(temperature), 1e-6)
    real_prob = torch.sigmoid(torch.tensor(logit / t)).item()
    return real_prob, 1.0 - real_prob


class DeepfakeEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.model_swin = timm.create_model("swin_tiny_patch4_window7_224", num_classes=1, img_size=256)
        self.load_model(self.model_swin, "best_swin_model.pth")

        self.model_resnet = models.resnet50(weights=None)
        self.model_resnet.fc = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )
        self.load_model(self.model_resnet, "best_resnet_model.pth")

        self.model_effi = timm.create_model("tf_efficientnet_b4", num_classes=1)
        self.model_effi.classifier = nn.Linear(self.model_effi.classifier.in_features, 1)
        self.load_model(self.model_effi, "best_effib4_model.pth")

        self.model_vit = timm.create_model("vit_small_patch16_224", num_classes=1, img_size=256)
        self.load_model(self.model_vit, "best_vit_model.pth")

        for m in [self.model_swin, self.model_resnet, self.model_effi, self.model_vit]:
            m.to(self.device).eval()

    def load_model(self, model, file_name):
        path = os.path.join(self.models_dir, file_name)
        state_dict = torch.load(path, map_location=self.device)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    def _get_nets(self):
        return {
            "Swin": self.model_swin,
            "ResNet": self.model_resnet,
            "EfficientNet": self.model_effi,
            "ViT": self.model_vit,
        }

    def _get_target_layers(self, model_name):
        return {
            "Swin": [self.model_swin.layers[-1].blocks[-1].norm1],
            "ResNet": [self.model_resnet.layer4[-1]],
            "EfficientNet": [self.model_effi.conv_head],
            "ViT": [self.model_vit.blocks[-1].norm1],
        }[model_name]

    def _build_cam_kwargs(self, model_name, net):
        cam_kwargs = {"model": net, "target_layers": self._get_target_layers(model_name)}
        if model_name == "Swin":
            cam_kwargs["reshape_transform"] = _reshape_transform_swin
            cam_kwargs["target_layers"] = [self.model_swin.layers[-1].blocks[-1]]
        elif model_name == "ViT":
            cam_kwargs["reshape_transform"] = _reshape_transform_vit
        return cam_kwargs

    def _compute_gradcam(self, model_name, input_tensor, is_fake):
        net = self._get_nets()[model_name]
        cam_kwargs = self._build_cam_kwargs(model_name, net)
        cam_target = BinaryClassifierOutputTarget(0 if is_fake else 1)
        cam = GradCAM(**cam_kwargs)
        return cam(input_tensor=input_tensor, targets=[cam_target])[0, :]

    def _predict_single(self, model_name, input_tensor):
        net = self._get_nets()[model_name]
        with torch.no_grad():
            logit = net(input_tensor).item()
        temperature = MODEL_CONFIG[model_name].get("temperature", 1.0)
        real_prob, fake_prob = temperature_scale(logit, temperature)
        return {
            "model_key": model_name,
            "logit": logit,
            "temperature": temperature,
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "is_fake": fake_prob > 0.5,
        }

    def _overlay_heatmap(self, image_path, grayscale_cam):
        img_cv = cv2.cvtColor(cv2.resize(cv2.imread(image_path), (256, 256)), cv2.COLOR_BGR2RGB)
        return show_cam_on_image(np.float32(img_cv) / 255.0, grayscale_cam, use_rgb=True)

    def process_image(self, image_path, model_name):
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        if model_name == ENSEMBLE_KEY:
            return self._process_ensemble(image_path, input_tensor)

        pred = self._predict_single(model_name, input_tensor)
        grayscale_cam = self._compute_gradcam(model_name, input_tensor, pred["is_fake"])
        heatmap = self._overlay_heatmap(image_path, grayscale_cam)
        label = "Fake" if pred["is_fake"] else "Real"
        return (
            pred["real_prob"], pred["fake_prob"], pred["logit"],
            heatmap, label, pred["temperature"], None,
        )

    def _process_ensemble(self, image_path, input_tensor):
        weight_info = get_ensemble_weight_info()
        weights = weight_info["weights"]
        member_results = []

        weighted_fake = 0.0
        weighted_logit = 0.0
        weighted_temp = 0.0
        cam_accum = None

        for key in MODEL_KEYS:
            pred = self._predict_single(key, input_tensor)
            w = weights[key]
            weighted_fake += w * pred["fake_prob"]
            weighted_logit += w * pred["logit"]
            weighted_temp += w * pred["temperature"]

            cam = self._compute_gradcam(key, input_tensor, pred["is_fake"])
            cam_accum = w * cam if cam_accum is None else cam_accum + w * cam

            member_results.append({
                "model_key": key,
                "model_label": MODEL_CONFIG[key]["label"],
                "weight": w,
                "metrics": get_model_metrics(key),
                "composite_score": get_model_composite_score(key),
                "relative_to_best": get_model_composite_score(key) - weight_info["best_score"],
                "real_prob": pred["real_prob"],
                "fake_prob": pred["fake_prob"],
                "logit": pred["logit"],
                "is_fake": pred["is_fake"],
            })

        fake_prob = weighted_fake
        cam_min, cam_max = cam_accum.min(), cam_accum.max()
        cam_norm = (cam_accum - cam_min) / (cam_max - cam_min + 1e-8)
        heatmap = self._overlay_heatmap(image_path, cam_norm)

        ensemble_details = {
            "weight_info": weight_info,
            "weights": weights,
            "members": member_results,
        }

        label = "Fake" if fake_prob > 0.5 else "Real"
        return (
            1.0 - fake_prob, fake_prob, weighted_logit,
            heatmap, label, weighted_temp, ensemble_details,
        )

    @staticmethod
    def classify(real_prob, fake_prob):
        is_fake = fake_prob > 0.5
        return {
            "is_fake": is_fake,
            "label": "위조(Fake)" if is_fake else "원본(Real)",
            "confidence": fake_prob if is_fake else real_prob,
            "real_prob": real_prob,
            "fake_prob": fake_prob,
        }

    @staticmethod
    def format_ensemble_weights_text() -> str:
        info = get_ensemble_weight_info()
        best = info["best_model_label"]
        lines = [
            f"기준 모델: {best} (종합 {info['best_score']:.3f})",
            f"집중도 SHARPNESS={info['sharpness']:.0f}",
            "",
        ]
        for key in sorted(MODEL_KEYS, key=lambda k: info["weights"][k], reverse=True):
            w = info["weights"][key]
            m = get_model_metrics(key)
            comp = info["scores"][key]
            gap = comp - info["best_score"]
            lines.append(
                f"  {MODEL_CONFIG[key]['label']}: {w * 100:.1f}% "
                f"(종합={comp:.3f}, 최고 대비 {gap:+.3f}, "
                f"F1={m['f1_score']:.2f} Rec={m['recall']:.2f} "
                f"Acc={m['accuracy']:.2f} Prec={m['precision']:.2f})"
            )
        return "\n".join(lines)
