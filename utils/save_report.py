import json
import os
import shutil
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def _get_font(size: int = 16):
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _build_summary_image(image_path: str, heatmap: np.ndarray, lines: list[str]) -> np.ndarray:
    """원본 + Grad-CAM + 결과 텍스트를 한 장으로 합성."""
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise ValueError(f"원본 이미지를 읽을 수 없습니다: {image_path}")

    orig_rgb = cv2.cvtColor(cv2.resize(orig_bgr, (320, 320)), cv2.COLOR_BGR2RGB)
    grad_rgb = cv2.resize(heatmap, (320, 320))
    image_row = np.hstack([orig_rgb, grad_rgb])

    line_h = 28
    text_h = max(140, line_h * len(lines) + 24)
    canvas = Image.new("RGB", (640, 320 + text_h), (255, 255, 255))
    canvas.paste(Image.fromarray(image_row), (0, 0))

    draw = ImageDraw.Draw(canvas)
    title_font = _get_font(18)
    body_font = _get_font(15)
    draw.text((16, 328), "딥페이크 탐지 분석 결과", fill=(34, 34, 34), font=title_font)

    y = 328 + 32
    for line in lines:
        draw.text((16, y), line, fill=(60, 60, 60), font=body_font)
        y += line_h

    return np.array(canvas)


def save_analysis_report(
    image_path: str,
    heatmap: np.ndarray,
    model_key: str,
    model_label: str,
    model_file: str,
    real_prob: float,
    fake_prob: float,
    logit: float,
    temperature: float,
    verdict_label: str,
    confidence: float,
    is_fake: bool,
    ensemble_details: dict | None = None,
) -> str:
    """분석 결과를 results/ 하위 폴더에 저장하고 폴더 경로를 반환."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{model_key}"
    save_dir = os.path.join(RESULTS_DIR, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict_ko = "위조 (Fake)" if is_fake else "원본 (Real)"
    src_name = os.path.basename(image_path)
    ext = os.path.splitext(image_path)[1].lower() or ".jpg"

    # 1) 원본 사진
    shutil.copy2(image_path, os.path.join(save_dir, f"original{ext}"))

    # 2) Grad-CAM
    gradcam_path = os.path.join(save_dir, "gradcam.jpg")
    cv2.imwrite(gradcam_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

    # 3) 결과 텍스트
    report_lines = [
        f"저장 시각: {saved_at}",
        f"원본 파일: {src_name}",
        f"사용 모델: {model_label} ({model_file})",
        f"판정: {verdict_ko}",
        f"온도 보정 (T={temperature:.1f})",
        f"Real {real_prob * 100:.1f}%  ·  Fake {fake_prob * 100:.1f}%",
        f"판정 신뢰도: {confidence * 100:.1f}%",
        f"Logit: {logit:.4f}",
    ]

    if ensemble_details:
        report_lines.append("")
        wi = ensemble_details.get("weight_info", {})
        if wi:
            report_lines.append(
                f"[앙상블 · 기준={wi.get('best_model_label', '?')} "
                f"(종합 {wi.get('best_score', 0):.3f}), "
                f"SHARPNESS={wi.get('sharpness', 0):.0f}]"
            )
        else:
            report_lines.append("[앙상블 · 최고 성능 모델 기준 상대 가중치]")
        for m in ensemble_details["members"]:
            tag = "Fake" if m["is_fake"] else "Real"
            met = m["metrics"]
            gap = m.get("relative_to_best", 0.0)
            report_lines.append(
                f"  - {m['model_label']}: {tag}, Fake {m['fake_prob'] * 100:.1f}%, "
                f"가중치 {m['weight'] * 100:.1f}% "
                f"(종합={m['composite_score']:.3f}, 최고 대비 {gap:+.3f}, "
                f"F1={met['f1_score']:.2f}, Rec={met['recall']:.2f}, "
                f"Acc={met['accuracy']:.2f}, Prec={met['precision']:.2f})"
            )

    report_txt_path = os.path.join(save_dir, "report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write("  딥페이크 탐지 분석 기록\n")
        f.write("=" * 40 + "\n\n")
        f.write("\n".join(report_lines))
        f.write("\n")

    report_data = {
        "saved_at": saved_at,
        "source_image": src_name,
        "source_path": image_path,
        "model_key": model_key,
        "model_label": model_label,
        "model_file": model_file,
        "verdict": verdict_ko,
        "verdict_en": "Fake" if is_fake else "Real",
        "real_prob": round(real_prob, 6),
        "fake_prob": round(fake_prob, 6),
        "confidence": round(confidence, 6),
        "logit": round(logit, 6),
        "temperature": temperature,
        "ensemble_details": ensemble_details,
        "files": {
            "original": f"original{ext}",
            "gradcam": "gradcam.jpg",
            "summary": "summary.jpg",
            "report": "report.txt",
        },
    }

    report_json_path = os.path.join(save_dir, "report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # 4) 요약 이미지 (원본 + Grad-CAM + 결과)
    summary_rgb = _build_summary_image(image_path, heatmap, report_lines[2:])
    summary_path = os.path.join(save_dir, "summary.jpg")
    cv2.imwrite(summary_path, cv2.cvtColor(summary_rgb, cv2.COLOR_RGB2BGR))

    return save_dir
