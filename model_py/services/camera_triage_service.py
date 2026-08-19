import base64
import json
import re

import cv2

from model_py.config.settings import VISION_MODEL
from model_py.prompts.camera_triage_prompt import CAMERA_TRIAGE_SYSTEM_PROMPT, CAMERA_TRIAGE_USER_PROMPT
from model_py.services.client import vision_client


def _sample_frames(video_path: str, sample_count: int = 3) -> list[str]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return []
    total_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    positions = [int(total_frames * ratio) for ratio in (0.2, 0.5, 0.8)][:sample_count]
    encoded_frames = []
    try:
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(position, total_frames - 1))
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            if width > 640:
                frame = cv2.resize(frame, (640, max(1, int(height * 640 / width))))
            ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if ok:
                encoded_frames.append(base64.b64encode(buffer).decode('utf-8'))
    finally:
        capture.release()
    return encoded_frames


def _parse_json(raw_text: str) -> dict:
    cleaned = (raw_text or '').strip()
    if '```' in cleaned:
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find('{'), cleaned.rfind('}')
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


def analyze_camera_triage(camera_videos: list[tuple[str, str]]) -> dict:
    content = [{"type": "text", "text": CAMERA_TRIAGE_USER_PROMPT}]
    sampled_ids = []
    for camera_id, video_path in camera_videos:
        frames = _sample_frames(video_path)
        if not frames:
            continue
        sampled_ids.append(camera_id)
        content.append({"type": "text", "text": f"{camera_id} temsilî kareleri:"})
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            })

    if not sampled_ids:
        raise ValueError("Hiçbir videodan temsilî kare çıkarılamadı.")

    response = vision_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": CAMERA_TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=1400,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    result = _parse_json(response.choices[0].message.content)
    result["mode"] = "quick_triage"
    result["sampled_camera_ids"] = sampled_ids
    return result
