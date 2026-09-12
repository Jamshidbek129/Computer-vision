import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import mediapipe as mp
import time
import os
import math
from collections import deque, Counter


# =========================
# 1. MODEL
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["fist", "palm", "point"]

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model_path = os.path.join(
    os.path.dirname(__file__),
    "gesture_efficientnet_b0.pth"
)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# =========================
# 2. IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 3. MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# =========================
# 4. CROP SOZLAMALARI
# =========================

PADDING_RATIO = 0.2


def get_square_crop(frame, hand_landmarks):
    """Qo'l atrofida kvadrat (aspect-ratio buzilmagan) crop qaytaradi."""

    h, w = frame.shape[:2]

    x_values = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_values = [int(lm.y * h) for lm in hand_landmarks.landmark]

    x1_raw = min(x_values)
    y1_raw = min(y_values)
    x2_raw = max(x_values)
    y2_raw = max(y_values)

    box_w = x2_raw - x1_raw
    box_h = y2_raw - y1_raw

    size = max(box_w, box_h)

    pad = int(size * PADDING_RATIO)
    size += pad * 2

    cx = (x1_raw + x2_raw) // 2
    cy = (y1_raw + y2_raw) // 2

    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, w)
    y2 = min(cy + size // 2, h)

    return x1, y1, x2, y2


# =========================
# 5. RASENGAN EFFEKTI
# =========================

# Palm necha soniya ushlab turilsa rasengan paydo bo'la boshlaydi
RASENGAN_TRIGGER_TIME = 2

# Trigger'dan keyin to'liq o'lchamga yetishi uchun ketadigan vaqt
RASENGAN_GROW_TIME = 0.8

palm_hold_start = None


def draw_rasengan(frame, center, box_size, animation_time, progress):
    """
    Kaft markazida ko'k rangli, aylanuvchi, pulslanuvchi shar chizadi.
    progress: 0.0 (hali ko'rinmaydi) dan 1.0 (to'liq o'lcham) gacha.
    """

    if progress <= 0:
        return

    cx, cy = center

    # Pulslanish effekti (nafas olish kabi kattalashib-kichraydi)
    pulse = 1.0 + 0.08 * math.sin(animation_time * 10)

    max_radius = int(box_size * 0.35)
    radius = max(int(max_radius * progress * pulse), 3)

    overlay = frame.copy()

    # Yadro - konsentrik doiralar bilan ko'k gradient hosil qilamiz
    steps = max(radius // 4, 1)
    for i in range(steps, 0, -1):
        r = int(radius * i / steps)
        # Markazga yaqinroq - ochroq ko'k/oq, chetga - to'q ko'k
        t = i / steps
        color = (
            255,                       # B
            int(120 + 100 * t),        # G
            int(40 + 60 * t)           # R
        )
        cv2.circle(overlay, (cx, cy), r, color, -1)

    frame_view = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    frame[:] = frame_view

    # Aylanuvchi spiral chiziqlar (Naruto rasenganidagi "shovqin" effekti)
    num_arms = 3
    for k in range(num_arms):
        base_angle = animation_time * 260 + k * (360 / num_arms)

        points = []
        for step in range(0, 14):
            t = step / 13
            angle = math.radians(base_angle + t * 160)
            r = radius * t

            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            points.append((x, y))

        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (255, 255, 255), 1)

    # Tashqi halqa
    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)


# =========================
# 6. GESTURE SEQUENCE
# =========================

TARGET_SEQUENCE = ["fist", "palm", "point"]

current_sequence = []

last_gesture = None
last_gesture_time = 0

GESTURE_DELAY = 0.6
CONFIDENCE_THRESHOLD = 0.70

SMOOTHING_WINDOW = 7
recent_predictions = deque(maxlen=SMOOTHING_WINDOW)


# =========================
# 7. LIVE VIDEO COUNT
# =========================

video_count = 1
MAX_VIDEOS = 6

gesture_history = []


# =========================
# 8. LIVE VIDEO SPLIT
# =========================

def create_live_videos(frame, count):

    h, w = frame.shape[:2]

    pane_width = w // count

    videos = []

    for i in range(count):

        video = cv2.resize(
            frame,
            (pane_width, h)
        )

        cv2.rectangle(
            video,
            (0, 0),
            (pane_width - 1, h - 1),
            (255, 255, 255),
            2
        )

        if i == 0:
            text = "ORIGINAL"
        else:
            text = f"CLONE {i}"

        cv2.putText(
            video,
            text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        videos.append(video)

    result = cv2.hconcat(videos)

    return result


# =========================
# 9. CAMERA
# =========================
cv2.namedWindow(
    "Naruto Shadow Clone",
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    "Naruto Shadow Clone",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera topilmadi!")
    exit()

print("Camera started!")
print("Target sequence:", TARGET_SEQUENCE)


# =========================
# 10. MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame olinmadi!")
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # =====================
    # HAND DETECTION
    # =====================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    predicted_gesture = None
    confidence = 0

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        x1, y1, x2, y2 = get_square_crop(frame, hand_landmarks)

        hand_crop = frame[y1:y2, x1:x2]

        if hand_crop.size > 0:

            # =====================
            # MODEL PREDICTION
            # =====================

            pil_image = Image.fromarray(
                cv2.cvtColor(
                    hand_crop,
                    cv2.COLOR_BGR2RGB
                )
            )

            input_tensor = transform(
                pil_image
            ).unsqueeze(0).to(device)

            with torch.no_grad():

                output = model(input_tensor)

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                confidence, predicted = torch.max(
                    probabilities,
                    1
                )

                confidence = confidence.item()

                predicted_gesture = class_names[
                    predicted.item()
                ]

                print({
                    class_names[i]: round(probabilities[0][i].item(), 4)
                    for i in range(len(class_names))
                })

            # =====================
            # DRAWING (landmark, bbox, label)
            # =====================

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{predicted_gesture} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # =========================
    # SMOOTHING (majority vote)
    # =========================

    if (
        predicted_gesture is not None
        and confidence >= CONFIDENCE_THRESHOLD
    ):
        recent_predictions.append(predicted_gesture)
    else:
        recent_predictions.append(None)

    smoothed_gesture = None

    if len(recent_predictions) == SMOOTHING_WINDOW:

        valid_preds = [p for p in recent_predictions if p is not None]

        if len(valid_preds) >= (SMOOTHING_WINDOW // 2 + 1):
            smoothed_gesture = Counter(valid_preds).most_common(1)[0][0]

    current_time = time.time()

    # =========================
    # RASENGAN HOLD TRACKING
    # =========================

    if smoothed_gesture == "palm":

        if palm_hold_start is None:
            palm_hold_start = current_time

        hold_duration = current_time - palm_hold_start

    else:
        palm_hold_start = None
        hold_duration = 0.0

    # Faqat shu frame'da qo'l aniq ko'ringan bo'lsa chizamiz
    if (
        results.multi_hand_landmarks
        and predicted_gesture == "palm"
        and hold_duration >= RASENGAN_TRIGGER_TIME
    ):

        h, w = frame.shape[:2]

        palm_landmark = hand_landmarks.landmark[9]  # kaft markaziga yaqin nuqta
        palm_cx = int(palm_landmark.x * w)
        palm_cy = int(palm_landmark.y * h)

        box_size = x2 - x1

        progress = min(
            (hold_duration - RASENGAN_TRIGGER_TIME) / RASENGAN_GROW_TIME,
            1.0
        )

        draw_rasengan(
            frame,
            (palm_cx, palm_cy),
            box_size,
            current_time,
            progress
        )

    # =========================
    # SEQUENCE LOGIC
    # =========================

    if (
        smoothed_gesture is not None
        and smoothed_gesture in TARGET_SEQUENCE
    ):

        if smoothed_gesture != last_gesture:

            if (
                current_time - last_gesture_time
                > GESTURE_DELAY
            ):

                current_sequence.append(smoothed_gesture)

                gesture_history.append(smoothed_gesture)

                if len(gesture_history) > 3:
                    gesture_history.pop(0)

                last_gesture = smoothed_gesture
                last_gesture_time = current_time

                print("Gesture history:", gesture_history)

                if len(current_sequence) > len(TARGET_SEQUENCE):

                    current_sequence = (
                        current_sequence[
                            -len(TARGET_SEQUENCE):
                        ]
                    )

                # =====================
                # SUCCESS
                # =====================

                if current_sequence == TARGET_SEQUENCE:

                    if video_count < MAX_VIDEOS:

                        video_count += 1

                        print(
                            f"SHADOW CLONE! "
                            f"Videos: {video_count}"
                        )

                    current_sequence = []

                    last_gesture = None


    # =========================
    # CREATE LIVE CLONES
    # =========================

    display_frame = create_live_videos(
        frame,
        video_count
    )
    history_text = " -> ".join(
        gesture_history
    )

    cv2.putText(
        display_frame,
        history_text.upper(),
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3
    )

    # =========================
    # INFORMATION
    # =========================

    cv2.putText(
        display_frame,
        f"VIDEOS: {video_count}",
        (15, display_frame.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        "FIST -> PALM -> POINT",
        (15, display_frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # =========================
    # SHOW
    # =========================

    cv2.imshow(
        "Naruto Shadow Clone",
        display_frame
    )


    # ESC
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# =========================
# CLEANUP
# =========================

cap.release()
hands.close()
cv2.destroyAllWindows()