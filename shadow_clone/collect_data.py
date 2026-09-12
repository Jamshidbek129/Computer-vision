import cv2
import os

# Qaysi gesture uchun dataset yig'amiz
gesture = input("Gesture nomini kiriting (fist/palm/point): ")

folder = f"dataset/{gesture}"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)

count = len(os.listdir(folder))

# Qo'l uchun ROI (Region of Interest) - kadrning belgilangan qismi
# Qo'lingizni shu kvadrat ichida ushlab turing
ROI_X1, ROI_Y1 = 200, 100
ROI_X2, ROI_Y2 = 500, 400

print()
print("Kamera ishga tushdi.")
print("Qo'lingizni ekrandagi yashil kvadrat ICHIDA ushlab turing.")
print("SPACE  -> rasm olish")
print("Q      -> chiqish")
print()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Kamera ochilmadi!")
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # MUHIM: saqlash uchun original (toza) frame'dan nusxa olamiz,
    # keyin faqat ROI qismini kesib olamiz.
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

    # Faqat KO'RSATISH uchun alohida nusxa - matn shu yerga chiziladi,
    # saqlanadigan rasmga (roi) hech qanday matn tushmaydi.
    display = frame.copy()

    cv2.rectangle(display, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 0), 2)

    cv2.putText(
        display,
        f"Gesture: {gesture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Images: {count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dataset Collection", display)

    key = cv2.waitKey(1) & 0xFF

    # SPACE
    if key == 32:

        filename = f"{folder}/{count}.jpg"

        # Faqat toza ROI qismi saqlanadi (matnsiz, butun fonsiz)
        cv2.imwrite(filename, roi)

        count += 1

        print(f"Saved: {filename}")

    # Q
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()