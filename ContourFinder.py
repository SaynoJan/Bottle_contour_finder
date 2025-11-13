import cv2
import numpy as np

class ObjectContourDetector:
    def __init__(self, background_path):
        self.background = cv2.imread(background_path)
        if self.background is None:
            raise ValueError("Не удалось загрузить фоновое изображение.")

    def preprocess(self, img):
        """Сглаживание и подавление шумов"""
        blurred = cv2.GaussianBlur(img, (5,5), 0)
        return blurred

    def detect_contour(self, img):
        """Нахождение контура объекта относительно фона"""
        bg_resized = cv2.resize(self.background, (img.shape[1], img.shape[0]))

        # --- 1. Сглаживание ---
        img_blur = self.preprocess(img)
        bg_blur = self.preprocess(bg_resized)

        # --- 2. Разница по цвету в LAB ---
        img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
        bg_lab = cv2.cvtColor(bg_blur, cv2.COLOR_BGR2LAB)

        diff = cv2.absdiff(img_lab, bg_lab)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # --- 3. Порог по разнице (чем меньше — тем чувствительнее к прозрачным объектам) ---
        _, mask = cv2.threshold(diff_gray, 10, 255, cv2.THRESH_BINARY)

        # --- 4. Морфология для объединения ---
        kernel = np.ones((9,9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # --- 5. Поиск контуров ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("Контуры не найдены.")
            return mask, None

        # --- 6. Выбираем самый большой контур (предположительно объект) ---
        largest = max(contours, key=cv2.contourArea)

        # --- 7. Рисуем контур на копии оригинала ---
        output = img.copy()
        cv2.drawContours(output, [largest], -1, (0, 255, 0), 3)

        return mask, output


# === Пример использования ===
if __name__ == "__main__":
    bg_path = "background.jpg"
    img_path = "images/20251105_165524_546600.jpg"

    detector = ObjectContourDetector(bg_path)
    img = cv2.imread(img_path)

    mask, output = detector.detect_contour(img)

    #cv2.imshow("Mask", mask)
    cv2.imshow("Contour", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
