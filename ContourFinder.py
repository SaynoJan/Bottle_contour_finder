import cv2
import numpy as np
import os
import glob
from datetime import datetime

class ObjectContourDetector:
  
    VOLUME_STANDARDS = {
        0.33: {
            'min': 180000,  # Минимальное значение из таблицы 
            'max': 280000,  # Максимальное значение из таблицы 
            'avg': 235000,  # Приблизительное среднее
            'values': [254669, 276907, 225615, 258788, 219435, 214870, 229095, 256460, 181391]
        },
        0.5: {
            'min': 275000,  
            'max': 363000,  
            'avg': 320000,  
            'values': [318355, 353922, 327285, 355826, 302441, 323851, 300237, 311080, 
                      286983, 326567, 300538, 333279, 330302, 275502, 347980, 362178, 275351]
        },
        1.0: {
            'min': 640000, 
            'max': 680000, 
            'avg': 660000, 
            'values': [644879, 679867]
        }
    }

    def adaptive_threshold(self, diff_gray, img):
        mean_diff = np.mean(diff_gray)
        if mean_diff < 3:
            threshold = 5
        elif mean_diff < 30:
            threshold = 10
        else:
            threshold = 15
        return threshold, mean_diff

    def __init__(self, background_path):
        self.background = cv2.imread(background_path)
        if self.background is None:
            raise ValueError("Не удалось загрузить фоновое изображение.")
        
        self.base_dir = os.path.dirname(background_path)
        self.images_dir = os.path.join(self.base_dir, "images")
        self.results_dir = os.path.join(self.base_dir, "results")
        self.good_dir = os.path.join(self.base_dir, "good_results")
        self.bad_dir = os.path.join(self.base_dir, "bad_results")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.good_dir, exist_ok=True)
        os.makedirs(self.bad_dir, exist_ok=True)

    def check_volume_standard(self, pixel_count):
        """Проверяет соответствует ли количество пикселей стандартам объема"""
        results = {}
        
        for volume, data in self.VOLUME_STANDARDS.items():
            if data['min'] <= pixel_count <= data['max']:
                # Вычисляем отклонение от среднего в процентах
                deviation_percent = abs(pixel_count - data['avg']) / data['avg'] * 100
                
                # Вычисляем стандартное отклонение для этого объема
                if len(data['values']) > 1:
                    std_dev = np.std(data['values'])
                    z_score = abs(pixel_count - np.mean(data['values'])) / std_dev
                else:
                    std_dev = 0
                    z_score = 0
                
                results[volume] = {
                    'within_range': True,
                    'deviation_percent': deviation_percent,
                    'z_score': z_score,
                    'avg': data['avg'],
                    'min': data['min'],
                    'max': data['max']
                }
            else:
                results[volume] = {
                    'within_range': False,
                    'deviation_percent': None,
                    'z_score': None
                }
        
        return results

    def determine_best_match(self, pixel_count, volume_results):
        """Определяет наиболее подходящий объем"""
        best_match = None
        best_score = float('inf')
        
        for volume, result in volume_results.items():
            if result['within_range']:
                # Чем меньше отклонение и z-score, тем лучше
                score = result['deviation_percent'] + result['z_score'] * 10
                if score < best_score:
                    best_score = score
                    best_match = volume
        
        return best_match

    def preprocess(self, img):
        blurred = cv2.GaussianBlur(img, (5,5), 0)
        return blurred

    def detect_contour(self, img, filename=""):
        if img is None:
            return None, 0, 0, None
        
        bg_resized = cv2.resize(self.background, (img.shape[1], img.shape[0]))

        img_blur = self.preprocess(img)
        bg_blur = self.preprocess(bg_resized)

        img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
        bg_lab = cv2.cvtColor(bg_blur, cv2.COLOR_BGR2LAB)

        diff = cv2.absdiff(img_lab, bg_lab)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        threshold, mean_diff = self.adaptive_threshold(diff_gray, img)
        _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((9,9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0, mean_diff, None

        largest = max(contours, key=cv2.contourArea)
        
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [largest], -1, 255, thickness=cv2.FILLED)
        pixel_count = np.sum(contour_mask == 255)
        
        # Проверяем соответствие стандартам
        volume_results = self.check_volume_standard(pixel_count)
        best_volume = self.determine_best_match(pixel_count, volume_results)
        
        # Рисуем контур на оригинальном изображении
        output = img.copy()
        cv2.drawContours(output, [largest], -1, (0, 255, 0), 3)
        
        # Определяем цвет текста в зависимости от качества
        if best_volume:
            # Хороший контур - зеленый
            text_color = (0, 255, 0)
            status = "GOOD"
        else:
            # Плохой контур - красный
            text_color = (0, 0, 255)
            status = "BAD"
        
        # Добавляем текст с информацией
        font = cv2.FONT_HERSHEY_SIMPLEX
        text1 = f"Pixels: {pixel_count:,}"
        text2 = f"Mean diff: {mean_diff:.2f}"
        text3 = f"Status: {status}"
        
        if best_volume:
            text4 = f"Volume: {best_volume}L"
            volume_data = volume_results[best_volume]
            text5 = f"Deviation: {volume_data['deviation_percent']:.1f}%"
        else:
            text4 = "Volume: UNKNOWN"
            text5 = "Out of range"
        
        # Вычисляем положение текста
        text_x = 20
        text_y = 40
        line_height = 40
        
        # Рисуем полупрозрачный фон для текста
        overlay = output.copy()
        cv2.rectangle(overlay, (text_x - 10, text_y - 30),
                     (text_x + 400, text_y + line_height * 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # Рисуем текст
        texts = [text1, text2, text3, text4, text5]
        for i, text in enumerate(texts):
            y_pos = text_y + line_height * i
            # Тень текста
            cv2.putText(output, text, (text_x + 2, y_pos + 2), 
                       font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            # Основной текст
            cv2.putText(output, text, (text_x, y_pos), 
                       font, 0.8, text_color, 2, cv2.LINE_AA)

        return output, pixel_count, mean_diff, best_volume

    def process_all_images(self):
        """Обрабатывает все изображения в папке images"""
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        if not image_paths:
            print(f"В папке {self.images_dir} не найдено изображений!")
            return []
        
        print(f"Найдено {len(image_paths)} изображений")
        print("-" * 60)
        
        results = []
        good_count = 0
        bad_count = 0
        
        for i, img_path in enumerate(image_paths, 1):
            filename = os.path.basename(img_path)
            print(f"[{i}/{len(image_paths)}] Обработка: {filename}")
            
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"  ✗ Ошибка загрузки")
                continue
            
            result_img, pixel_count, mean_diff, volume = self.detect_contour(img, filename)
            
            if result_img is not None:
                # Определяем папку для сохранения
                if volume:
                    save_dir = self.good_dir
                    good_count += 1
                    status = "✓ GOOD"
                else:
                    save_dir = self.bad_dir
                    bad_count += 1
                    status = "✗ BAD"
                
                # Сохраняем результат
                base_name = os.path.splitext(filename)[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if volume:
                    save_name = f"{base_name}_{volume}L_px{pixel_count}_md{mean_diff:.1f}.jpg"
                else:
                    save_name = f"{base_name}_UNKNOWN_px{pixel_count}_md{mean_diff:.1f}.jpg"
                
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, result_img)
                results.append(save_path)
                
                print(f"  {status}")
                print(f"  Пикселей: {pixel_count:,}, Mean diff: {mean_diff:.2f}")
                if volume:
                    print(f"  Определен объем: {volume} л")
                else:
                    print(f"  Объем: не определен (вне диапазонов)")
                print(f"  Сохранено в: {save_dir}")
        
        print("-" * 60)
        print("СТАТИСТИКА ОБРАБОТКИ:")
        print(f"Всего изображений: {len(image_paths)}")
        print(f"Хороших контуров: {good_count} ✓")
        print(f"Плохих контуров: {bad_count} ✗")
        print(f"Процент успеха: {good_count/max(len(image_paths),1)*100:.1f}%")
        print("-" * 60)
        
        if good_count > 0:
            print(f"Хорошие результаты в папке: {self.good_dir}")
        if bad_count > 0:
            print(f"Плохие результаты в папке: {self.bad_dir}")
        
        return results

    def analyze_existing_results(self, results_dir=None):
        """Анализирует уже обработанные результаты"""
        if results_dir is None:
            results_dir = self.results_dir
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(results_dir, ext)))
        
        if not image_paths:
            print(f"В папке {results_dir} не найдено обработанных изображений!")
            return
        
        print(f"\nАнализ {len(image_paths)} обработанных изображений:")
        print("-" * 60)
        
        volumes_summary = {}
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            
            # Пытаемся извлечь информацию из имени файла
            if 'px' in filename and 'md' in filename:
                try:
                    # Извлекаем количество пикселей из имени файла
                    px_part = filename.split('px')[1].split('_')[0]
                    pixel_count = int(px_part)
                    
                    # Проверяем соответствие стандартам
                    volume_results = self.check_volume_standard(pixel_count)
                    best_volume = self.determine_best_match(pixel_count, volume_results)
                    
                    if best_volume:
                        if best_volume not in volumes_summary:
                            volumes_summary[best_volume] = []
                        volumes_summary[best_volume].append(pixel_count)
                        
                        print(f"{filename[:30]:30} - {pixel_count:8,} px -> {best_volume} л ✓")
                    else:
                        print(f"{filename[:30]:30} - {pixel_count:8,} px -> Вне диапазона ✗")
                        
                except (ValueError, IndexError):
                    print(f"{filename[:30]:30} - Не удалось проанализировать")
        
        # Выводим статистику по объемам
        if volumes_summary:
            print("\n" + "="*60)
            print("СВОДКА ПО ОБЪЕМАМ:")
            print("="*60)
            
            for volume, pixel_counts in sorted(volumes_summary.items()):
                avg_px = np.mean(pixel_counts)
                std_px = np.std(pixel_counts)
                std_percent = (std_px / avg_px * 100) if avg_px > 0 else 0
                
                standard = self.VOLUME_STANDARDS[volume]
                standard_avg = np.mean(standard['values'])
                standard_std = np.std(standard['values'])
                
                print(f"\nОбъем: {volume} л")
                print(f"  Количество образцов: {len(pixel_counts)}")
                print(f"  Среднее пикселей: {avg_px:,.0f}")
                print(f"  Стандартное отклонение: {std_px:,.0f} ({std_percent:.1f}%)")
                print(f"  Стандарт из таблицы: {standard_avg:,.0f} ± {standard_std:,.0f}")
                print(f"  Диапазон в таблице: {standard['min']:,} - {standard['max']:,}")


def main():
    base_dir = "C:\\Users\\User\\Documents\\Bottle_contour_finder"
    bg_path = os.path.join(base_dir, "background.jpg")
    
    if not os.path.exists(bg_path):
        print(f"Ошибка: Фоновое изображение не найдено!")
        return
    
    detector = ObjectContourDetector(bg_path)
    
    print("=" * 60)
    print("АВТОМАТИЧЕСКИЙ ДЕТЕКТОР КОНТУРОВ БУТЫЛОК")
    print("С ПРОВЕРКОЙ СООТВЕТСТВИЯ СТАНДАРТАМ ОБЪЕМА")
    print("=" * 60)
    
    # Выводим стандарты
    print("\nИСПОЛЬЗУЕМЫЕ СТАНДАРТЫ:")
    print("-" * 40)
    for volume, data in detector.VOLUME_STANDARDS.items():
        print(f"{volume} л: {data['min']:,} - {data['max']:,} пикселей")
        print(f"  (на основе {len(data['values'])} образцов)")
    
    print("\n" + "=" * 60)
    print("НАЧАЛО ОБРАБОТКИ ИЗОБРАЖЕНИЙ")
    print("=" * 60)
    
    results = detector.process_all_images()
    
    # Анализируем результаты если есть
    if os.path.exists(detector.results_dir):
        detector.analyze_existing_results(detector.results_dir)


if __name__ == "__main__":
    main()
