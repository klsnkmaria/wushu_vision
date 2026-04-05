import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import json


class ImageKeypointExtractor:
    def __init__(self):
        # Ініціалізація інструментів MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Налаштування моделі для обробки окремих фотографій
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # Обов'язково для роботи з окремими кадрами
            model_complexity=2,  # Використання найбільш точної моделі (Heavy)
            min_detection_confidence=0.5
        )

    def extract_from_image(self, image_path):
        """
        Вилучення ключових точок з одного зображення.
        Повертає масив точок та зображення з візуалізацією скелета.
        """
        # Завантаження зображення
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Помилка: не вдалося відкрити файл {image_path}")
            return None, None

        # MediaPipe працює з RGB, тоді як OpenCV завантажує в BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            print(f"Попередження: об'єкт не знайдений на фото {image_path}")
            return None, None

        # Збір координат усіх 33 точок (x, y, z та видимість)
        landmarks = results.pose_landmarks.landmark
        keypoints = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in landmarks
        ])

        # Створення візуалізації (малювання скелета поверх копії фото)
        image_with_pose = image.copy()
        self.mp_drawing.draw_landmarks(
            image_with_pose,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)
        )

        return keypoints, image_with_pose

    def process_all_images(self, input_dir='raw_images', output_dir='processed_data'):
        """
        Обхід усіх підпапок (категорій) та обробка кожного зображення всередині.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Словник для збору статистики обробки
        stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'categories': {}
        }

        # Перегляд папок категорій (gongbu_good, mabu_bad тощо)
        for category_folder in input_path.iterdir():
            if not category_folder.is_dir():
                continue

            category_name = category_folder.name
            print(f"\n--- Обробка категорії: {category_name} ---")

            # Створення структур для результатів
            output_category = output_path / category_name
            output_category.mkdir(exist_ok=True)

            vis_folder = Path('visualizations') / category_name
            vis_folder.mkdir(parents=True, exist_ok=True)

            category_stats = {'total': 0, 'successful': 0, 'failed': 0}

            # Пошук файлів поширених форматів
            image_files = []
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                image_files.extend(list(category_folder.glob(ext)))

            for img_path in image_files:
                stats['total_images'] += 1
                category_stats['total'] += 1

                print(f"Файл: {img_path.name} - ", end='')

                keypoints, image_with_pose = self.extract_from_image(img_path)

                if keypoints is None:
                    stats['failed'] += 1
                    category_stats['failed'] += 1
                    print("Помилка")
                    continue

                # Збереження числових даних у форматі .npy
                output_file = output_category / f"{img_path.stem}.npy"
                np.save(output_file, keypoints)

                # Збереження результату візуалізації
                vis_file = vis_folder / f"{img_path.stem}_pose.jpg"
                cv2.imwrite(str(vis_file), image_with_pose)

                stats['successful'] += 1
                category_stats['successful'] += 1
                print("Успішно")

            stats['categories'][category_name] = category_stats

        # Вивід підсумкової інформації
        self.print_summary(stats, output_dir)

        # Збереження статистики у JSON файл
        with open('processing_report.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return stats

    def print_summary(self, stats, output_dir):
        print("\n" + "=" * 40)
        print("ЗАГАЛЬНИЙ ЗВІТ ОБРОБКИ")
        print("=" * 40)
        print(f"Оброблено всього фото: {stats['total_images']}")
        print(f"Успішно вилучено точок: {stats['successful']}")
        print(f"Не вдалося обробити: {stats['failed']}")
        print("-" * 40)
        print(f"Дані збережені у папці: {output_dir}")
        print("Візуалізації скелетів збережені у папці: visualizations")
        print("=" * 40)


# Точка входу в програму
if __name__ == "__main__":
    print("Запуск програми для вилучення ключових точок тіла...")

    # Ініціалізація та запуск процесу
    # Вкажіть правильні шляхи до ваших папок
    extractor = ImageKeypointExtractor()
    extractor.process_all_images(
        input_dir='../raw_images',
        output_dir='../processed_data'
    )

    print("\nПроцес завершено. Перегляньте результати для перевірки точності моделі.")
