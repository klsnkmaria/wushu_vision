import numpy as np
from pathlib import Path
import pickle


class KeypointNormalizer:
    """Нормалізація ключових точок відносно тазу та довжини торсу"""

    def __init__(self):
        # Індекси ключових точок MediaPipe
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12

    def normalize(self, keypoints):
        """
        Нормалізація keypoints для інваріантності до розміру та положення людини

        Вхід: (33, 3) або (33, 4) - keypoints з visibility
        Вихід: (33, 3) - нормалізовані координати
        """
        # Беремо тільки координати x, y, z
        kp = keypoints[:, :3].copy()

        # Обчислюємо центр тазу
        hip_center = (kp[self.LEFT_HIP] + kp[self.RIGHT_HIP]) / 2

        # Центруємо всі точки відносно тазу
        centered = kp - hip_center

        # Обчислюємо довжину торсу для масштабування
        shoulder_center = (kp[self.LEFT_SHOULDER] + kp[self.RIGHT_SHOULDER]) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center)

        # Масштабуємо відносно довжини торсу
        if torso_length > 1e-6:
            scaled = centered / torso_length
        else:
            scaled = centered

        return scaled


class FeatureExtractor:
    """Вилучення ознак з нормалізованих ключових точок"""

    def __init__(self):
        # Словник індексів ключових точок MediaPipe Pose
        self.indices = {
            'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
            'LEFT_HIP': 23, 'RIGHT_HIP': 24,
            'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
            'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
            'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32,
        }

    def calculate_angle(self, point_a, point_b, point_c):
        """
        Обчислення кута в точці B між векторами BA та BC

        Формула: angle = arccos((BA · BC) / (|BA| * |BC|))
        """
        vector_ba = point_a - point_b
        vector_bc = point_c - point_b

        # Скалярний добуток векторів
        dot_product = np.dot(vector_ba, vector_bc)

        # Добуток норм векторів
        norms = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)

        # Обчислення косинуса кута
        cosine_angle = dot_product / (norms + 1e-8)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        # Переведення в градуси
        angle_radians = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    def extract_features(self, keypoints):
        """
        Вилучення вектора ознак з ключових точок

        Повертає список числових ознак для класифікації
        """
        features = []

        # 1. Кути колін (найважливіші для стійок)
        left_knee_angle = self.calculate_angle(
            keypoints[self.indices['LEFT_HIP']],
            keypoints[self.indices['LEFT_KNEE']],
            keypoints[self.indices['LEFT_ANKLE']]
        )
        features.append(left_knee_angle)

        right_knee_angle = self.calculate_angle(
            keypoints[self.indices['RIGHT_HIP']],
            keypoints[self.indices['RIGHT_KNEE']],
            keypoints[self.indices['RIGHT_ANKLE']]
        )
        features.append(right_knee_angle)

        # 2. Кути в тазостегновому суглобі
        left_hip_angle = self.calculate_angle(
            keypoints[self.indices['LEFT_SHOULDER']],
            keypoints[self.indices['LEFT_HIP']],
            keypoints[self.indices['LEFT_KNEE']]
        )
        features.append(left_hip_angle)

        right_hip_angle = self.calculate_angle(
            keypoints[self.indices['RIGHT_SHOULDER']],
            keypoints[self.indices['RIGHT_HIP']],
            keypoints[self.indices['RIGHT_KNEE']]
        )
        features.append(right_hip_angle)

        # 3. Кут спини відносно вертикалі
        shoulder_mid = (
                               keypoints[self.indices['LEFT_SHOULDER']] +
                               keypoints[self.indices['RIGHT_SHOULDER']]
                       ) / 2
        hip_mid = (
                          keypoints[self.indices['LEFT_HIP']] +
                          keypoints[self.indices['RIGHT_HIP']]
                  ) / 2

        spine_vector = shoulder_mid - hip_mid
        vertical_vector = np.array([0, 1, 0])

        back_angle = self.calculate_angle(
            vertical_vector,
            np.zeros(3),
            spine_vector
        )
        features.append(back_angle)

        # 4. Геометричні характеристики стійки
        left_ankle = keypoints[self.indices['LEFT_ANKLE']]
        right_ankle = keypoints[self.indices['RIGHT_ANKLE']]

        # Ширина стійки (відстань між щиколотками)
        stance_width = np.linalg.norm(left_ankle - right_ankle)
        features.append(stance_width)

        # Висота тазу (Y-координата після нормалізації)
        hip_height = hip_mid[1]
        features.append(hip_height)

        # Глибина стійки (різниця по осі Z)
        stance_depth = abs(left_ankle[2] - right_ankle[2])
        features.append(stance_depth)

        # 5. Симетрія стійки
        knee_symmetry = abs(left_knee_angle - right_knee_angle)
        features.append(knee_symmetry)

        # 6. Ширина тазу (нормалізована)
        hip_width = np.linalg.norm(
            keypoints[self.indices['LEFT_HIP']] -
            keypoints[self.indices['RIGHT_HIP']]
        )
        features.append(hip_width)

        # 7. Кути ліктів (для контролю рук)
        left_elbow_angle = self.calculate_angle(
            keypoints[self.indices['LEFT_SHOULDER']],
            keypoints[self.indices['LEFT_ELBOW']],
            keypoints[self.indices['LEFT_HIP']]
        )
        features.append(left_elbow_angle)

        right_elbow_angle = self.calculate_angle(
            keypoints[self.indices['RIGHT_SHOULDER']],
            keypoints[self.indices['RIGHT_ELBOW']],
            keypoints[self.indices['RIGHT_HIP']]
        )
        features.append(right_elbow_angle)

        return features


class DatasetBuilder:
    """Побудова датасету для навчання моделі"""

    def __init__(self, data_dir='processed_data'):
        self.data_dir = Path(data_dir)
        self.normalizer = KeypointNormalizer()
        self.feature_extractor = FeatureExtractor()

        # Назви ознак для інтерпретації
        self.feature_names = [
            'left_knee_angle',
            'right_knee_angle',
            'left_hip_angle',
            'right_hip_angle',
            'back_angle',
            'stance_width',
            'hip_height',
            'stance_depth',
            'knee_symmetry',
            'hip_width',
            'left_elbow_angle',
            'right_elbow_angle'
        ]

    def build(self):
        """
        Створення датасету X, y для навчання

        Повертає:
            X: масив ознак (n_samples, n_features)
            y: масив міток класів (n_samples,)
            label_map: словник {назва_стійки: числова_мітка}
            metadata: додаткова інформація
        """
        X = []
        y = []
        metadata = {
            'filenames': [],
            'categories': [],
            'quality_labels': [],
            'feature_names': self.feature_names
        }

        # Знаходимо всі категорії в processed_data
        categories = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])

        # Визначаємо типи стійок (без суфіксів _good/_bad)
        stance_types = set()
        for cat in categories:
            # Очікуємо формат: "mabu_good", "gongbu_bad" тощо
            parts = cat.rsplit('_', 1)
            if len(parts) == 2:
                stance_type = parts[0]
                stance_types.add(stance_type)

        stance_types = sorted(stance_types)

        # Створюємо відображення стійка -> числова мітка
        label_map = {stance: idx for idx, stance in enumerate(stance_types)}

        print("Створення датасету...")
        print("\nВідображення міток:")
        for stance, label in label_map.items():
            print(f"  {label}: {stance}")
        print()

        # Обробка кожної категорії
        for category in categories:
            category_path = self.data_dir / category
            npy_files = list(category_path.glob('*.npy'))

            # Розбиваємо назву категорії на тип стійки та якість
            parts = category.rsplit('_', 1)
            if len(parts) != 2:
                print(f"УВАГА: Некоректна назва категорії {category}, пропускаємо")
                continue

            stance_type = parts[0]
            quality = parts[1]  # "good" або "bad"

            if stance_type not in label_map:
                print(f"УВАГА: Невідомий тип стійки {stance_type}, пропускаємо")
                continue

            label = label_map[stance_type]

            print(f"Обробка {category}: {len(npy_files)} файлів")

            # Обробка кожного файлу в категорії
            for npy_file in npy_files:
                # Завантаження ключових точок
                keypoints = np.load(npy_file)

                # Нормалізація
                normalized_kp = self.normalizer.normalize(keypoints)


                # Вилучення ознак
                features = self.feature_extractor.extract_features(normalized_kp)

                # Додавання якості як бінарної ознаки
                quality_feature = 1.0 if quality == 'good' else 0.0
                features.append(quality_feature)

                # Додавання до датасету
                X.append(features)
                y.append(label)

                # Збереження метаданих
                metadata['filenames'].append(str(npy_file))
                metadata['categories'].append(category)
                metadata['quality_labels'].append(quality)

        # Конвертація в numpy масиви
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Додавання назви ознаки якості
        metadata['feature_names'].append('quality')

        # Виведення інформації про датасет
        print(f"\nДатасет створено:")
        print(f"  Розмір X: {X.shape}")
        print(f"  Розмір y: {y.shape}")
        print(f"  Кількість класів: {len(label_map)}")
        print(f"  Кількість ознак: {X.shape[1]}")

        # Розподіл зразків по класах
        print(f"\nРозподіл по класах:")
        for stance, label in label_map.items():
            count_total = np.sum(y == label)
            count_good = np.sum([
                (y[i] == label and metadata['quality_labels'][i] == 'good')
                for i in range(len(y))
            ])
            count_bad = count_total - count_good
            print(f"  {stance}:")
            print(f"    Всього: {count_total}")
            print(f"    Правильні: {count_good}")
            print(f"    Неправильні: {count_bad}")

        return X, y, label_map, metadata

    def save_dataset(self, X, y, label_map, metadata,
                     output_file='processed_data/dataset.pkl'):
        """Збереження датасету на диск"""

        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)

        # Збереження у форматі pickle
        dataset = {
            'X': X,
            'y': y,
            'label_map': label_map,
            'metadata': metadata
        }

        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"\nДатасет збережено: {output_file}")

        # Додатково зберігаємо у форматі npz для швидкого завантаження
        npz_file = str(output_file).replace('.pkl', '.npz')
        np.savez(npz_file, X=X, y=y)
        print(f"Також збережено: {npz_file}")


if __name__ == "__main__":
    print("Побудова датасету для навчання\n")

    builder = DatasetBuilder(data_dir='../processed_data')

    # Створення датасету
    X, y, label_map, metadata = builder.build()

    # Збереження датасету
    builder.save_dataset(X, y, label_map, metadata,
                         output_file='../processed_data/dataset.pkl')

    print("\nПроцес завершено успішно")