import pickle
import numpy as np
from pathlib import Path


def load_and_inspect_dataset(dataset_path='processed_data/dataset.pkl'):
    """Завантаження та швидка перевірка датасету"""

    print("Завантаження датасету...")

    if not Path(dataset_path).exists():
        print(f"ПОМИЛКА: Файл {dataset_path} не знайдено")
        return

    # Завантаження
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']
    label_map = data['label_map']
    metadata = data['metadata']

    # Інверсне відображення для декодування міток
    reverse_label_map = {v: k for k, v in label_map.items()}

    # Виведення основної інформації
    print("\n" + "=" * 60)
    print("ІНФОРМАЦІЯ ПРО ДАТАСЕТ")
    print("=" * 60)

    print(f"\nРозмірність даних:")
    print(f"  Кількість зразків: {len(X)}")
    print(f"  Кількість ознак: {X.shape[1]}")

    print(f"\nНазви ознак ({len(metadata['feature_names'])} всього):")
    for i, name in enumerate(metadata['feature_names']):
        print(f"  [{i:2d}] {name}")

    print(f"\nКласи стійок:")
    for stance, label in sorted(label_map.items(), key=lambda x: x[1]):
        count = np.sum(y == label)
        percentage = (count / len(y)) * 100
        print(f"  {label} - {stance}: {count} зразків ({percentage:.1f}%)")

    # Статистика якості виконання
    quality_feature_idx = metadata['feature_names'].index('quality')
    good_count = np.sum(X[:, quality_feature_idx] == 1.0)
    bad_count = np.sum(X[:, quality_feature_idx] == 0.0)

    print(f"\nРозподіл за якістю виконання:")
    print(f"  Правильні стійки: {good_count}")
    print(f"  Неправильні стійки: {bad_count}")
    print(f"  Співвідношення good/bad: {good_count / bad_count:.2f}" if bad_count > 0 else "")

    # Приклад даних
    print(f"\n" + "=" * 60)
    print("ПРИКЛАД ДАНИХ (перший зразок)")
    print("=" * 60)

    print(f"\nОзнаки:")
    for i, (name, value) in enumerate(zip(metadata['feature_names'], X[0])):
        print(f"  {name:20s} = {value:8.3f}")

    print(f"\nМітка класу: {y[0]} ({reverse_label_map[y[0]]})")
    print(f"Якість: {'Правильна' if X[0, quality_feature_idx] == 1.0 else 'Неправильна'}")
    print(f"Файл: {metadata['filenames'][0]}")

    # Статистика ознак
    print(f"\n" + "=" * 60)
    print("СТАТИСТИКА ОЗНАК")
    print("=" * 60)

    print(f"\n{'Ознака':<20} {'Мін':>10} {'Макс':>10} {'Середнє':>10} {'Std':>10}")
    print("-" * 60)

    for i, name in enumerate(metadata['feature_names']):
        min_val = X[:, i].min()
        max_val = X[:, i].max()
        mean_val = X[:, i].mean()
        std_val = X[:, i].std()

        print(f"{name:<20} {min_val:10.3f} {max_val:10.3f} {mean_val:10.3f} {std_val:10.3f}")

    # Перевірка на аномалії
    print(f"\n" + "=" * 60)
    print("ПЕРЕВІРКА НА АНОМАЛІЇ")
    print("=" * 60)

    anomalies_found = False

    # Перевірка наявності NaN або Inf
    if np.isnan(X).any():
        print("  УВАГА: Знайдено NaN значення в даних")
        anomalies_found = True

    if np.isinf(X).any():
        print("  УВАГА: Знайдено Inf значення в даних")
        anomalies_found = True

    # Перевірка діапазонів кутів (0-180 градусів)
    angle_features = [i for i, name in enumerate(metadata['feature_names']) if 'angle' in name]
    for idx in angle_features:
        if X[:, idx].min() < 0 or X[:, idx].max() > 180:
            print(f"  УВАГА: Кут {metadata['feature_names'][idx]} виходить за межі [0, 180]")
            anomalies_found = True

    if not anomalies_found:
        print("  Аномалій не виявлено")

    print(f"\n" + "=" * 60)
    print("Датасет готовий до навчання моделі")
    print("=" * 60)

    return data


if __name__ == "__main__":
    dataset = load_and_inspect_dataset('../processed_data/dataset.pkl')

    print("\nДля тренування моделі запустіть наступний скрипт")