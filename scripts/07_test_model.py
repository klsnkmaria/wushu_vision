import numpy as np
import pickle
from pathlib import Path
import joblib


def test_saved_model():
    """Тестування збереженої моделі на всіх даних"""

    print("Тестування збереженої моделі\n")

    model_path = '../models/stance_classifier.pkl'

    if not Path(model_path).exists():
        print(f"ПОМИЛКА: Модель не знайдена за шляхом {model_path}")
        return

    print(f"Завантаження моделі з {model_path}...")
    model_data = joblib.load(model_path)

    model = model_data['model']
    label_map = model_data['label_map']
    reverse_label_map = model_data['reverse_label_map']
    feature_names = model_data['feature_names']

    print(f"Модель завантажено успішно")
    print(f"Кількість класів: {len(label_map)}")
    print(f"Кількість ознак: {len(feature_names)}\n")

    dataset_path = '../processed_data/dataset.pkl'

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']
    metadata = data['metadata']

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    errors = []
    for i in range(len(X)):
        if predictions[i] != y[i]:
            errors.append({
                'index': i,
                'true_label': reverse_label_map[y[i]],
                'pred_label': reverse_label_map[predictions[i]],
                'confidence': probabilities[i][predictions[i]],
                'filename': metadata['filenames'][i],
                'quality': metadata['quality_labels'][i]
            })


    print(f"\n" + "=" * 70)
    print("РЕЗУЛЬТАТИ ТЕСТУВАННЯ")
    print("=" * 70)

    accuracy = np.mean(predictions == y)
    print(f"\nЗагальна точність: {accuracy:.4f}")
    print(f"Кількість помилок: {len(errors)} з {len(X)} ({len(errors) / len(X) * 100:.2f}%)")

    # Аналіз помилок по класах
    print(f"\n" + "=" * 70)
    print("АНАЛІЗ ПОМИЛОК ПО КЛАСАХ")
    print("=" * 70)

    for stance_name, stance_label in sorted(label_map.items(), key=lambda x: x[1]):
        indices = np.where(y == stance_label)[0]
        correct = np.sum(predictions[indices] == y[indices])
        total = len(indices)

        print(f"\n{stance_name}:")
        print(f"  Всього зразків: {total}")
        print(f"  Правильно: {correct}")
        print(f"  Точність: {correct / total:.4f}")

    # Топ-10 найгірших передбачень
    if len(errors) > 0:
        print(f"\n" + "=" * 70)
        print("ТОП-10 НАЙГІРШИХ ПЕРЕДБАЧЕНЬ")
        print("=" * 70)

        sorted_errors = sorted(errors, key=lambda x: x['confidence'], reverse=True)[:10]

        for i, error in enumerate(sorted_errors, 1):
            print(f"\n{i}. Файл: {Path(error['filename']).name}")
            print(f"   Справжній клас: {error['true_label']}")
            print(f"   Передбачено: {error['pred_label']} (впевненість: {error['confidence']:.4f})")
            print(f"   Якість виконання: {error['quality']}")


    print(f"\n" + "=" * 70)
    print("ПРИКЛАДИ ПЕРЕДБАЧЕНЬ")
    print("=" * 70)

    for stance_name, stance_label in list(label_map.items())[:3]:
        indices = np.where(y == stance_label)[0]
        if len(indices) > 0:
            idx = indices[0]
            pred = predictions[idx]
            probs = probabilities[idx]

            print(f"\nПриклад: {stance_name}")
            print(f"  Файл: {Path(metadata['filenames'][idx]).name}")
            print(f"  Передбачено: {reverse_label_map[pred]}")
            print(f"  Ймовірності:")
            for label_name, label_id in sorted(label_map.items(), key=lambda x: x[1]):
                print(f"    {label_name}: {probs[label_id]:.4f}")


if __name__ == "__main__":
    test_saved_model()