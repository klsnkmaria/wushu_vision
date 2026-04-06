import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class StanceClassifier:
    """Класифікатор стійок ушу на основі Random Forest"""

    def __init__(self, n_estimators=500, max_depth=20, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1 
        )
        self.label_map = None
        self.feature_names = None
        self.reverse_label_map = None

    def train(self, X, y, label_map, feature_names, test_size=0.2):
        """
        Навчання моделі з оцінкою якості

        Параметри:
            X: масив ознак
            y: масив міток
            label_map: словник стійка->мітка
            feature_names: назви ознак
            test_size: частка тестової вибірки
        """
        self.label_map = label_map
        self.feature_names = feature_names
        self.reverse_label_map = {v: k for k, v in label_map.items()}

        print("Початок навчання моделі...")
        print(f"  Розмір датасету: {X.shape}")
        print(f"  Кількість класів: {len(label_map)}")
        print(f"  Розбиття train/test: {int((1 - test_size) * 100)}/{int(test_size * 100)}")

        # Розбиття на train/test з стратифікацією
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        print(f"\nРозмір навчальної вибірки: {X_train.shape[0]}")
        print(f"Розмір тестової вибірки: {X_test.shape[0]}")

        # Навчання
        print("\nНавчання Random Forest...")
        self.model.fit(X_train, y_train)
        print("Навчання завершено")

        # Оцінка на тренувальній вибірці
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"\nТочність на навчальній вибірці: {train_accuracy:.4f}")

        # Оцінка на тестовій вибірці
        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Точність на тестовій вибірці: {test_accuracy:.4f}")

        # Крос-валідація
        print("\nКрос-валідація (5 фолдів)...")
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=5,
            scoring='accuracy'
        )
        print(f"Середня точність CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Детальний звіт класифікації
        print("\n" + "=" * 70)
        print("ЗВІТ КЛАСИФІКАЦІЇ (тестова вибірка)")
        print("=" * 70)

        target_names = [self.reverse_label_map[i] for i in range(len(label_map))]
        print(classification_report(
            y_test, y_test_pred,
            target_names=target_names,
            digits=4
        ))

        # Матриця помилок
        print("\n" + "=" * 70)
        print("МАТРИЦЯ ПОМИЛОК")
        print("=" * 70)
        cm = confusion_matrix(y_test, y_test_pred)
        print("\n" + self._format_confusion_matrix(cm, target_names))

        # Важливість ознак
        self._print_feature_importance()

        return X_train, X_test, y_train, y_test, y_test_pred

    def _format_confusion_matrix(self, cm, labels):
        """Форматування матриці помилок для виводу"""
        max_label_len = max(len(label) for label in labels)

        # Заголовок
        header = " " * (max_label_len + 2) + "  ".join(f"{label:>8s}" for label in labels)
        lines = [header]
        lines.append("-" * len(header))

        # Рядки матриці
        for i, label in enumerate(labels):
            row = f"{label:<{max_label_len}}  " + "  ".join(f"{cm[i, j]:8d}" for j in range(len(labels)))
            lines.append(row)

        return "\n".join(lines)

    def _print_feature_importance(self):
        """Виведення важливості ознак"""
        print("\n" + "=" * 70)
        print("ВАЖЛИВІСТЬ ОЗНАК")
        print("=" * 70)

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"\n{'Ранг':<6} {'Ознака':<25} {'Важливість':<12} {'Гістограма'}")
        print("-" * 70)

        for rank, idx in enumerate(indices, 1):
            importance = importances[idx]
            name = self.feature_names[idx]
            bar = "#" * int(importance * 50)
            print(f"{rank:<6} {name:<25} {importance:<12.4f} {bar}")

    def predict(self, features):
        """Передбачення класу для одного зразка"""
        if isinstance(features, list):
            features = np.array(features)

        features = features.reshape(1, -1)
        prediction = self.model.predict(features)[0]

        return prediction

    def predict_proba(self, features):
        """Передбачення ймовірностей для всіх класів"""
        if isinstance(features, list):
            features = np.array(features)

        features = features.reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]

        return probabilities

    def save(self, output_dir='models'):
        """Збереження натренованої моделі"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        model_file = output_path / 'stance_classifier.pkl'

        model_data = {
            'model': self.model,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'feature_names': self.feature_names
        }

        joblib.dump(model_data, model_file)
        print(f"\nМодель збережено: {model_file}")

        return model_file

    def load(self, model_path):
        """Завантаження збереженої моделі"""
        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.label_map = model_data['label_map']
        self.reverse_label_map = model_data['reverse_label_map']
        self.feature_names = model_data['feature_names']

        print(f"Модель завантажено з: {model_path}")


class ModelVisualizer:
    """Візуалізація результатів навчання"""

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels, output_file='visualizations/confusion_matrix.png'):
        """Створення графіку матриці помилок"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Кількість зразків'}
        )
        plt.title('Матриця помилок', fontsize=14, pad=20)
        plt.ylabel('Справжній клас', fontsize=12)
        plt.xlabel('Передбачений клас', fontsize=12)
        plt.tight_layout()

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Матриця помилок збережена: {output_file}")
        plt.close()

    @staticmethod
    def plot_feature_importance(importances, feature_names, top_n=15,
                                output_file='visualizations/feature_importance.png'):
        """Графік важливості ознак"""
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Важливість', fontsize=12)
        plt.title(f'Топ-{top_n} найважливіших ознак', fontsize=14, pad=20)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Графік важливості ознак збережено: {output_file}")
        plt.close()

    @staticmethod
    def plot_class_distribution(y, label_map,
                                output_file='visualizations/class_distribution.png'):
        """Графік розподілу класів"""
        reverse_map = {v: k for k, v in label_map.items()}
        labels = [reverse_map[i] for i in range(len(label_map))]
        counts = [np.sum(y == i) for i in range(len(label_map))]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, counts, color='steelblue', alpha=0.7)
        plt.xlabel('Тип стійки', fontsize=12)
        plt.ylabel('Кількість зразків', fontsize=12)
        plt.title('Розподіл зразків по класах', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')

        # Додавання значень на стовпчики
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(count)}',
                     ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Графік розподілу класів збережено: {output_file}")
        plt.close()


def main():
    """Основна функція для навчання моделі"""

    print("=" * 70)
    print("НАВЧАННЯ МОДЕЛІ КЛАСИФІКАЦІЇ СТІЙОК УШУ")
    print("=" * 70)

    # Завантаження датасету
    dataset_path = '../processed_data/dataset.pkl'

    print(f"\nЗавантаження датасету з {dataset_path}...")

    if not Path(dataset_path).exists():
        print(f"ПОМИЛКА: Файл {dataset_path} не знайдено")
        print("Спочатку запустіть скрипт 03_build_dataset.py")
        return

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']
    label_map = data['label_map']
    metadata = data['metadata']
    feature_names = metadata['feature_names']

    print(f"Датасет завантажено: {X.shape[0]} зразків, {X.shape[1]} ознак")

    # Створення візуалізації розподілу класів
    visualizer = ModelVisualizer()
    visualizer.plot_class_distribution(y, label_map)

    # Створення та навчання моделі
    classifier = StanceClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42
    )

    X_train, X_test, y_train, y_test, y_pred = classifier.train(
        X, y, label_map, feature_names, test_size=0.2
    )

    # Створення візуалізацій
    print("\n" + "=" * 70)
    print("СТВОРЕННЯ ВІЗУАЛІЗАЦІЙ")
    print("=" * 70)
    print()

    target_names = [classifier.reverse_label_map[i] for i in range(len(label_map))]

    visualizer.plot_confusion_matrix(
        y_test, y_pred, target_names,
        output_file='../visualizations/confusion_matrix.png'
    )

    visualizer.plot_feature_importance(
        classifier.model.feature_importances_,
        feature_names,
        top_n=min(15, len(feature_names)),
        output_file='../visualizations/feature_importance.png'
    )

    # Збереження моделі
    print("\n" + "=" * 70)
    print("ЗБЕРЕЖЕННЯ МОДЕЛІ")
    print("=" * 70)

    model_path = classifier.save(output_dir='../models')

    # Тестування завантаження моделі
    print("\nПеревірка завантаження моделі...")
    test_classifier = StanceClassifier()
    test_classifier.load(model_path)

    # Тест на одному зразку
    print("\nТест передбачення на випадковому зразку:")
    test_idx = np.random.randint(0, len(X_test))
    test_features = X_test[test_idx]
    test_true = y_test[test_idx]

    prediction = test_classifier.predict(test_features)
    probabilities = test_classifier.predict_proba(test_features)

    print(f"  Справжній клас: {classifier.reverse_label_map[test_true]}")
    print(f"  Передбачений клас: {classifier.reverse_label_map[prediction]}")
    print(f"  Ймовірності:")
    for label, prob in zip(target_names, probabilities):
        print(f"    {label}: {prob:.4f}")

    # Підсумок
    print(f"\nМодель успішно навчено та збережено")
    print(f"Точність на тестовій вибірці: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"\nФайли:")
    print(f"  Модель: {model_path}")
    print(f"  Візуалізації: ../visualizations/")


if __name__ == "__main__":
    main()
