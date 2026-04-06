import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class DataQualityChecker:
    """Перевірка якості вилучених ключових точок"""

    def __init__(self, data_dir='processed_data'):
        self.data_dir = Path(data_dir)

    def check_all(self):
        """Аналіз усіх категорій даних"""
        print("Перевірка якості даних...\n")

        all_stats = {}
        total_files = 0
        problematic_files = []

        for category_folder in self.data_dir.iterdir():
            if not category_folder.is_dir():
                continue

            category_name = category_folder.name
            npy_files = list(category_folder.glob('*.npy'))

            print(f"Категорія: {category_name}")
            print(f"  Кількість файлів: {len(npy_files)}")

            if len(npy_files) == 0:
                print(f"Порожня папка\n")
                continue

            # Аналіз структури даних
            shapes = []
            visibilities = []
            low_visibility_files = []

            for npy_file in npy_files:
                kp = np.load(npy_file)
                shapes.append(kp.shape)

                # Перевірка середньої видимості точок
                if kp.shape[1] == 4:
                    avg_vis = np.mean(kp[:, 3])
                    visibilities.append(avg_vis)

                    if avg_vis < 0.5:
                        low_visibility_files.append((npy_file.name, avg_vis))

            total_files += len(npy_files)

            # Перевірка консистентності розмірів
            unique_shapes = set(shapes)
            if len(unique_shapes) > 1:
                print(f" Знайдено різні розміри: {unique_shapes}")
                problematic_files.append(category_name)
            else:
                print(f"  Розмір даних: {shapes[0]}")

            # Статистика видимості
            if visibilities:
                avg_visibility = np.mean(visibilities)
                min_visibility = np.min(visibilities)
                max_visibility = np.max(visibilities)

                print(f"  Видимість точок:")
                print(f"    Середня: {avg_visibility:.2%}")
                print(f"    Мінімальна: {min_visibility:.2%}")
                print(f"    Максимальна: {max_visibility:.2%}")

                if avg_visibility < 0.5:
                    print(f" Низька середня видимість!")

                if low_visibility_files:
                    print(f"  Файли з низькою видимістю ({len(low_visibility_files)}):")
                    for fname, vis in low_visibility_files[:5]:
                        print(f"    {fname}: {vis:.2%}")

            all_stats[category_name] = {
                'count': len(npy_files),
                'shapes': [str(s) for s in unique_shapes],
                'avg_visibility': np.mean(visibilities) if visibilities else None,
                'low_visibility_count': len(low_visibility_files)
            }

            print()

        # Загальна статистика
        print("=" * 50)
        print("ЗАГАЛЬНА СТАТИСТИКА")
        print("=" * 50)
        print(f"Всього файлів оброблено: {total_files}")
        print(f"Кількість категорій: {len(all_stats)}")

        if problematic_files:
            print(f"\nКатегорії з проблемами: {', '.join(problematic_files)}")
        else:
            print("\nВсі дані мають консистентну структуру")

        return all_stats

    def visualize_samples(self, categories=None, n_samples=5):
        """Візуалізація зразків ключових точок"""

        if categories is None:
            categories = [d.name for d in self.data_dir.iterdir() if d.is_dir()]

        print("\nСтворення візуалізацій...")

        for category in categories:
            category_path = self.data_dir / category

            if not category_path.exists():
                print(f"  Категорія {category} не знайдена")
                continue

            npy_files = list(category_path.glob('*.npy'))[:n_samples]

            if len(npy_files) == 0:
                print(f"  Категорія {category} порожня")
                continue

            fig, axes = plt.subplots(1, len(npy_files), figsize=(15, 3))
            if len(npy_files) == 1:
                axes = [axes]

            for ax, npy_file in zip(axes, npy_files):
                kp = np.load(npy_file)

                # Відображення точок на 2D площині (x, y)
                ax.scatter(kp[:, 0], kp[:, 1], c='red', s=30, alpha=0.6)
                ax.set_title(npy_file.stem, fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(1, 0)  # Інверсія осі Y
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

            plt.suptitle(f'Зразки keypoints: {category}', fontsize=12)
            plt.tight_layout()

            output_file = f'visualizations/analysis_sample_{category}.png'
            plt.savefig(output_file, dpi=100)
            print(f"  Збережено: {output_file}")
            plt.close()

    def check_keypoint_ranges(self):
        """Перевірка діапазонів координат всіх точок"""
        print("\nАналіз діапазонів координат...\n")

        all_keypoints = []

        for category_folder in self.data_dir.iterdir():
            if not category_folder.is_dir():
                continue

            for npy_file in category_folder.glob('*.npy'):
                kp = np.load(npy_file)
                all_keypoints.append(kp[:, :3])  # Тільки x, y, z

        if len(all_keypoints) == 0:
            print("Дані для аналізу відсутні")
            return

        all_keypoints = np.vstack(all_keypoints)

        print("Діапазони координат:")
        print(f"  X: [{all_keypoints[:, 0].min():.3f}, {all_keypoints[:, 0].max():.3f}]")
        print(f"  Y: [{all_keypoints[:, 1].min():.3f}, {all_keypoints[:, 1].max():.3f}]")
        print(f"  Z: [{all_keypoints[:, 2].min():.3f}, {all_keypoints[:, 2].max():.3f}]")

        # Перевірка аномалій
        if all_keypoints[:, 0].min() < 0 or all_keypoints[:, 0].max() > 1:
            print("  УВАГА: Координата X виходить за межі [0, 1]")
        if all_keypoints[:, 1].min() < 0 or all_keypoints[:, 1].max() > 1:
            print("  УВАГА: Координата Y виходить за межі [0, 1]")


if __name__ == "__main__":
    print("Перевірка якості даних\n")

    checker = DataQualityChecker(data_dir='../processed_data')

    stats = checker.check_all()

    checker.check_keypoint_ranges()

    checker.visualize_samples()

