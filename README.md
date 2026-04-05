# 武術 Wushu Stance Analyzer

Real-time система аналізу стійок ушу на основі комп'ютерного зору.

Програма через веб-камеру розпізнає стійку спортсмена, класифікує її та надає підказки щодо виправлення помилок згідно з **офіційними правилами змагань**.

---

## Демо

```
python scripts/06_realtime_clean.py
```

Відкривається вікно камери з лініями скелета та браузер з аналізом стійки в реальному часі.

---

## Підтримувані стійки

| Стійка | Китайська | Назва | Пункт правил |
|--------|-----------|-------|--------------|
| `mabu`   | 馬步 | Позиція «вершника» | код 51 |
| `gongbu` | 弓步 | Позиція «лучника»  | код 50 |
| `pubu`   | 仆步 | Позиція «ковзаючий крок» | код 53 |
| `suibu`  | 虛步 | Позиція «порожній крок» | код 52 |
| `tisi`   | 提膝獨立 | Позиція з піднятим коліном | код 26 |

---

## Архітектура проєкту

```
wushu_vision/
│
├── raw_images/                 
│   ├── mabu_good/              
│   ├── mabu_bad/
│   ├── gongbu_good/
│   └── ...
│
├── processed_data/             
│   ├── mabu_good/
│   ├── dataset.pkl            
│   └── dataset.npz
│
├── models/
│   └── stance_classifier.pkl   
│
├── visualizations/             
│
└── scripts/
    ├── 01_extract_keypoints.py  
    ├── 02_check_data.py        
    ├── 03_build_dataset.py     
    ├── 04_inspect_dataset.py   
    ├── 05_train_model.py        
    ├── 06_realtime_clean.py    
    ├── 07_test_model.py     
    ├── stance_rules.py              
    └── web_dashboard.html         
```

---

## Як це працює

```
Камера → MediaPipe Pose → 33 keypoints
    ↓
Нормалізація (відносно тазу та довжини тіла)
    ↓
Витягування 13 ознак (кути суглобів, кути стійки)
    ↓
Random Forest класифікатор → назва стійки
    ↓
Перевірка за офіційними правилами змагань
    ↓
WebSocket → браузер → підказки для виправлення
```

**Технічний стек:**
- **MediaPipe Pose** — визначення 33 точок тіла
- **Random Forest** (scikit-learn) — класифікація стійки
- **OpenCV** — захоплення відео, відображення скелета
- **WebSocket** (websockets) — передача даних у браузер
- **HTTP**  —  веб-сторінка

---

## Встановлення та запуск

### 1. Клонування репозиторію

```bash
git clone https://github.com/YOUR_USERNAME/wushu-stance-analyzer.git
cd wushu-stance-analyzer
```

### 2. Створення віртуального середовища

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 4. Запуск 

```bash
python scripts/06_realtime_clean.py
```

Автоматично відкриється браузер на `http://localhost:8080`

### 5. Навчання моделі з нуля

```bash
# Крок 1: Підготуй фото у папці raw_images/{stance_good|bad}/

# Крок 2: Витягни keypoints
python scripts/01_extract_keypoints.py

# Крок 3: Перевір якість
python scripts/02_check_data.py

# Крок 4: Побудуй датасет
python scripts/03_build_dataset.py

# Крок 5: Перевір датасет
python scripts/04_inspect_dataset.py

# Крок 6: Навчи модель
python scripts/05_train_model.py

# Крок 7: Запуск
python scripts/06_realtime_clean.py
```

---

