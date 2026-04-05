import sys, threading, asyncio, json, time, webbrowser
from collections import Counter, deque
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, HTTPServer

import cv2
import mediapipe as mp
import numpy as np
import joblib
import websockets

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / 'models' / 'stance_classifier.pkl'
HTTP_PORT  = 8080
WS_PORT    = 8765

sys.path.insert(0, str(BASE_DIR))
from stance_rules import validate, STANCE_RULES


# ── HTTP сервер ──────────────────────────────────────────────────────────────

class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(BASE_DIR), **kw)
    def log_message(self, *a):
        pass

def _http_thread():
    HTTPServer(('0.0.0.0', HTTP_PORT), _Handler).serve_forever()


# ── WebSocket сервер ─────────────────────────────────────────────────────────

_clients = set()
_loop    = None

async def _ws_handler(ws):
    _clients.add(ws)
    try:
        async for _ in ws:
            pass
    except:
        pass
    finally:
        _clients.discard(ws)

async def _send(msg):
    for c in list(_clients):
        try:
            await c.send(msg)
        except:
            _clients.discard(c)

def _ws_thread():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    async def _run():
        async with websockets.serve(_ws_handler, '0.0.0.0', WS_PORT):
            await asyncio.Future()
    _loop.run_until_complete(_run())

def ws_send(data):
    if _loop:
        asyncio.run_coroutine_threadsafe(_send(json.dumps(data, ensure_ascii=False)), _loop)


# ── Скелет — тiльки лiнiї ───────────────────────────────────────────────────

LINES = [
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(12,14),(14,16),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32),
]

def draw_skeleton(frame, lms, h, w):
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
    vis = [lm.visibility for lm in lms]
    for a, b in LINES:
        if vis[a] > 0.35 and vis[b] > 0.35:
            cv2.line(frame, pts[a], pts[b], (0,215,80), 3, cv2.LINE_AA)


# ── Нормалiзацiя ─────────────────────────────────────────────────────────────

def normalize(kp):
    xyz   = kp[:, :3].copy()
    hip   = (xyz[23] + xyz[24]) / 2
    sh    = (xyz[11] + xyz[12]) / 2
    scale = float(np.linalg.norm(sh - hip)) + 1e-6
    return (xyz - hip) / scale


# ── Екстракцiя ознак (13 штук — точно як у моделi) ──────────────────────────

def _ang(kp, a, b, c):
    ba = kp[a]-kp[b]; bc = kp[c]-kp[b]
    cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
    return float(np.degrees(np.arccos(np.clip(cos,-1,1))))

def extract_features(kp):
    lk = _ang(kp,23,25,27)
    rk = _ang(kp,24,26,28)
    lh = _ang(kp,11,23,25)
    rh = _ang(kp,12,24,26)
    shm = (kp[11]+kp[12])/2
    hm  = (kp[23]+kp[24])/2
    sp  = shm - hm
    bk  = float(np.degrees(np.arccos(np.clip(
              np.dot(sp,np.array([0,1,0]))/(np.linalg.norm(sp)+1e-8), -1, 1))))
    sw  = float(np.linalg.norm(kp[27]-kp[28]))
    hh  = float(hm[1])
    sd  = float(abs(kp[27][2]-kp[28][2]))
    ks  = abs(lk-rk)
    hw  = float(np.linalg.norm(kp[23]-kp[24]))
    le  = _ang(kp,11,13,23)
    re  = _ang(kp,12,14,24)
    return [lk, rk, lh, rh, bk, sw, hh, sd, ks, hw, le, re, 1.0]

def get_angles(kp):
    shm = (kp[11]+kp[12])/2
    hm  = (kp[23]+kp[24])/2
    sp  = shm - hm
    back = float(np.degrees(np.arccos(np.clip(
               np.dot(sp,np.array([0,1,0]))/(np.linalg.norm(sp)+1e-8),-1,1))))
    return {
        'Лiве колiно':  round(_ang(kp,23,25,27),1),
        'Праве колiно': round(_ang(kp,24,26,28),1),
        'Спина':        round(back,1),
        'Лiве стегно':  round(_ang(kp,11,23,25),1),
        'Праве стегно': round(_ang(kp,12,24,26),1),
    }


# ── Основний детектор ─────────────────────────────────────────────────────────

class Detector:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Модель не знайдена: {MODEL_PATH}")
        data = joblib.load(MODEL_PATH)
        self.clf     = data['model']
        self.rev_map = data['reverse_label_map']
        print(f"Модель завантажена. Класи: {self.rev_map}")

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.buf = deque(maxlen=10)

    def _smooth(self, pred, conf):
        if conf < 0.4:
            return None
        self.buf.append(pred)
        if len(self.buf) < 3:
            return pred
        top, cnt = Counter(self.buf).most_common(1)[0]
        return top if cnt/len(self.buf) >= 0.5 else pred

    def run(self, cam=0):
        cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            print(f"[ERROR] Камера {cam} недоступна"); return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cv2.namedWindow('Wushu', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Wushu', 960, 540)

        print("Камера OK. Q/ESC — вихiд.\n")
        last_print = ''

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01); continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            res   = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            payload = {'detected': False}

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                draw_skeleton(frame, lms, h, w)

                # координати
                kp_raw  = np.array([[lm.x,lm.y,lm.z,lm.visibility] for lm in lms], dtype=np.float32)
                kp_norm = normalize(kp_raw)

                # класифiкацiя
                feats = extract_features(kp_norm)
                arr   = np.array(feats, dtype=np.float32).reshape(1,-1)
                pred  = int(self.clf.predict(arr)[0])
                probs = self.clf.predict_proba(arr)[0]
                conf  = float(probs[pred])

                smoothed   = self._smooth(pred, conf)
                stance_key = self.rev_map.get(smoothed, '') if smoothed is not None else ''

                # валiдацiя
                val  = validate(stance_key, kp_raw)
                info = val['stance_info']

                # кути для вiдображення (беремо з kp_raw бо вони не нормалiзованi — правила їх порiвнюють з порогами)
                angles = get_angles(kp_norm)

                payload = {
                    'detected':    True,
                    'stance_key':  stance_key,
                    'name_ua':     info.get('name_ua', stance_key),
                    'name_zh':     info.get('name_zh', ''),
                    'subtitle':    info.get('subtitle', ''),
                    'rule_id':     info.get('rule_id', ''),
                    'confidence':  round(conf*100),
                    'hints':       val['hints'],
                    'angles':      angles,
                }

                # консоль debug
                line = f"[{stance_key}] {info.get('name_ua','')}  conf={conf:.2f}  помилок={len(val['hints'])}"
                if line != last_print:
                    print(line)
                    last_print = line

            ws_send(payload)
            cv2.imshow('Wushu', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()


# ── Старт ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("="*55)
    print("  Ушу — Аналiз стiйок")
    print("="*55)

    threading.Thread(target=_http_thread, daemon=True).start()
    time.sleep(0.3)
    print(f"HTTP:      http://localhost:{HTTP_PORT}  OK")

    threading.Thread(target=_ws_thread, daemon=True).start()
    time.sleep(0.5)
    print(f"WebSocket: ws://localhost:{WS_PORT}  OK")

    url = f"http://localhost:{HTTP_PORT}/web_dashboard.html"
    webbrowser.open(url)
    print(f"Браузер:   {url}")
    print("-"*55)

    try:
        Detector().run(cam=0)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
    except KeyboardInterrupt:
        print("\nПерервано.")
    except Exception:
        import traceback; traceback.print_exc()
