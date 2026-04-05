import numpy as np

IDX = {
    'NOSE': 0,
    'LEFT_SHOULDER': 11,  'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,     'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,     'RIGHT_WRIST': 16,
    'LEFT_HIP': 23,       'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,      'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,     'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29,      'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31,'RIGHT_FOOT_INDEX': 32,
}


def _angle(a, b, c):
    """Кут у точцi b (градуси)."""
    ba = a - b
    bc = c - b
    n  = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / n, -1, 1))))


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _hip_mid(kp):
    return (kp[IDX['LEFT_HIP']] + kp[IDX['RIGHT_HIP']]) / 2


def _shoulder_mid(kp):
    return (kp[IDX['LEFT_SHOULDER']] + kp[IDX['RIGHT_SHOULDER']]) / 2


def _shoulder_width(kp):
    return _dist(kp[IDX['LEFT_SHOULDER']], kp[IDX['RIGHT_SHOULDER']]) + 1e-6


def _front_back(kp):
    """Визначає яка нога попереду (менший Z = ближче до камери)."""
    lz = kp[IDX['LEFT_KNEE']][2]
    rz = kp[IDX['RIGHT_KNEE']][2]
    return ('LEFT', 'RIGHT') if lz < rz else ('RIGHT', 'LEFT')


def _back_tilt_deg(kp):
    """Кут нахилу торсу вперед вiд вертикалi (0 = прямо)."""
    spine = _shoulder_mid(kp) - _hip_mid(kp)
    # У MediaPipe Y росте вниз, тому вертикаль = [0,-1,0]
    vert  = np.array([0.0, -1.0, 0.0])
    n = np.linalg.norm(spine) + 1e-8
    return float(np.degrees(np.arccos(np.clip(np.dot(spine, vert) / n, -1, 1))))


def _thigh_angle_from_horiz(kp, side):
    """Кут стегна вiд горизонталi (0 = горизонтально)."""
    hip   = kp[IDX[f'{side}_HIP']]
    knee  = kp[IDX[f'{side}_KNEE']]
    vec   = knee[:2] - hip[:2]
    return float(np.degrees(np.arctan2(abs(vec[1]), abs(vec[0]) + 1e-8)))


def _heel_lifted(kp, side):
    """P'ята вище носка (пiдйом)."""
    heel = kp[IDX[f'{side}_HEEL']]
    toe  = kp[IDX[f'{side}_FOOT_INDEX']]
    return bool(heel[1] < toe[1] - 0.025)


def _knee_valgus(kp, side):
    """Колiно завалене всередину."""
    hip   = kp[IDX[f'{side}_HIP']]
    knee  = kp[IDX[f'{side}_KNEE']]
    ankle = kp[IDX[f'{side}_ANKLE']]
    mid_x = (hip[0] + ankle[0]) / 2
    if side == 'LEFT':
        return bool(knee[0] > mid_x + 0.015)
    else:
        return bool(knee[0] < mid_x - 0.015)


def _foot_angle(kp, side):
    """Кут стопи вiд горизонталi (носок вiдносно п'яти)."""
    heel = kp[IDX[f'{side}_HEEL']]
    toe  = kp[IDX[f'{side}_FOOT_INDEX']]
    vec  = toe[:2] - heel[:2]
    return float(abs(np.degrees(np.arctan2(abs(vec[1]), abs(vec[0]) + 1e-8))))


# ─── Перевiрки по стiйках ────────────────────────────────────────────────────

# GONGBU ─────────────────────────────────────────────────────────────────────

def _gongbu_knee_over_foot(kp):
    front, _ = _front_back(kp)
    knee  = kp[IDX[f'{front}_KNEE']]
    ankle = kp[IDX[f'{front}_ANKLE']]
    return bool(abs(knee[0] - ankle[0]) > _shoulder_width(kp) * 0.18)


def _gongbu_thigh_horiz(kp):
    front, _ = _front_back(kp)
    ang = _thigh_angle_from_horiz(kp, front)
    return bool(ang < 12)  # стегно надто крутe, не горизонтальне


def _gongbu_back_foot_lifted(kp):
    _, back = _front_back(kp)
    front, _ = _front_back(kp)
    back_heel  = kp[IDX[f'{back}_HEEL']]
    front_ankle = kp[IDX[f'{front}_ANKLE']]
    return bool(back_heel[1] < front_ankle[1] - 0.04)


def _gongbu_back_foot_turn(kp):
    _, back = _front_back(kp)
    ang = _foot_angle(kp, back)
    return bool(ang < 8)  # стопа не повернута всередину


# MABU ───────────────────────────────────────────────────────────────────────

def _mabu_thighs_horiz(kp):
    for side in ('LEFT', 'RIGHT'):
        if _thigh_angle_from_horiz(kp, side) > 22:
            return True
    return False


def _mabu_stance_narrow(kp):
    ankle_w = _dist(kp[IDX['LEFT_ANKLE']], kp[IDX['RIGHT_ANKLE']])
    return bool(ankle_w < _shoulder_width(kp) * 0.85)


def _mabu_heels_lifted(kp):
    return _heel_lifted(kp, 'LEFT') or _heel_lifted(kp, 'RIGHT')


def _mabu_knee_valgus(kp):
    return _knee_valgus(kp, 'LEFT') or _knee_valgus(kp, 'RIGHT')


def _mabu_toes_outward(kp):
    for side in ('LEFT', 'RIGHT'):
        if _foot_angle(kp, side) >= 42:
            return True
    return False


def _mabu_torso_tilt(kp):
    return bool(_back_tilt_deg(kp) >= 38)


# SUIBU ────────────────────────────────────────────────────────────────

def _suibu_support_thigh(kp):
    _, back = _front_back(kp)
    return bool(_thigh_angle_from_horiz(kp, back) > 22)


def _suibu_heel(kp):
    _, back = _front_back(kp)
    return _heel_lifted(kp, back)


# PUBU ───────────────────────────────────────────────────────────────────────

def _pubu_rear_not_bent(kp):
    _, back = _front_back(kp)
    ang = _angle(kp[IDX[f'{back}_HIP']], kp[IDX[f'{back}_KNEE']], kp[IDX[f'{back}_ANKLE']])
    return bool(ang > 85)


def _pubu_straight_bent(kp):
    front, _ = _front_back(kp)
    ang = _angle(kp[IDX[f'{front}_HIP']], kp[IDX[f'{front}_KNEE']], kp[IDX[f'{front}_ANKLE']])
    return bool(ang < 158)


def _pubu_front_foot(kp):
    front, _ = _front_back(kp)
    return bool(_foot_angle(kp, front) < 18)


# TISI ───────────────────────────────────────────────────────────────────────

def _tisi_knee_height(kp):
    lk_y = kp[IDX['LEFT_KNEE']][1]
    rk_y = kp[IDX['RIGHT_KNEE']][1]
    raised_y = min(lk_y, rk_y)
    hip_y    = _hip_mid(kp)[1]
    return bool(raised_y > hip_y - 0.01)


def _tisi_toe_direction(kp):
    lk_y = kp[IDX['LEFT_KNEE']][1]
    rk_y = kp[IDX['RIGHT_KNEE']][1]
    side  = 'LEFT' if lk_y < rk_y else 'RIGHT'
    toe   = kp[IDX[f'{side}_FOOT_INDEX']]
    heel  = kp[IDX[f'{side}_HEEL']]
    toe_lower  = bool(toe[1] > heel[1] + 0.015)
    center_x   = _hip_mid(kp)[0]
    toe_inward = (toe[0] > heel[0] - 0.01) if side == 'LEFT' else (toe[0] < heel[0] + 0.01)
    return not (toe_lower and toe_inward)


STANCE_RULES = {
    'gongbu': {
        'name_ua':  'Гун бу',
        'name_zh':  '弓步',
        'subtitle': 'Позицiя «лучника»',
        'rule_id':  50,
        'checks': [
            {'id': 'knee_over_foot',    'msg': 'Колiно передньої ноги не над стопою',                       'sev': 'error',   'fn': _gongbu_knee_over_foot},
            {'id': 'thigh_horiz',       'msg': 'Стегно передньої ноги не на рiвнi горизонту',               'sev': 'error',   'fn': _gongbu_thigh_horiz},
            {'id': 'back_foot_lifted',  'msg': 'Стопа задньої ноги вiдiрвана вiд килима',                   'sev': 'error',   'fn': _gongbu_back_foot_lifted},
            {'id': 'back_foot_turn',    'msg': 'Стопа задньої ноги не повернута всередину навскiс уперед',  'sev': 'warning', 'fn': _gongbu_back_foot_turn},
        ],
    },

    'mabu': {
        'name_ua':  'Ма бу',
        'name_zh':  '馬步',
        'subtitle': 'Позицiя «вершника»',
        'rule_id':  51,
        'checks': [
            {'id': 'thighs_horiz',   'msg': 'Стегна не на рiвнi горизонту — опустись нижче',        'sev': 'error',   'fn': _mabu_thighs_horiz},
            {'id': 'stance_narrow',  'msg': 'Вiдстань мiж стопами менша ширини плечей',              'sev': 'error',   'fn': _mabu_stance_narrow},
            {'id': 'heels_lifted',   'msg': "П'ята вiдривається вiд килима",                         'sev': 'error',   'fn': _mabu_heels_lifted},
            {'id': 'knee_valgus',    'msg': 'Колiна завалено всередину — розгорни назовнi',          'sev': 'error',   'fn': _mabu_knee_valgus},
            {'id': 'toes_outward',   'msg': 'Пальцi стоп направленi назовнi на 45° або бiльше',     'sev': 'warning', 'fn': _mabu_toes_outward},
            {'id': 'torso_tilt',     'msg': 'Корпус нахилений вперед — випрями спину',               'sev': 'error',   'fn': _mabu_torso_tilt},
        ],
    },

    'suibu': {
        'name_ua':  'Сюй бу',
        'name_zh':  '虛步',
        'subtitle': 'Позицiя «порожнiй крок»',
        'rule_id':  52,
        'checks': [
            {'id': 'support_thigh', 'msg': 'Стегно опорної ноги не на рiвнi горизонту',   'sev': 'error', 'fn': _suibu_support_thigh},
            {'id': 'support_heel',  'msg': "П'ята опорної ноги вiдривається вiд килима",  'sev': 'error', 'fn': _suibu_heel},
        ],
    },

    'pubu': {
        'name_ua':  'Пу бу',
        'name_zh':  '仆步',
        'subtitle': 'Позицiя «ковзаючий крок»',
        'rule_id':  53,
        'checks': [
            {'id': 'rear_not_bent',  'msg': 'Опорна нога не зiгнута, стегно не торкається гомiлки', 'sev': 'error', 'fn': _pubu_rear_not_bent},
            {'id': 'straight_bent',  'msg': 'Пряма нога зiгнута — випрями повнiстю',                'sev': 'error', 'fn': _pubu_straight_bent},
            {'id': 'front_foot',     'msg': 'Стопа передньої ноги не повернута всередину (45°+)',    'sev': 'warning', 'fn': _pubu_front_foot},
        ],
    },

    'tisi': {
        'name_ua':  'Тi сi ду лi',
        'name_zh':  '提膝獨立',
        'subtitle': 'Позицiя з пiднятим колiном',
        'rule_id':  26,
        'checks': [
            {'id': 'knee_height',    'msg': 'Пiднiмiть колiно вище рiвня поясу',                          'sev': 'error',   'fn': _tisi_knee_height},
            {'id': 'toe_direction',  'msg': 'Носок не спрямований всередину вниз по дiагоналi',           'sev': 'warning', 'fn': _tisi_toe_direction},
        ],
    },
}


# ─── Головна функцiя ─────────────────────────────────────────────────────────

def validate(stance_key: str, kp_raw: np.ndarray) -> dict:
    """
    Перевiряє стiйку за офiцiйними правилами.

    Args:
        stance_key : ключ ('mabu', 'gongbu', 'pubu', 'suibu', 'tisi')
        kp_raw     : (33, 4) — координати MediaPipe (x, y, z, visibility)

    Returns:
        {'hints': [...], 'stance_info': {...}}
    """
    rules = STANCE_RULES.get(stance_key)
    if not rules:
        return {'hints': [], 'stance_info': {}}

    kp    = kp_raw[:, :3]
    hints = []

    for chk in rules['checks']:
        try:
            fired = chk['fn'](kp)
        except Exception:
            fired = False
        if fired:
            hints.append({'id': chk['id'], 'text': chk['msg'], 'severity': chk['sev']})

    return {
        'hints': hints,
        'stance_info': {
            'name_ua':  rules['name_ua'],
            'name_zh':  rules['name_zh'],
            'subtitle': rules['subtitle'],
            'rule_id':  rules['rule_id'],
        },
    }
