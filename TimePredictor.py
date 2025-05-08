import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer
import math
import shap

district_office_location = {
    "พระนคร": (13.764928875843303, 100.49876110201305),
    "ดุสิต": (13.777137865519553, 100.52060070041834),
    "หนองจอก": (13.855559348924416, 100.86250879511516),
    "บางรัก": (13.730665492976234, 100.52348632968096),
    "บางเขน": (13.873390441022337, 100.59607817647071),
    "บางกะปิ": (13.765784458084667, 100.64746466418079),
    "ปทุมวัน": (13.744775870775214, 100.5221679872862),
    "ป้อมปราบศัตรูพ่าย": (13.758131278192316, 100.51305765683877),
    "พระโขนง": (13.70245388463174, 100.60201995932142),
    "มีนบุรี": (13.813520000292137, 100.73129315090954),
    "ลาดกระบัง": (13.722351512642533, 100.75973821018188),
    "ยานนาวา": (13.696255941440228, 100.54241169046436),
    "สัมพันธวงศ์": (13.731558621128627, 100.51390480776924),
    "พญาไท": (13.779841550638082, 100.54260649347228),
    "ธนบุรี": (13.724893996581253, 100.48586473528377),
    "บางกอกใหญ่": (13.72329283806953, 100.47632489414428),
    "ห้วยขวาง": (13.776625058927335, 100.57937113544617),
    "คลองสาน": (13.73053678969251, 100.50925221037632),
    "ตลิ่งชัน": (13.776756576321823, 100.45648886139783),
    "บางกอกน้อย": (13.770793361442882, 100.46804511658367),
    "บางขุนเทียน": (13.66075667619237, 100.4353903434257),
    "ภาษีเจริญ": (13.714805942839531, 100.43692254219468),
    "หนองแขม": (13.70541823094582, 100.34920888486124),
    "ราษฎร์บูรณะ": (13.681850188598025, 100.5057611421854),
    "บางพลัด": (13.793908773714579, 100.5050753627661),
    "ดินแดง": (13.769825608522584, 100.55312850612616),
    "บึงกุ่ม": (13.785556222552538, 100.66950529166158),
    "สาทร": (13.707986680426469, 100.52630474790409),
    "บางซื่อ": (13.809566088625214, 100.53722971209244),
    "จตุจักร": (13.828652086836676, 100.55997360729539),
    "บางคอแหลม": (13.692952387248127, 100.50253057855271),
    "ประเวศ": (13.717246339846502, 100.69467236477642),
    "คลองเตย": (13.708220611673642, 100.58369685106754),
    "สวนหลวง": (13.730170291834963, 100.651251012509),
    "จอมทอง": (13.677562701369771, 100.4841897677573),
    "ดอนเมือง": (13.90995378737537, 100.59470365026687),
    "ราชเทวี": (13.759173863154476, 100.5341275810791),
    "ลาดพร้าว": (13.803577537305433, 100.60752329243601),
    "วัฒนา": (13.742408399794114, 100.5859127880872),
    "บางแค": (13.696203940032568, 100.4091302644525),
    "หลักสี่": (13.887444345533599, 100.57891693181126),
    "สายไหม": (13.895108760246032, 100.66051827468355),
    "คันนายาว": (13.799302669559399, 100.68267045862126),
    "สะพานสูง": (13.768967962029048, 100.68565663792235),
    "วังทองหลาง": (13.764234425711646, 100.60572336211685),
    "คลองสามวา": (13.859902323144107, 100.70417999674628),
    "บางนา": (13.681182642473885, 100.59210380320427),
    "ทวีวัฒนา": (13.77301485649326, 100.35310584806938),
    "ทุ่งครุ": (13.61136886293151, 100.50876871494297),
    "บางบอน": (13.63394263372994, 100.3689626684714)
}

def get_office_location(district):
    return district_office_location[district]

def haversine(coord1, coord2):
    R = 6371.0
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def distFromDistOff(office_coords, coords):
    coor = coords.split(",")
    return haversine((float(coor[1]), float(coor[0])), office_coords)


def predict_time(model, problem_types, timestamp, coords, district, subdistrict):
    """
    พยากรณ์เวลาที่ใช้ในการแก้ไขปัญหาเฉพาะกรณีเดียว

    Parameters:
        model: โมเดล XGBoost trained
        problem_types: list ของประเภทปัญหา
        timestamp: single datetime string หรือ pd.Timestamp
        coords: tuple (lat, lon)
        district: string ชื่อเขต
        subdistrict: string ชื่อแขวง

    Returns:
        preds: numpy array shape (1, n_classes)
        margins: numpy array shape (1, n_classes)
        estimated_time: float ชั่วโมง (weighted sum)
        total_conf: float ค่า confidence รวม
        details: dict (reserved for SHAP หรือ feature importance)
    """
    
    ts = pd.to_datetime(timestamp, format='mixed')
    timestamp_date = ts.tz_localize(None).floor('s')

    df = pd.DataFrame({
        'hour_of_day': [ts.hour],
        'unix_time': [int(timestamp_date.timestamp())]
    })

    office_coords = get_office_location(district)
    df['dist_from_office'] = distFromDistOff(office_coords, coords)

    # one-hot
    df = df.assign(district=[district], subdistrict=[subdistrict])
    df = pd.get_dummies(df, columns=['district','subdistrict'], prefix=['district','subdistrict'], prefix_sep='__', dtype=int)

    # one-hot type
    mlb = MultiLabelBinarizer()
    type_df = pd.DataFrame(mlb.fit_transform([problem_types]), columns=[f"type_{c}" for c in mlb.classes_])
    df = pd.concat([df, type_df], axis=1)

    df = df.reindex(columns=model.feature_names, fill_value=0)

    dmat = xgb.DMatrix(df, feature_names=model.feature_names)
    preds = model.predict(dmat)[0]
    margins = model.predict(dmat, output_margin=True)[0]

    # estimate class index
    estimated_time = int(preds.argmax())
    total_conf = float(preds.max()) * 100

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    groupings = {
        "Problem Type": [col for col in model.feature_names if col.startswith("type_")],
        "Date": ["unix_time"],
        "Time": ["hour_of_day"],
        "Location": ["dist_from_office"] + [col for col in model.feature_names if col.startswith("district__") or col.startswith("subdistrict__")]
    }

    contrib_data = []
    feature_index_map = {name: idx for idx, name in enumerate(model.feature_names)}
    labels = ["1 day", "1-3 days", "3-7 days", "7-21 days", "21-90 days", "90+ days"]

    for i, class_label in enumerate(labels):
        shap_row = shap_values[0, :, i]
        contributions = []
        for group in ["Problem Type", "Date", "Time", "Location"]:
            indices = [feature_index_map[c] for c in groupings[group] if c in feature_index_map]
            score = sum(abs(shap_row[j]) for j in indices)
            contributions.append(score)
        total = sum(contributions)
        contributions = [(c / total) * 100 if total else 0 for c in contributions]

        contrib_data.append({
            "label": class_label,
            "confidence": preds[i] * 100,
            "contributions": contributions
        })

    contrib_data.sort(key=lambda x: -x["confidence"])

    return preds, margins, estimated_time, total_conf, contrib_data