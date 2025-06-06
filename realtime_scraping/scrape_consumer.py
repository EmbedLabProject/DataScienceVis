from confluent_kafka import Consumer
from sklearn.preprocessing import MultiLabelBinarizer
from fastavro import parse_schema, schemaless_reader
import io, json, pandas as pd
import time
import math
import numpy as np
import xgboost as xgb
from datetime import datetime
from datetime import datetime, timezone
import re
import math
import shap
################## config here #######################################

# current path used is the relative path from this folder route
# open schema file, change path to your syatem path
schema = {
  "type": "record",
  "name": "TraffyReport",
  "fields": [
    {"name": "ticket_id", "type": "string"},
    {"name": "report_time", "type": "string"},
    {"name": "address", "type": "string"},
    {"name": "district", "type": "string"},         # <-- added
    {"name": "subdistrict", "type": "string"},      # <-- added
    {"name": "status", "type": "string"},
    {"name": "description", "type": "string"},
    {"name": "resolution", "type": "string"},
    {"name": "reporting_agency", "type": "string"},
    {"name": "tags", "type": "string"},
    {"name": "upvotes", "type": "int"},
    {"name": "image_url", "type": "string"},
    {"name": "latitude", "type": ["null", "double"], "default": None},
    {"name": "longitude", "type": ["null", "double"], "default": None}
  ]
}

parsed_schema = parse_schema(schema)

# Kafka consumer setup, please replace the server and and group if needed
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'traffy-csv-timer',
    'auto.offset.reset': 'earliest'
})

# topic setup, change the text inside to the topic in producer
consumer.subscribe(["traffy_reports"])

# output path, where your csv will be at
csv_path = "./realtime_scraping/out/traffy_reports_latest.csv"
model_path = "./model.json"



# set the amount to the same as producer
report_per_scan = 6
###################################################################


def convert_thai_datetime(thai_str):
    # Clean and normalize
    thai_str = re.sub(r'น\.?|น', '', thai_str).strip()
    parts = thai_str.split()

    if len(parts) != 4:
        raise ValueError(f"Unexpected datetime format: {thai_str}")

    day = int(parts[0])
    month_thai = parts[1]
    year_be = int(parts[2])
    time_part = parts[3]

    month_map = {
        'ม.ค.': 1, 'ก.พ.': 2, 'มี.ค.': 3, 'เม.ย.': 4,
        'พ.ค.': 5, 'มิ.ย.': 6, 'ก.ค.': 7, 'ส.ค.': 8,
        'ก.ย.': 9, 'ต.ค.': 10, 'พ.ย.': 11, 'ธ.ค.': 12
    }

    month = month_map.get(month_thai)
    if not month:
        raise ValueError(f"Unknown Thai month: {month_thai}")

    year_ad = year_be + 2500 - 543 if year_be < 100 else year_be - 543

    dt_str = f"{year_ad}-{month:02d}-{day:02d} {time_part}"
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")

    # Return string without timezone (naive datetime)
    #return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # keep microseconds if needed
    print(dt_str+':09.453003+00')
    return dt_str+':09.453003+00'

# Recreate the model with the same parameters
model = xgb.Booster()

# Load the model structure into the internal Booster
model.load_model(model_path)

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

# ตัวอย่างการเรียกใช้งาน:
# preds, margins, est, conf, details = predict_time(
#     model, ['คลอง','สัตว์จรจัด'], '2021-09-03 12:51:09.453003+00',
#     "13.81865,100.53084", 'บางซื่อ', 'บางซื่อ'
# )
# print(f"Class probabilities: {preds}\nEstimated days: {est}\nConfidence: {conf}")

print("🟢 Consumer started. Will overwrite every 5 minutes...")

try:
    while True:
        batch = []
        start_time = time.time()

        # collect reports during interval
        while len(batch) < report_per_scan:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                print("Error:", msg.error())
                continue

            buf = io.BytesIO(msg.value())
            report = schemaless_reader(buf, parsed_schema)
            batch.append(report)

        # save latest batch to CSV
        if batch:
            predicted_batch = []

            for report in batch:
                # Extract required input fields for prediction
                tags_raw = report.get("tags", "")
                tags = tags_raw.split(",") if isinstance(tags_raw, str) else tags_raw
                raw_time_str = report.get("report_time", "")
                try:
                    time_str = convert_thai_datetime(raw_time_str)
                except Exception as e:
                    print(f"Failed to convert time '{raw_time_str}': {e}")
                    time_str = ""
                report["report_time"] = time_str
                lat = report.get("latitude")
                lon = report.get("longitude")
                location = f"{lat},{lon}" if lat is not None and lon is not None else ""
                district = report.get("district", "").removeprefix("เขต")
                subdistrict = report.get("subdistrict", "").removeprefix("แขวง")

                try:
                    # Call your model prediction function
                    print(tags)
                    print( time_str)
                    print( location)
                    print( district)
                    print( subdistrict)
                    print(type(tags), type(time_str), type(location), type(district), type(subdistrict))
                    preds, margins, est, conf, details = predict_time(
                        model, tags, time_str, location, district, subdistrict
                    )
                except Exception as e:
                    print(f"⚠️ Prediction failed for ticket_id {report.get('ticket_id')}: {e}")
                    est, conf = None, None

                # Add results to report
                est_labels = ["1 day", "1-3 days", "3-7 days", "7-21 days", "21-90 days", "90+ days"]
                if(est is None) :
                    report["est"] = "Unknown"
                else :
                    report["est"] = est_labels[est]
                report["conf"] = conf
                predicted_batch.append(report)
        
        if predicted_batch:
            df = pd.DataFrame(predicted_batch)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ {len(df)} reports (with predictions) saved to {csv_path}.")
    
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    consumer.close()
