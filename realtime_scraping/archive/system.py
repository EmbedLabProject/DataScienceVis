from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xgboost as xgb
import math
import re
import os
from sklearn.preprocessing import MultiLabelBinarizer

# ---------------- CONFIG ----------------
stream_csv_path = "./realtime_scraping/out/traffy_reports_stream.csv"
output_csv_path = "./realtime_scraping/out/traffy_reports_latest.csv"
model_path = "./model.json"

# ---------------- DISTRICT COORDINATES ----------------
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

model = xgb.Booster()
model.load_model(model_path)

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
    # print(dt_str+':09.453003+00')
    return dt_str+':09.453003+00'

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


# กำหนดชั่วโมงเฉลี่ย (representative hours) สำหรับแต่ละคลาส
# class 1 (<1 วัน): 12h, class 2 (1-3 วัน): 48h, class 3 (3-7 วัน): 120h,
# class 4 (7-21 วัน): 336h, class 5 (21-90 วัน): 1332h, class 6 (>90 วัน): 2880h
class_hours = np.array([12, 48, 120, 336, 1332, 2880])

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
    # แปลง input เป็นแบบ list/Series
    ts = pd.to_datetime([timestamp])

    # สร้าง DataFrame
    df = pd.DataFrame({
        'hour_of_day': ts.hour,
        'unix_time': ts.astype(int) // 10**9,
    })

    # ระยะจากสำนักงาน
    office_coords = get_office_location(district)
    df['dist_from_office'] = distFromDistOff(office_coords, coords)

    # one-hot เขตและแขวง
    df = df.assign(district=[district], subdistrict=[subdistrict])
    df = pd.get_dummies(df, columns=['district','subdistrict'], prefix=['district','subdistrict'], prefix_sep='__', dtype=int)

    # one-hot ปัญหา
    mlb = MultiLabelBinarizer()
    type_df = pd.DataFrame(mlb.fit_transform([problem_types]), columns=[f"type_{c}" for c in mlb.classes_])
    df = pd.concat([df, type_df], axis=1)

    # เติมคอลัมน์ที่ขาด
    df = df.reindex(columns=model.feature_names, fill_value=0)

    # Predict
    dmat = xgb.DMatrix(df, feature_names=model.feature_names)
    preds = model.predict(dmat)[0]
    margins = model.predict(dmat, output_margin=True)[0]

    # ประมาณเวลารวม
    estimated_time = float((preds * class_hours).sum()) % 24
    total_conf = float(margins.mean())

    # รายละเอียด (placeholder)
    details = {}

    return preds, margins, estimated_time, total_conf, details



def reset_csv():
    open(stream_csv_path, 'w').close()
    print(f"🧹 Cleared file")

def infer_latest_reports():
    if not os.path.exists(stream_csv_path):
        print("⚠️ No CSV found to process.")
        return

    raw_df = pd.read_csv(stream_csv_path)
    if raw_df.empty:
        print("⚠️ No data found in CSV.")
        return

    raw_df = raw_df.sort_values("report_time", ascending=False).head(6)
    enriched = []

    for _, report in raw_df.iterrows():
        try:
            tags = report["tags"].split(",") if isinstance(report["tags"], str) else []
            time_str = convert_thai_datetime(report["report_time"])
            report["report_time"] = time_str
            location = f"{report['latitude']},{report['longitude']}"
            preds, margins, est, conf, _ = predict_time(
                model, tags, time_str, location, report["district"], report["subdistrict"]
            )
            report["est"] = est
            report["conf"] = conf
        except Exception as e:
            print(f"⚠️ Prediction failed for ticket_id {report.get('ticket_id', 'unknown')}: {e}")
            report["est"] = None
            report["conf"] = None

        enriched.append(report)

    pd.DataFrame(enriched).to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Saved top 6 predictions to {output_csv_path}")

# ---------------- DAG DEFINITIONS ----------------
def create_hourly_reset_dag():
    dag = DAG(
        dag_id="reset_traffy_csv_hourly",
        start_date=datetime(2025, 5, 6),
        schedule_interval="@hourly",
        catchup=False
    )
    PythonOperator(
        task_id="clear_raw_csv",
        python_callable=reset_csv,
        dag=dag
    )
    return dag

def create_minutely_infer_dag():
    dag = DAG(
        dag_id="infer_top6_minutely",
        start_date=datetime(2025, 5, 6),
        schedule_interval="* * * * *",
        catchup=False
    )
    PythonOperator(
        task_id="predict_top6_latest_reports",
        python_callable=infer_latest_reports,
        dag=dag
    )
    return dag

reset_dag = create_hourly_reset_dag()
infer_dag = create_minutely_infer_dag()