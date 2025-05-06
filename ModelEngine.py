import xgboost as xgb
import json
import pandas as pd
import numpy as np
from Constant import *
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

model = xgb.XGBClassifier()
is_model_load = False
weight = {}
is_weight_load = False


def load_model() :
    global model
    global is_model_load
    if(is_model_load) :
        return
    model.load_model("model.json") 
    is_model_load = True

def load_weight() :
    global weight
    global is_weight_load
    if(is_weight_load) :
        return
    with open("classWight.json", "w") as f:
        json.dump(weight, f)
    is_weight_load = True




def get_estimate_time(list) :
    time = weight["time"]
    weight = weight["weight"]
    sum = 0
    for i in range(len(time)) :
        sum += time[i] * weight[i]
    return sum



def analyze_problem_types(df: pd.DataFrame, problem_types: list[str]):
    if(len(problem_types) == 0) :
        problem_types = problem_type

    df = df.copy()
    df['timestamp_date'] = pd.to_datetime(df['timestamp_date'], errors='coerce')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df.dropna(subset=['timestamp_date', 'duration', 'distance'], inplace=True)

    # --- 1. Quartiles by Day of Week (same as before)
    weekday_cols = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    q_df_by_day = pd.DataFrame(index=weekday_cols, columns=["Q1", "Q2", "Q3", "Q4"])
    mask = df[problem_types].any(axis=1)
    day_df = df[mask]
    for day in weekday_cols:
        d = day_df[day_df[day] == 1]
        if not d.empty:
            q_df_by_day.loc[day] = d['duration'].quantile([0.25, 0.5, 0.75, 1.0]).values

    # --- 2. Average by District (same as before)
    district_cols = [
        'คลองสาน', 'คลองสามวา', 'คลองเตย', 'คันนายาว', 'จตุจักร', 'จอมทอง',
        'ดอนเมือง', 'ดินแดง', 'ดุสิต', 'ตลิ่งชัน', 'ทวีวัฒนา', 'ทุ่งครุ',
        'ธนบุรี', 'บางกอกน้อย', 'บางกอกใหญ่', 'บางกะปิ', 'บางขุนเทียน',
        'บางคอแหลม', 'บางซื่อ', 'บางนา', 'บางบอน', 'บางพลัด', 'บางรัก',
        'บางเขน', 'บางแค', 'บึงกุ่ม', 'ปทุมวัน', 'ประเวศ', 'ป้อมปราบศัตรูพ่าย',
        'พญาไท', 'พระนคร', 'พระโขนง', 'ภาษีเจริญ', 'มีนบุรี', 'ยานนาวา',
        'ราชเทวี', 'ราษฎร์บูรณะ', 'ลาดกระบัง', 'ลาดพร้าว', 'วังทองหลาง',
        'วัฒนา', 'สวนหลวง', 'สะพานสูง', 'สัมพันธวงศ์', 'สาทร', 'สายไหม',
        'หนองจอก', 'หนองแขม', 'หลักสี่', 'ห้วยขวาง'
    ]
    avg_df_by_district = pd.DataFrame(index=district_cols, columns=['Average Duration'])
    for d in district_cols:
        d_df = day_df[day_df[d] == 1]
        avg_df_by_district.loc[d] = d_df['duration'].mean()

    # --- 3. Time Quartiles by Problem Type
    time_grouped = {}
    for pt in problem_types:
        sub = df[(df['timestamp_date'].dt.year >= 2022) & (df['timestamp_date'].dt.year <= 2024)]
        sub = sub[sub[pt] == 1]
        if sub.empty:
            continue
        q_time = (
            sub.set_index('timestamp_date')
            .resample('2W')['duration']
            .agg(Q1=lambda x: x.quantile(0.25),
                 Q2=lambda x: x.quantile(0.5),
                 Q3=lambda x: x.quantile(0.75),
                 Q4=lambda x: x.quantile(1.0))
            .interpolate(method='time')
        )
        time_grouped[pt] = q_time

    # Total (union of all selected types)
    union_df = df[df[problem_types].any(axis=1)]
    union_df = union_df[(union_df['timestamp_date'].dt.year >= 2022) & (union_df['timestamp_date'].dt.year <= 2024)]
    time_grouped["total"] = (
        union_df.set_index('timestamp_date')
        .resample('2W')['duration']
        .agg(Q1=lambda x: x.quantile(0.25),
             Q2=lambda x: x.quantile(0.5),
             Q3=lambda x: x.quantile(0.75),
             Q4=lambda x: x.quantile(1.0))
        .interpolate(method='time')
    )

    # --- 4. Distance Quartiles by Problem Type
    distance_grouped = {}
    for pt in problem_types:
        sub = df[(df['timestamp_date'].dt.year >= 2022) & (df['timestamp_date'].dt.year <= 2024)]
        sub = sub[sub[pt] == 1]
        if sub.empty:
            continue
        max_d = sub['distance'].max()
        bin_width = max_d / 400 if max_d > 0 else 1
        bins = np.arange(0, max_d + bin_width, bin_width)
        sub['distance_bin'] = pd.cut(sub['distance'], bins, right=False)
        q_dist = sub.groupby('distance_bin')['duration'].agg(
            Q1=lambda x: x.quantile(0.25),
            Q2=lambda x: x.quantile(0.5),
            Q3=lambda x: x.quantile(0.75),
            Q4=lambda x: x.quantile(1.0)
        ).interpolate(method='linear')
        distance_grouped[pt] = q_dist

    # Total for distance
    union_df['distance_bin'] = pd.cut(union_df['distance'], np.arange(0, union_df['distance'].max() + bin_width, bin_width), right=False)
    distance_grouped["total"] = union_df.groupby('distance_bin')['duration'].agg(
        Q1=lambda x: x.quantile(0.25),
        Q2=lambda x: x.quantile(0.5),
        Q3=lambda x: x.quantile(0.75),
        Q4=lambda x: x.quantile(1.0)
    ).interpolate(method='linear')

    return q_df_by_day, avg_df_by_district, time_grouped, distance_grouped


    load_df()
    # Convert and filter by date range
    df = df.copy()
    df['timestamp_date'] = pd.to_datetime(df['timestamp_date'], errors='coerce')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df.dropna(subset=['timestamp_date', 'duration', 'distance'], inplace=True)

    mask = (df['timestamp_date'] >= start_date) & (df['timestamp_date'] <= end_date)
    df = df.loc[mask]

    # --------------------------------
    # 1. Quartiles by day of week
    weekday_cols = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    q_by_day = pd.DataFrame(index=weekday_cols, columns=['Q1', 'Q2', 'Q3', 'Q4'])
    for day in weekday_cols:
        day_df = df[df[day] == 1]
        if not day_df.empty:
            q_by_day.loc[day] = day_df['duration'].quantile([0.25, 0.5, 0.75, 1.0]).values

    # --------------------------------
    # 2. Quartiles by problem type
    problem_cols = [col for col in df.columns if col.startswith("type_")]
    q_by_type = pd.DataFrame(index=problem_cols, columns=['Q1', 'Q2', 'Q3', 'Q4'])
    for col in problem_cols:
        p_df = df[df[col] == 1]
        if not p_df.empty:
            q_by_type.loc[col] = p_df['duration'].quantile([0.25, 0.5, 0.75, 1.0]).values

    # --------------------------------
    # 3. Mean by district
    district_cols = [
        'คลองสาน', 'คลองสามวา', 'คลองเตย', 'คันนายาว', 'จตุจักร', 'จอมทอง',
        'ดอนเมือง', 'ดินแดง', 'ดุสิต', 'ตลิ่งชัน', 'ทวีวัฒนา', 'ทุ่งครุ',
        'ธนบุรี', 'บางกอกน้อย', 'บางกอกใหญ่', 'บางกะปิ', 'บางขุนเทียน',
        'บางคอแหลม', 'บางซื่อ', 'บางนา', 'บางบอน', 'บางพลัด', 'บางรัก',
        'บางเขน', 'บางแค', 'บึงกุ่ม', 'ปทุมวัน', 'ประเวศ', 'ป้อมปราบศัตรูพ่าย',
        'พญาไท', 'พระนคร', 'พระโขนง', 'ภาษีเจริญ', 'มีนบุรี', 'ยานนาวา',
        'ราชเทวี', 'ราษฎร์บูรณะ', 'ลาดกระบัง', 'ลาดพร้าว', 'วังทองหลาง',
        'วัฒนา', 'สวนหลวง', 'สะพานสูง', 'สัมพันธวงศ์', 'สาทร', 'สายไหม',
        'หนองจอก', 'หนองแขม', 'หลักสี่', 'ห้วยขวาง'
    ]
    mean_by_district = pd.DataFrame(index=district_cols, columns=['Mean Duration'])
    for dist in district_cols:
        d_df = df[df[dist] == 1]
        if not d_df.empty:
            mean_by_district.loc[dist] = d_df['duration'].mean()

    # --------------------------------
    # 4. Quartiles over distance (binned)
    max_dist = df['distance'].max()
    bin_width = max_dist / 400
    bins = np.arange(0, max_dist + bin_width, bin_width)
    df['distance_bin'] = pd.cut(df['distance'], bins, right=False)

    q_by_distance = df.groupby('distance_bin')['duration'].agg(
        Q1=lambda x: x.quantile(0.25),
        Q2=lambda x: x.quantile(0.5),
        Q3=lambda x: x.quantile(0.75),
        Q4=lambda x: x.quantile(1.0)
    ).interpolate(method='linear')

    return q_by_day, q_by_type, mean_by_district, q_by_distance

def analyze_duration_in_period(df: pd.DataFrame, start_date: str, end_date: str):
    # Convert and filter by date range
    df = df.copy()
    df['timestamp_date'] = pd.to_datetime(df['timestamp_date'], errors='coerce')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df.dropna(subset=['timestamp_date', 'duration', 'distance'], inplace=True)

    mask = (df['timestamp_date'] >= start_date) & (df['timestamp_date'] <= end_date)
    df = df.loc[mask]

    # --------------------------------
    # 1. Quartiles by day of week
    weekday_cols = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    q_by_day = pd.DataFrame(index=weekday_cols, columns=['Q1', 'Q2', 'Q3', 'Q4'])
    for day in weekday_cols:
        day_df = df[df[day] == 1]
        if not day_df.empty:
            q_by_day.loc[day] = day_df['duration'].quantile([0.25, 0.5, 0.75, 1.0]).values

    # --------------------------------
    # 2. Quartiles by problem type + total
    problem_cols = ['PM2.5','การเดินทาง','กีดขวาง','คนจรจัด','คลอง','ความปลอดภัย','ความสะอาด','จราจร','ต้นไม้',
                'ถนน','ทางเท้า','ท่อระบายน้ำ','น้ำท่วม','ป้าย','ป้ายจราจร','ร้องเรียน','สอบถาม','สะพาน','สัตว์จรจัด',
                'สายไฟ','ห้องน้ำ','เสนอแนะ','เสียงรบกวน','แสงสว่าง']
    q_by_type = pd.DataFrame(index=problem_cols + ['total'], columns=['Q1', 'Q2', 'Q3', 'Q4'])
    for col in problem_cols:
        p_df = df[df[col] == 1]
        if not p_df.empty:
            q_by_type.loc[col] = p_df['duration'].quantile([0.25, 0.5, 0.75, 1.0]).values

    # Add total row
    if not df.empty:
        q_by_type.loc['total'] = df['duration'].quantile([0.25, 0.5, 0.75, 1.0]).values

    # --------------------------------
    # 3. Mean by district
    district_cols = [
        'คลองสาน', 'คลองสามวา', 'คลองเตย', 'คันนายาว', 'จตุจักร', 'จอมทอง',
        'ดอนเมือง', 'ดินแดง', 'ดุสิต', 'ตลิ่งชัน', 'ทวีวัฒนา', 'ทุ่งครุ',
        'ธนบุรี', 'บางกอกน้อย', 'บางกอกใหญ่', 'บางกะปิ', 'บางขุนเทียน',
        'บางคอแหลม', 'บางซื่อ', 'บางนา', 'บางบอน', 'บางพลัด', 'บางรัก',
        'บางเขน', 'บางแค', 'บึงกุ่ม', 'ปทุมวัน', 'ประเวศ', 'ป้อมปราบศัตรูพ่าย',
        'พญาไท', 'พระนคร', 'พระโขนง', 'ภาษีเจริญ', 'มีนบุรี', 'ยานนาวา',
        'ราชเทวี', 'ราษฎร์บูรณะ', 'ลาดกระบัง', 'ลาดพร้าว', 'วังทองหลาง',
        'วัฒนา', 'สวนหลวง', 'สะพานสูง', 'สัมพันธวงศ์', 'สาทร', 'สายไหม',
        'หนองจอก', 'หนองแขม', 'หลักสี่', 'ห้วยขวาง'
    ]
    mean_by_district = pd.DataFrame(index=district_cols, columns=['Mean Duration'])
    for dist in district_cols:
        d_df = df[df[dist] == 1]
        if not d_df.empty:
            mean_by_district.loc[dist] = d_df['duration'].mean()

    # --------------------------------
    # 4. Quartiles over distance (binned)
    max_dist = df['distance'].max()
    bin_width = max_dist / 400
    bins = np.arange(0, max_dist + bin_width, bin_width)
    df['distance_bin'] = pd.cut(df['distance'], bins, right=False)

    q_by_distance = df.groupby('distance_bin')['duration'].agg(
        Q1=lambda x: x.quantile(0.25),
        Q2=lambda x: x.quantile(0.5),
        Q3=lambda x: x.quantile(0.75),
        Q4=lambda x: x.quantile(1.0)
    ).interpolate(method='linear')

    return q_by_day, q_by_type, mean_by_district, q_by_distance

def analyze_by_districts(df, districts):
    # Filter rows that match any of the given districts
    district_mask = df[districts].sum(axis=1) > 0
    filtered_df = df[district_mask].copy()

    # Get list of problem types and weekdays
    problem_types = ['PM2.5','การเดินทาง','กีดขวาง','คนจรจัด','คลอง','ความปลอดภัย','ความสะอาด','จราจร','ต้นไม้',
                'ถนน','ทางเท้า','ท่อระบายน้ำ','น้ำท่วม','ป้าย','ป้ายจราจร','ร้องเรียน','สอบถาม','สะพาน','สัตว์จรจัด',
                'สายไฟ','ห้องน้ำ','เสนอแนะ','เสียงรบกวน','แสงสว่าง']
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    quantiles = [0.25, 0.5, 0.75, 1.0]  # Include Q4 (max)

    ### 1. Quartiles of duration per problem type ###
    rows = []
    for t in problem_types:
        subset = filtered_df[filtered_df[t] == 1]
        if not subset.empty:
            q = subset['duration'].quantile(quantiles)
            rows.append((t, *q.values))
    df_q1 = pd.DataFrame(rows, columns=['ProblemType', 'Q1', 'Q2', 'Q3', 'Q4'])
    total_q = filtered_df['duration'].quantile(quantiles)
    df_q1.loc[len(df_q1)] = ['total', *total_q.values]

    ### 2. Quartiles of duration per weekday ###
    rows = []
    for day in weekdays:
        subset = filtered_df[filtered_df[day] == 1]
        if not subset.empty:
            q = subset['duration'].quantile(quantiles)
            rows.append((day, *q.values))
    df_q2 = pd.DataFrame(rows, columns=['Weekday', 'Q1', 'Q2', 'Q3', 'Q4'])

    ### 3. Quartiles of duration over time (uniformly sampled) ###
    filtered_df['timestamp_date'] = pd.to_datetime(filtered_df['timestamp_date'])
    df_time = filtered_df[['timestamp_date', 'duration']].copy()
    df_time = df_time.sort_values('timestamp_date')

    time_min = df_time['timestamp_date'].min()
    time_max = df_time['timestamp_date'].max()
    time_range = pd.date_range(start=time_min, end=time_max, periods=401)
    time_points = np.linspace(0, (time_max - time_min).total_seconds(), 401)

    df_q3 = pd.DataFrame({'timestamp': time_range})
    for q in quantiles:
        grouped = df_time.groupby('timestamp_date')['duration'].quantile(q).reset_index()
        grouped['timestamp_date'] = pd.to_datetime(grouped['timestamp_date'])
        grouped['timestamp_num'] = (grouped['timestamp_date'] - time_min).dt.total_seconds()
        f = interp1d(grouped['timestamp_num'], grouped['duration'], kind='linear', fill_value='extrapolate')
        df_q3[f'Q{int(q*4)}'] = f(time_points)

    ### 4. Quartiles of duration over distance (uniformly sampled) ###
    dist_min, dist_max = filtered_df['distance'].min(), filtered_df['distance'].max()
    dist_points = np.linspace(dist_min, dist_max, 401)
    df_q4 = pd.DataFrame({'distance': dist_points})
    for q in quantiles:
        grouped = filtered_df.groupby('distance')['duration'].quantile(q).reset_index()
        f = interp1d(grouped['distance'], grouped['duration'], kind='linear', fill_value='extrapolate')
        df_q4[f'Q{int(q*4)}'] = f(dist_points)

    return df_q1, df_q2, df_q3, df_q4