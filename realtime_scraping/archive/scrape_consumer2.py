from confluent_kafka import Consumer
from fastavro import parse_schema, schemaless_reader
import io
import os
import time
import pandas as pd
from datetime import datetime

# ---------------- SCHEMA ----------------
schema = {
    "type": "record",
    "name": "TraffyReport",
    "fields": [
        {"name": "ticket_id", "type": "string"},
        {"name": "report_time", "type": "string"},
        {"name": "address", "type": "string"},
        {"name": "district", "type": "string"},
        {"name": "subdistrict", "type": "string"},
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

# ---------------- CONSUMER SETUP ----------------
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'traffy-avro-clean-v1',  # <-- Use a fresh group ID to re-read messages
    'auto.offset.reset': 'earliest'
})

consumer.subscribe(["traffy_reports"])  # Update topic name if needed
csv_path = "./realtime_scraping/out/traffy_reports_stream.csv"

print("🟢 Kafka consumer started. Listening for Avro messages...")

# ---------------- CONSUMPTION LOOP ----------------
try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            print("❌ Kafka Error:", msg.error())
            continue

        val = msg.value()
        if not val:
            print("⚠️ Empty message received. Skipping.")
            continue

        print(f"📦 Received raw bytes: {val[:10]}...")

        buf = io.BytesIO(val)
        try:
            report = schemaless_reader(buf, parsed_schema)
        except Exception as e:
            print(f"❌ Failed to decode Avro message: {e}")
            continue

        # Add additional metadata
        report["source"] = "kafka"
        report["received_time"] = datetime.utcnow().isoformat()

        # Append to CSV
        df = pd.DataFrame([report])
        file_exists = os.path.isfile(csv_path)
        df.to_csv(csv_path, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')

        print(f"✅ Appended ticket_id: {report.get('ticket_id')}")

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
finally:
    consumer.close()
