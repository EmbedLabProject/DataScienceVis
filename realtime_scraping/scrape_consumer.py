from confluent_kafka import Consumer
from fastavro import parse_schema, schemaless_reader
import io, json, pandas as pd
import time



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
csv_path = "./out/traffy_reports_latest.csv"

# set the amount to the same as producer
report_per_scan = 20
###################################################################




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
            df = pd.DataFrame(batch)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ {len(batch)} reports saved to {csv_path} (overwritten).")
        else:
            print("⚠️ No new reports received this cycle.")

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
finally:
    consumer.close()
