import asyncio, json, io, time, requests
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from confluent_kafka import Producer
from fastavro import parse_schema, schemaless_writer

# ========== CONFIG ==========
KAFKA_TOPIC = "traffy_reports"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
interval = 60
report_per_scan = 6
page_scrolled_amount = 500
html_path = "./realtime_scraping/tmp/traffy.html"
report_path = "./realtime_scraping/tmp/report.html"
API_key = "dad93076eb9a905a9122a00806c70616"
# =============================

# ========== Updated Avro Schema ==========
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

# ========== Kafka Producer ==========
producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})

def send_report_avro(report):
    buf = io.BytesIO()
    schemaless_writer(buf, parsed_schema, report)
    producer.produce(KAFKA_TOPIC, key=report["ticket_id"], value=buf.getvalue())
    producer.flush()

# ========== Address Enhancer ==========
def extract_address_from_div_map(html):
    soup = BeautifulSoup(html, "html.parser")
    div_map = soup.find("div", class_="div-map")
    if not div_map:
        return None
    target_div = div_map.find("div", style=lambda s: s and "gap: 0.2rem" in s and "margin-top: 0.5rem" in s)
    if target_div:
        p = target_div.find("p")
        return p.get_text(strip=True) if p else None
    return None

def get_coords(address):
    encoded_address = address.replace(" ", "%20")
    url = f"https://search.longdo.com/addresslookup/api/addr/geocoding?text={encoded_address}&key={API_key}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    try:
        point = data["data"][0]["point"][0]
        return point["lat"], point["lon"]
    except (KeyError, IndexError, TypeError):
        return None

# ========== Fetch & Enhance ==========
async def fetch_traffy_html():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://bangkok.traffy.in.th/", wait_until="networkidle")
        for _ in range(20):
            await page.mouse.wheel(0, page_scrolled_amount)
            await asyncio.sleep(1.5)
        html = await page.content()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        await browser.close()

async def fetch_detail_and_get_latlon(ticket_id):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(f'https://bangkok.traffy.in.th/detail?ticketID={ticket_id}&i=undefined', wait_until="networkidle")
        for _ in range(10):
            await page.mouse.wheel(0, 100)
            await asyncio.sleep(1.0)
        html = await page.content()
        await browser.close()

        address = extract_address_from_div_map(html)
        coords = get_coords(address) if address else (None, None)
        return coords + (address,) if coords else (None, None, address)


# ========== Main Extract & Send ==========
async def extract_and_send():
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    containers = soup.find_all("div", class_="containerData")[:report_per_scan]

    for container in containers:
        report = {}
        get = lambda c, cls: c.find("p", class_=cls)
        get_text = lambda tag: tag.get_text(strip=True) if tag else ""

        report["ticket_id"] = get(container, "ticket_id").get_text(strip=True)
        times = container.find_all("p", class_="detailTimes")
        report["report_time"] = times[1].get_text(strip=True) if len(times) > 1 else ""
        full_address = get(container, "address").get_text(strip=True)
        parts = full_address.split()
        report["district"] = parts[0] if len(parts) > 0 else ""
        report["subdistrict"] = parts[1] if len(parts) > 1 else ""
        report["status"] = get(container, "title-state").get_text(strip=True)
        report["description"] = container.find("span", class_="description").get_text(strip=True)
        resolution = container.find("p", class_="detailReportPost description")
        report["resolution"] = resolution.find("span").get_text(strip=True) if resolution and resolution.find("span") else (resolution.get_text(strip=True) if resolution else "")
        by = container.find("span", string="โดย:")
        agency = by.find_next_sibling("span") if by else None
        report["reporting_agency"] = agency.get_text(strip=True) if agency else ""
        tags = container.find_all("div", class_="tags-problemType")
        report["tags"] = ', '.join(tag.get_text(strip=True) for tag in tags)
        upvote = container.find("p", class_="countPoint")
        report["upvotes"] = int(upvote.get_text(strip=True)) if upvote and upvote.get_text(strip=True).isdigit() else 0
        img = container.find("img", class_="img_post")
        report["image_url"] = img["src"] if img else ""

        # Add latitude and longitude from detail page
        report["latitude"], report["longitude"], report["address"] = await fetch_detail_and_get_latlon(report["ticket_id"])

        send_report_avro(report)

# ========== Scheduler ==========
async def main():
    await fetch_traffy_html()
    await extract_and_send()

# Run on interval
while True:
    asyncio.run(main())
    print("✅ Sent reports to Kafka.")
    time.sleep(interval)
