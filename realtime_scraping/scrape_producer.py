import asyncio, json
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from confluent_kafka import Producer
from fastavro import parse_schema, schemaless_writer
import io
import time

# Avro schema load
import os
import pandas as pd

################## config here #######################################

# current path used is the relative path from this folder route
# open schema file, change path to your syatem path
with open("./traffy_report.avsc", "r") as f:
    schema = json.load(f)
parsed_schema = parse_schema(schema)

# path for html to saved
html_path = "./tmp/traffy.html"

# path for detail reports to be saved
report_path = "./tmp/report.html"

# topic, match the consumer
KAFKA_TOPIC = "traffy_reports"

# broker, replace if using external
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

# time between each scan in second, default is 300 s
interval = 60

# must match the consumer
# the report_per_scan will be buggy if the page_scrolled_amount is too low
# currently 100 page_scroll allow 20-30 reports per scan
# but increase page scrolled amount too much will result in slow process
report_per_scan = 20
page_scrolled_amount = 500

########################################################################

# producer setup
producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})


# parrarelizer + sender 
def send_report_avro(report):
    buf = io.BytesIO()
    schemaless_writer(buf, parsed_schema, report)
    producer.produce(KAFKA_TOPIC, key=report["ticket_id"], value=buf.getvalue())
    producer.flush()



# HTML getter (simulate scroll)
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

async def fetch_detail_traffy_report(ticket_id):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(f'https://bangkok.traffy.in.th/detail?ticketID={ticket_id}&i=undefined', wait_until="networkidle")
            for _ in range(20):
                await page.mouse.wheel(0, 100)
                await asyncio.sleep(1.5)
            html = await page.content()
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html)
            await browser.close()

#async def extract_detail_traffy_report():
    

# Scrape from HTML and sent
def extract_and_send():
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    for container in soup.find_all("div", class_="containerData")[:report_per_scan]:
        report = {}
        get = lambda c, cls: c.find("p", class_=cls)
        get_text = lambda tag: tag.get_text(strip=True) if tag else ""

        report["ticket_id"] = get(container, "ticket_id").get_text(strip=True)
        times = container.find_all("p", class_="detailTimes")
        report["report_time"] = times[1].get_text(strip=True) if len(times) > 1 else ""
        report["address"] = get(container, "address").get_text(strip=True)
        report["status"] = get(container, "title-state").get_text(strip=True)
        report["description"] = container.find("span", class_="description").get_text(strip=True)
        resolution = container.find("p", class_="detailReportPost description")
        if resolution:
            span = resolution.find("span")
            report["resolution"] = span.get_text(strip=True) if span else resolution.get_text(strip=True)
        else:
            report["resolution"] = ""
        by = container.find("span", string="โดย:")
        agency = by.find_next_sibling("span") if by else None
        report["reporting_agency"] = agency.get_text(strip=True) if agency else ""
        tags = container.find_all("div", class_="tags-problemType")
        report["tags"] = ', '.join(tag.get_text(strip=True) for tag in tags)
        upvote = container.find("p", class_="countPoint")
        report["upvotes"] = int(upvote.get_text(strip=True)) if upvote and upvote.get_text(strip=True).isdigit() else 0
        img = container.find("img", class_="img_post")
        report["image_url"] = img["src"] if img else ""

        send_report_avro(report)

async def main():
    await fetch_traffy_html()
    extract_and_send()

# Loop every interval minutes
while True:
    asyncio.run(main())
    print("✅ Sent reports to Kafka.")
    time.sleep(interval)
