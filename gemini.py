# gemini.py
import os
import re
import io
import json
import datetime
import pandas as pd
import shutil
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from dateutil.parser import parse

load_dotenv()
DEFAULT_INPUT_FOLDER = 'input_folder'
OUTPUT_CSV_FILENAME = 'event_data.csv'
CACHE_FILE = 'processed_images.log'
ALLOWED_EXTENSIONS = ('.png', '.jpeg', '.jpg')
CSV_HEADERS = ['event', 'festivalName', 'imageURL', 'raceVideo', 'type', 'date', 'city', 'organiser','participationType', 'firstEdition', 'lastEdition', 'countEditions', 'mode', 'raceAccredition','theme', 'numberOfparticipants', 'startTime', 'scenic', 'registrationCost', 'ageLimitation','eventWebsite', 'organiserWebsite', 'bookingLink', 'newsCoverage', 'lastDate','participationCriteria', 'refundPolicy', 'swimDistance', 'swimType', 'swimmingLocation','waterTemperature', 'swimCoursetype', 'swimCutoff', 'swimRoutemap', 'cyclingDistance','cyclingElevation', 'cyclingSurface', 'cyclingElevationgain', 'cycleCoursetype', 'cycleCutoff','cyclingRoutemap', 'runningDistance', 'runningElevation', 'runningSurface', 'runningElevationgain','runningElevationloss', 'runningCoursetype', 'runCutoff', 'runRoutemap', 'organiserRating','triathlonType', 'standardTag', 'region', 'approvalStatus', 'difficultyLevel', 'month','primaryKey', 'latitude', 'longitude', 'country', 'editionYear', 'aidStations','restrictedTraffic', 'user_id', 'femaleParticpation', 'jellyFishRelated','registrationOpentag', 'eventConcludedtag', 'state', 'nextEdition']
CHOICE_FIELDS = {"participationType": ["Individual", "Relay", "Group"], "mode": ["Virtual", "On-Ground"], "runningSurface": ["Road", "Trail", "Track", "Road + Trail"], "runningCourseType": ["Single Loop", "Multiple Loop", "Out and Back", "Point to Point"], "region": ["West India", "Central and East India", "North India", "South India", "Nepal", "Bhutan", "Sri Lanka"], "runningElevation": ["Flat", "Rolling", "Hilly", "Skyrunning"], "type": ["Triathlon", "Aquabike", "Aquathlon", "Duathlon", "Run", "Cycling", "Swimathon"], "swimType": ["Lake", "Beach", "River", "Pool"], "swimCoursetype": ["Single Loop", "Multiple Loops", "Out and Back", "Point to Point"], "cyclingElevation": ["Flat", "Rolling", "Hilly"], "cycleCoursetype": ["Single Loop", "Multiple Loops", "Out and Back", "Point to Point"], "triathlonType": ["Super Sprint", "Sprint Distance", "Olympic Distance", "Half Iron(70.3)", "Iron Distance (140.6)","Ultra Distance"], "standardTag": ["Standard", "Non Standard"], "restrictedTraffic": ["Yes", "No"], "jellyFishRelated": ["Yes", "No"], "approvalStatus": ["Approved", "Pending Approval"]}

def clean_value(value):
    if value is None or str(value).strip().upper() in ["NA", "N/A", "NONE", "NOT SPECIFIED", ""]: return ""
    return str(value).encode('utf-8', 'ignore').decode('utf-8').strip()
def validate_choice(value, options):
    cleaned_value = clean_value(value)
    if not cleaned_value: return ""
    for option in options:
        if cleaned_value.lower() == option.lower(): return option
    return ""
def format_date_value(date_str):
    cleaned_str = clean_value(date_str)
    if not cleaned_str: return ""
    try:
        dt = parse(cleaned_str, fuzzy=True, dayfirst=True)
        if dt and dt.year >= 2025: return dt.strftime("%d/%m/%Y")
        return ""
    except (ValueError, TypeError): return ""
def format_time_value(time_str):
    cleaned_str = clean_value(time_str)
    if not cleaned_str: return ""
    try:
        match = re.search(r'(\d{1,2})[:.]?(\d{2})?\s*(am|pm)?', cleaned_str, re.IGNORECASE)
        if not match: return ""
        hour, minute, am_pm = match.groups()
        hour = int(hour)
        minute = int(minute) if minute else 0
        if am_pm: am_pm = am_pm.upper()
        else: am_pm = "AM" if 5 <= hour < 12 else "PM"
        if am_pm == "PM" and hour < 12: hour += 12
        if am_pm == "AM" and hour == 12: hour = 0
        dt = datetime.time(hour, minute)
        return dt.strftime("%I:%M %p")
    except Exception: return ""
def extract_numeric(value_str):
    cleaned_str = clean_value(str(value_str))
    if not cleaned_str: return ""
    match = re.search(r'(\d+\.?\d*|\.\d+)', cleaned_str)
    return match.group(1) if match else ""
def extract_registration_cost(value_str):
    cleaned_str = clean_value(value_str)
    if not cleaned_str: return ""
    if "free" in cleaned_str.lower(): return "0"
    cleaned_str = cleaned_str.replace(',', '')
    match = re.search(r'(\d+)', cleaned_str)
    return match.group(1) if match else ""
def extract_age_limit(value_str):
    cleaned_str = clean_value(value_str)
    if not cleaned_str: return ""
    match = re.search(r'(\d+\+?)', cleaned_str)
    return match.group(1) if match else ""

def get_gemini_response(image_path: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise ValueError("GEMINI_API_KEY not found in environment.")
    try:
        genai.configure(api_key=api_key)
        with Image.open(image_path) as img:
            img_byte_arr = io.BytesIO()
            if img.mode == 'RGBA': img = img.convert('RGB')
            img.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = """
        You are a highly intelligent data extraction assistant... [Your detailed Gemini prompt goes here]
        """
        response = model.generate_content([prompt, image_part], request_options={"timeout": 120})
        clean_response_text = re.sub(r'^```json\s*|\s*```$', '', response.text.strip(), flags=re.MULTILINE)
        return json.loads(clean_response_text)
    except Exception as e:
        print(f"  -> Error calling Gemini API for {os.path.basename(image_path)}: {e}")
        return {}

def process_image_data(raw_data: dict) -> dict:
    processed_row = {header: "" for header in CSV_HEADERS}
    # ... [Data processing logic is unchanged and omitted for brevity] ...
    return processed_row

def main(output_dir_override=None, input_dir_override=None):
    print("--- Event Data Extraction Script ---")
    effective_input_dir = input_dir_override if input_dir_override else DEFAULT_INPUT_FOLDER
    print(f"INFO: Reading images from: {os.path.abspath(effective_input_dir)}")
    if output_dir_override and not os.path.exists(output_dir_override):
        os.makedirs(output_dir_override)
    final_output_path = os.path.join(output_dir_override, OUTPUT_CSV_FILENAME) if output_dir_override else OUTPUT_CSV_FILENAME
    print(f"INFO: Output file will be saved to: {os.path.abspath(final_output_path)}")
    if not os.path.exists(effective_input_dir):
        os.makedirs(effective_input_dir)
        print(f"Created '{effective_input_dir}'. Please add images and run again.")
        return
    try:
        with open(CACHE_FILE, 'r') as f: processed_images = set(f.read().splitlines())
    except FileNotFoundError: processed_images = set()
    all_images = [f for f in os.listdir(effective_input_dir) if f.lower().endswith(ALLOWED_EXTENSIONS)]
    new_images = [f for f in all_images if f not in processed_images]
    if not new_images:
        print("No new images to process. Exiting.")
        return
    print(f"Found {len(new_images)} new image(s) to process.")
    new_data_rows = []
    processed_this_run = []
    for image_name in new_images:
        image_path = os.path.join(effective_input_dir, image_name)
        print(f"\nProcessing '{image_name}'...")
        raw_data = get_gemini_response(image_path)
        if not raw_data:
            print(f"  -> Skipping {image_name} due to API error.")
            continue
        processed_row = process_image_data(raw_data)
        new_data_rows.append(processed_row)
        processed_this_run.append(image_name)
        print(f"  -> Successfully extracted data for '{image_name}'.")
    if not new_data_rows:
        print("\nNo data was successfully extracted. Exiting.")
        return
    df = pd.DataFrame(new_data_rows)
    df = df[CSV_HEADERS]
    if os.path.exists(final_output_path):
        df.to_csv(final_output_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df.to_csv(final_output_path, mode='w', header=True, index=False, encoding='utf-8')
    print(f"Data successfully saved to '{final_output_path}'.")
    with open(CACHE_FILE, 'a') as f:
        for image_name in processed_this_run: f.write(f"{image_name}\n")
    print("\n--- Script Finished ---")

# CRITICAL FIX: This guard prevents the script from running on import
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Crawl4AI Gemini Image Processor")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory to save output files.")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_FOLDER, help="Directory containing input images.")
    args = parser.parse_args()
    main(output_dir_override=args.output_dir, input_dir_override=args.input_dir)