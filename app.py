# AI Contacts Search Assistant
# Copyright (C) 2023 Alex Furmansky, Magnetic Ventures LLC
#
# This file is part of the AI Contacts Search Assistant.
#
# AI Contacts Search Assistant is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AI Contacts Search Assistant is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with AI Contacts Search Assistant. If not, see <https://www.gnu.org/licenses/>.

from flask import Flask, render_template, request, redirect, flash, url_for
import os
import time
import re
import requests
import logging
import datetime

from dotenv import load_dotenv
import csv
import pandas as pd

from db_handler import insert_or_update_contact, fetch_contact_by_linkedin, fetch_all_contacts, init_db
from ai_actions import reformat_experiences


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')
proxycurl_api_key = os.getenv('PROXYCURL_API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the database
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file provided')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        logging.info(f"File saved to {filepath}")
        
        # Process the CSV file
        df, valid = process_csv(filepath)
        if not valid:
            flash('CSV file is missing required fields')
            return redirect(request.url)
        
        # Fetch and process detailed information from Proxycurl
        process_proxycurl_data(df)
        
        return render_template('confirmation.html', data=fetch_all_contacts())
    else:
        flash('Invalid file type')
        return redirect(request.url)

@app.route('/confirmation')
def confirmation():
    data = fetch_all_contacts()
    print(data)  # Debugging statement to verify the data
    return render_template('confirmation.html', data=data)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def process_csv(filepath):
    required_fields = ['First Name', 'Last Name', 'URL', 'Email Address', 'Company', 'Position']
    
    # Read the CSV file
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        
        # Identify the header row
        while not set(required_fields).issubset(set(header)):
            header = next(reader)
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in header]
        if missing_fields:
            logging.warning(f"The following required fields are missing from the CSV: {', '.join(missing_fields)}")
            return None, False
        
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(filepath, skiprows=reader.line_num - 1)
        logging.info(f"CSV file loaded into DataFrame with {len(df)} rows")
        
        # Filter out rows with empty URL fields
        df = df.dropna(subset=['URL'])
        df = df[df['URL'].str.strip() != '']
        logging.info(f"DataFrame filtered to {len(df)} rows with non-empty URLs")
        
        return df, True

# Function to fetch data from Proxycurl and store it in the database
def fetch_data_from_proxycurl(linkedin_url, full_name, max_retries=1, delay=5):
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    headers = {'Authorization': f'Bearer {proxycurl_api_key}'}
    params = {'linkedin_profile_url': linkedin_url}
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1} - Fetching data from Proxycurl API for URL: {linkedin_url}")
            response = requests.get(api_endpoint, params=params, headers=headers, timeout=30)
            
            response.raise_for_status()
            return response.json(), None
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.warning("Max retries reached. Falling back to OpenAI mock data.")
                return None, "Failed to fetch data"
    
    logging.error("Failed to fetch data from Proxycurl and OpenAI fallback.")
    return None, "Failed to fetch data"

def process_proxycurl_data(df):
    
    
    for index, row in df.iterrows():
        linkedin_url = row['URL']
        full_name = row['First Name'] + ' ' + row['Last Name']
        
        if pd.isna(linkedin_url) or not linkedin_url:
            logging.warning(f"Skipping row {index} due to invalid LinkedIn URL")
            
            continue
        
        existing_contact, experiences, education = fetch_contact_by_linkedin(linkedin_url, full_name)
        if existing_contact:
            logging.info(f"Using stored data for URL: {linkedin_url}")
            
        else:
            data, error = fetch_data_from_proxycurl(linkedin_url, full_name)
            if data:
                formatted_data = format_data(row, data, linkedin_url)
                insert_or_update_contact(formatted_data)  # Store data in SQLite
                
            else:
                logging.warning(f"Failed to fetch data for URL {linkedin_url}: {error}")
                
    
    return "success"

def format_data(row, data, linkedin_url):
    full_name = clean_string(data.get('full_name', ''))
    occupation = clean_string(data.get('occupation', ''))
    headline = clean_string(data.get('headline', ''))
    summary = clean_string(data.get('summary', ''))
    location = clean_string(f"{data.get('city', '')}, {data.get('state', '')}, {data.get('country', '')}")
    
    experiences = sorted(data.get('experiences', []), key=lambda x: (x['ends_at'] is not None, (x['ends_at']['year'], x['ends_at']['month'], x['ends_at']['day']) if x['ends_at'] else ()), reverse=True)
    experience_list = [
        {
            "Company": clean_string(exp['company']),
            "Title": clean_string(exp['title']),
            "Description": clean_string(exp['description']),
            "Start Date": format_date(exp['starts_at']),
            "End Date": format_date(exp['ends_at']) if exp['ends_at'] else None
        }
        for exp in experiences
    ]
    
    education_list = [
        {
            "School": clean_string(edu['school']),
            "Field of Study": clean_string(edu.get('field_of_study', '')),
            "Degree": clean_string(edu.get('degree_name', '')),
            "Description": clean_string(edu.get('description', '')),
            "Start Date": format_date(edu.get('starts_at')),
            "End Date": format_date(edu.get('ends_at'))
        }
        for edu in data.get('education', [])
    ]
    
    profile_pic_url = clean_string(data.get('profile_pic_url', ''))
    
    ai_summary = reformat_experiences(experience_list)
    combined_summary = clean_string(f"{summary}\n\n{ai_summary}")
    
    return {
        'full_name': full_name,
        'occupation': occupation,
        'headline': headline,
        'summary': combined_summary,
        'location': location,
        'experience_list': experience_list,
        'education_list': education_list,
        'profile_pic_url': profile_pic_url,
        'linkedin_url': linkedin_url  # Add LinkedIn URL for generating unique ID
    }

def format_date(date_dict):
    if not date_dict:
        return None
    try:
        return datetime.date(date_dict['year'], date_dict['month'], 1).strftime('%Y-%m-%d')
    except (KeyError, ValueError):
        return None

def default_contact_data(row):
    return {
        'full_name': clean_string(row['First Name'] + ' ' + row['Last Name']),
        'occupation': 'N/A',
        'headline': 'N/A',
        'summary': 'N/A',
        'location': 'N/A',
        'experience_list': [],
        'education_list': [],
        'profile_pic_url': '',
        'linkedin_url': clean_string(row['URL'])
    }

def clean_string(input_string):
    if input_string is None:
        return ''
    # Remove any special characters except for basic punctuation and spaces
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', input_string)
    return cleaned_string
    


if __name__ == '__main__':
    app.run(debug=True)