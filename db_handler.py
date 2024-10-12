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

import sqlite3
import datetime
import hashlib
import re
import logging
from vector_handler import chunk_text, generate_embeddings, EmbeddingStore
from typing import List, Dict, Any

# Initialize the embedding store
embedding_store = EmbeddingStore(dimension=1536)  # Adjust dimension based on the model used



# Database initialization
def init_db(db_name='contacts.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Enable foreign key support
    cursor.execute('PRAGMA foreign_keys = ON')
    
    # Create contacts table with TEXT id
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS contacts (
        id TEXT PRIMARY KEY,
        full_name TEXT,
        occupation TEXT,
        headline TEXT,
        summary TEXT,
        location TEXT,
        profile_pic_url TEXT,
        linkedin_url TEXT
    )
    ''')
    
    # Create experiences table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        contact_id TEXT,
        company TEXT,
        title TEXT,
        description TEXT,
        start_date DATE,
        end_date DATE,
        FOREIGN KEY (contact_id) REFERENCES contacts(id)
    )
    ''')
    
    # Create education table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS education (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        contact_id TEXT,
        school TEXT,
        field_of_study TEXT,
        degree TEXT,
        description TEXT,
        start_date DATE,
        end_date DATE,
        FOREIGN KEY (contact_id) REFERENCES contacts(id)
    )
    ''')
    
    # Create FTS tables
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS contacts_fts USING fts5(
        id, full_name, occupation, headline, summary, location, profile_pic_url, linkedin_url
        
    )
    ''')
    
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts USING fts5(
        contact_id, company, title, description, start_date, end_date
        
    )
    ''')
    
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS education_fts USING fts5(
        contact_id, school, field_of_study, degree, description, start_date, end_date
        
    )
    ''')
    
    # Create user_queries table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_text TEXT NOT NULL,
        query_criteria TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create result_sets table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS result_sets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id INTEGER,
        contact_id TEXT,
        relevancy_score TEXT CHECK(relevancy_score IN ('Low', 'Medium', 'High')),
        reasoning TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (query_id) REFERENCES user_queries(id),
        FOREIGN KEY (contact_id) REFERENCES contacts(id),
        UNIQUE(query_id, contact_id)
    )
    ''')
    
    # Create indexes for frequently queried fields
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_linkedin_url ON contacts(linkedin_url)')
    
    conn.commit()
    conn.close()

# Generate unique ID from the LinkedIn URL. add the first name to the hash to make it easier to read
def generate_unique_id(full_name, linkedin_url):
       
    # get first name from full name
    first_name = full_name.split(' ')[0]

    return first_name + hashlib.md5(linkedin_url.encode()).hexdigest()

# Insert or update a contact in the database
def insert_or_update_contact(contact_data, db_name='contacts.db'):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        contact_id = generate_unique_id(contact_data['full_name'], contact_data['linkedin_url'])
        
        # Log the data being inserted
        logging.info(f"Inserting data for contact: {contact_data['full_name']}")
        logging.info(f"Contact data: {contact_data}")
        
        # Insert or update the contact in the main table
        try:
            cursor.execute('''
                INSERT INTO contacts (id, full_name, occupation, headline, summary, location, profile_pic_url, linkedin_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    full_name = excluded.full_name,
                    occupation = excluded.occupation,
                    headline = excluded.headline,
                    summary = excluded.summary,
                    location = excluded.location,
                    profile_pic_url = excluded.profile_pic_url,
                    linkedin_url = excluded.linkedin_url
            ''', (contact_id, contact_data['full_name'], contact_data['occupation'], contact_data['headline'],
                  contact_data['summary'], contact_data['location'], contact_data['profile_pic_url'],
                  contact_data['linkedin_url']))
            logging.info(f"Successfully inserted/updated contact: {contact_id}")
        except sqlite3.Error as e:
            logging.error(f"Error inserting/updating contact: {contact_id}")
            logging.error(f"Error details: {e}")
            logging.error(f"Contact data causing error: {contact_data}")
            raise  # Re-raise the exception to be caught by the outer try-except block
        
        # Delete the existing record in the FTS table if it exists
        cursor.execute('DELETE FROM contacts_fts WHERE id = ?', (contact_id,))
        
        # Insert the new record into the FTS table
        cursor.execute('''
            INSERT INTO contacts_fts (id, full_name, occupation, headline, summary, location, profile_pic_url, linkedin_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (contact_id, contact_data['full_name'], contact_data['occupation'], contact_data['headline'],
              contact_data['summary'], contact_data['location'], contact_data['profile_pic_url'],
              contact_data['linkedin_url']))
        
        # Clear previous experiences and education before re-inserting
        cursor.execute('DELETE FROM experiences WHERE contact_id = ?', (contact_id,))
        cursor.execute('DELETE FROM education WHERE contact_id = ?', (contact_id,))
        cursor.execute('DELETE FROM experiences_fts WHERE contact_id = ?', (contact_id,))
        cursor.execute('DELETE FROM education_fts WHERE contact_id = ?', (contact_id,))
        
        # Insert each experience into the experiences table and FTS table
        for exp in contact_data['experience_list']:
            try:
                cursor.execute('''
                    INSERT INTO experiences (contact_id, company, title, description, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (contact_id, exp['Company'], exp['Title'], exp['Description'], exp['Start Date'], exp['End Date']))
                logging.info(f"Successfully inserted experience for contact {contact_id}: {exp}")
                
                cursor.execute('''
                    INSERT INTO experiences_fts (contact_id, company, title, description, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (contact_id, exp['Company'], exp['Title'], exp['Description'], exp['Start Date'], exp['End Date']))
                logging.info(f"Successfully inserted experience into FTS for contact {contact_id}: {exp}")
            except sqlite3.Error as e:
                logging.error(f"Error inserting experience for contact {contact_id}: {exp}")
                logging.error(f"Error details: {e}")
        
        # Insert each education record into the education table and FTS table
        for edu in contact_data['education_list']:
            try:
                cursor.execute('''
                    INSERT INTO education (contact_id, school, field_of_study, degree, description, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (contact_id, edu['School'], edu['Field of Study'], edu['Degree'], edu['Description'], edu['Start Date'], edu['End Date']))
                logging.info(f"Successfully inserted education for contact {contact_id}: {edu}")
                
                cursor.execute('''
                    INSERT INTO education_fts (contact_id, school, field_of_study, degree, description, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (contact_id, edu['School'], edu['Field of Study'], edu['Degree'], edu['Description'], edu['Start Date'], edu['End Date']))
                logging.info(f"Successfully inserted education into FTS for contact {contact_id}: {edu}")
            except sqlite3.Error as e:
                logging.error(f"Error inserting education for contact {contact_id}: {edu}")
                logging.error(f"Error details: {e}")
        
        # Delete existing embeddings for the contact's summary
        embedding_store.delete_embeddings_by_metadata_prefix(f"{contact_id}_summary_")

        # Generate and store embeddings for the summary field
        summary_chunks = chunk_text(contact_data['summary'])
        summary_embeddings = generate_embeddings(summary_chunks)
        metadata = [f"{contact_id}_summary_{i}" for i in range(len(summary_chunks))]
        embedding_store.add_embeddings(summary_embeddings, metadata, summary_chunks)

        conn.commit()
        logging.info(f"Successfully committed all changes for contact: {contact_id}")
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        logging.error(f"Error occurred with data: {contact_data}")
        conn.rollback()
        logging.info("Changes rolled back due to error")
    finally:
        conn.close()
        logging.info(f"Database connection closed for contact: {contact_id}")

def fetch_all_contacts(db_name='contacts.db'):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM contacts')
        records = cursor.fetchall()
        
        # Log the raw records fetched from the database
        logging.debug(f"Raw records fetched from database: {records}")
        
        # Convert the records to a list of dictionaries
        contacts = []
        for record in records:
            contact = {
                'id': record[0],
                'full_name': record[1],
                'occupation': record[2],
                'headline': record[3],
                'summary': record[4],
                'location': record[5],
                'profile_pic_url': record[6],
                'linkedin_url': record[7],
                'experience_list': fetch_experiences(record[0], cursor),
                'education_list': fetch_education(record[0], cursor)
            }
            contacts.append(contact)
        
        # Log the processed contacts data
        logging.debug(f"Processed contacts data: {contacts}")
        
        return contacts
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return []
    finally:
        conn.close()



def fetch_experiences(contact_id, cursor):
    cursor.execute('SELECT company, title, description, start_date, end_date FROM experiences WHERE contact_id = ?', (contact_id,))
    experiences = cursor.fetchall()
    return [
        {
            'Company': exp[0],
            'Title': exp[1],
            'Description': exp[2],
            'Start Date': format_date_for_display(exp[3]),
            'End Date': format_date_for_display(exp[4])
        }
        for exp in experiences
    ]

def fetch_education(contact_id, cursor):
    cursor.execute('SELECT school, field_of_study, degree, description, start_date, end_date FROM education WHERE contact_id = ?', (contact_id,))
    education = cursor.fetchall()
    return [
        {
            'School': edu[0],
            'Field of Study': edu[1],
            'Degree': edu[2],
            'Description': edu[3],
            'Start Date': format_date_for_display(edu[4]),
            'End Date': format_date_for_display(edu[5])
        }
        for edu in education
    ]

def format_date_for_display(date_str):
    if date_str is None:
        return None
    try:
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%m/%d/%Y')
    except ValueError:
        return date_str  # Return as-is if it's not in the expected format

def fetch_education(contact_id, cursor):
    cursor.execute('SELECT school, field_of_study, degree, description FROM education WHERE contact_id = ?', (contact_id,))
    education = cursor.fetchall()
    return [
        {
            'School': edu[0],
            'Field of Study': edu[1],
            'Degree': edu[2],
            'Description': edu[3]
        }
        for edu in education
    ]

# Fetch a contact by LinkedIn URL (unique ID is based on LinkedIn URL)
def fetch_contact_by_linkedin(linkedin_url, full_name, db_name='contacts.db'):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        contact_id = generate_unique_id(full_name, linkedin_url)
        cursor.execute('SELECT * FROM contacts WHERE id = ?', (contact_id,))
        contact_record = cursor.fetchone()
        
        # Fetch experiences
        cursor.execute('SELECT * FROM experiences WHERE contact_id = ?', (contact_id,))
        experience_records = cursor.fetchall()
        
        # Fetch education
        cursor.execute('SELECT * FROM education WHERE contact_id = ?', (contact_id,))
        education_records = cursor.fetchall()
        
        return contact_record, experience_records, education_records
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return None, [], []
    finally:
        conn.close()

def validate_query(query: str) -> bool:
    """
    Validate that the query is a SELECT statement and does not contain any modifying keywords.

    Args:
        query (str): The SQL query string.

    Returns:
        bool: True if the query is valid, False otherwise.
    """
    # Check if the query starts with SELECT (case-insensitive)
    if not re.match(r'^\s*SELECT', query, re.IGNORECASE):
        return False
    
    # Disallow modifying keywords (case-insensitive)
    disallowed_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'REPLACE']
    for keyword in disallowed_keywords:
        if re.search(r'\b' + keyword + r'\b', query, re.IGNORECASE):
            return False
    
    return True


def general_contacts_search(query: str, db_name='contacts.db') -> List[Dict[str, Any]]:
    """
    Execute a general search query on the contacts database.

    Args:
        query (str): The SQL query string.
        db_name (str): The name of the database.

    Returns:
        List[Dict[str, Any]]: The query results.
    """
    if not validate_query(query):
        raise ValueError("Invalid query. Only SELECT statements are allowed.")
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return []
    finally:
        conn.close()



def create_or_update_results(query_id: int, contact_ids: List[int], db_name='contacts.db'):
    """
    Create or update the result sets for a given query ID.

    Args:
        query_id (int): The ID of the user query.
        contact_ids (List[int]): The list of contact IDs to add to the result sets.
        db_name (str): The name of the database.

    Returns:
        int: The number of results added.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.executemany('''
            INSERT OR IGNORE INTO result_sets (query_id, contact_id)
            VALUES (?, ?)
        ''', [(query_id, contact_id) for contact_id in contact_ids])
        
        inserted_count = cursor.rowcount
        conn.commit()
        return inserted_count
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return 0
    finally:
        conn.close()

def fetch_detailed_contact_profile(contact_id: int, db_name='contacts.db') -> Dict[str, Any]:
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Fetch contact details
        cursor.execute('SELECT * FROM contacts WHERE id = ?', (contact_id,))
        contact = cursor.fetchone()
        contact_columns = [description[0] for description in cursor.description]
        contact_data = dict(zip(contact_columns, contact))
        
        # Fetch experiences
        cursor.execute('SELECT * FROM experiences WHERE contact_id = ?', (contact_id,))
        experiences = cursor.fetchall()
        experience_columns = [description[0] for description in cursor.description]
        experience_data = [dict(zip(experience_columns, exp)) for exp in experiences]
        
        # Fetch education
        cursor.execute('SELECT * FROM education WHERE contact_id = ?', (contact_id,))
        education = cursor.fetchall()
        education_columns = [description[0] for description in cursor.description]
        education_data = [dict(zip(education_columns, edu)) for edu in education]
        
        # Combine into a single JSON object
        detailed_profile = {
            "contact": contact_data,
            "experiences": experience_data,
            "education": education_data
        }
        
        return detailed_profile
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return {}
    finally:
        conn.close()


def validate_query(query: str) -> bool:
    """
    Validate that the query is a SELECT statement and does not contain any modifying keywords.

    Args:
        query (str): The SQL query string.

    Returns:
        bool: True if the query is valid, False otherwise.
    """
    # Check if the query starts with SELECT (case-insensitive)
    if not re.match(r'^\s*SELECT', query, re.IGNORECASE):
        return False
    
    # Disallow modifying keywords (case-insensitive)
    disallowed_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'REPLACE']
    for keyword in disallowed_keywords:
        if re.search(r'\b' + keyword + r'\b', query, re.IGNORECASE):
            return False
    
    return True






def get_database_schema() -> Dict[str, Any]:
    schema = {
        'contacts': {
            "description": 'Regular table for storing contact information. Use for exact match and range queries.',
            "columns": [
                {"name": 'id', "type": 'TEXT', "description": 'Unique identifier for the contact'},
                {"name": 'full_name', "type": 'TEXT', "description": 'Full name of the contact'},
                {"name": 'occupation', "type": 'TEXT', "description": 'Occupation of the contact'},
                {"name": 'headline', "type": 'TEXT', "description": 'Headline of the contact'},
                {"name": 'summary', "type": 'TEXT', "description": 'Summary of the contact.'},
                {"name": 'location', "type": 'TEXT', "description": 'Location of the contact'},
                {"name": 'profile_pic_url', "type": 'TEXT', "description": 'Profile picture URL of the contact'},
                {"name": 'linkedin_url', "type": 'TEXT', "description": 'LinkedIn URL of the contact'}
            ]
        },
        'experiences': {
            "description": 'Regular table for storing experience information. Use for exact match and range queries.',
            "columns": [
                {"name": 'contact_id', "type": 'TEXT', "description": 'Unique identifier for the contact'},
                {"name": 'company', "type": 'TEXT', "description": 'Company name'},
                {"name": 'title', "type": 'TEXT', "description": 'Job title'},
                {"name": 'description', "type": 'TEXT', "description": 'Job description'},
                {"name": 'start_date', "type": 'TEXT', "description": 'Start date of the job'},
                {"name": 'end_date', "type": 'TEXT', "description": 'End date of the job'}
            ]
        },
        'education': {
            "description": 'Regular table for storing education information. Use for exact match and range queries.',
            "columns": [
                {"name": 'contact_id', "type": 'TEXT', "description": 'Unique identifier for the contact'},
                {"name": 'school', "type": 'TEXT', "description": 'School name'},
                {"name": 'field_of_study', "type": 'TEXT', "description": 'Field of study'},
                {"name": 'degree', "type": 'TEXT', "description": 'Degree obtained'},
                {"name": 'description', "type": 'TEXT', "description": 'Description of the education'}
            ]
        },
        'contacts_fts': {
            "description": 'FTS table for storing contact information. Use for fuzzy search queries.',
            "columns": [
                {"name": 'id', "type": 'TEXT', "description": 'Unique identifier for the contact'},
                {"name": 'full_name', "type": 'TEXT', "description": 'Full name of the contact'},
                {"name": 'occupation', "type": 'TEXT', "description": 'Occupation of the contact'},
                {"name": 'headline', "type": 'TEXT', "description": 'Headline of the contact'},
                {"name": 'summary', "type": 'TEXT', "description": 'Summary of the contact'},
                {"name": 'location', "type": 'TEXT', "description": 'Location of the contact'},
                {"name": 'profile_pic_url', "type": 'TEXT', "description": 'Profile picture URL of the contact'},
                {"name": 'linkedin_url', "type": 'TEXT', "description": 'LinkedIn URL of the contact'}
            ]
        },
        'experiences_fts': {
            "description": 'Regular table for storing experience information. Use for exact match and range queries.',
            "columns": [
                {"name": 'contact_id', "type": 'INTEGER', "description": 'Unique identifier for the contact'},
                {"name": 'company', "type": 'TEXT', "description": 'Company name'},
                {"name": 'title', "type": 'TEXT', "description": 'Job title'},
                {"name": 'description', "type": 'TEXT', "description": 'Job description'},
                {"name": 'start_date', "type": 'DATE', "description": 'Start date of the job (mm/dd/yyyy)'},
                {"name": 'end_date', "type": 'DATE', "description": 'End date of the job (mm/dd/yyyy)'}
            ]
        },
        'education': {
            "description": 'Regular table for storing education information. Use for exact match and range queries.',
            "columns": [
                {"name": 'contact_id', "type": 'INTEGER', "description": 'Unique identifier for the contact'},
                {"name": 'school', "type": 'TEXT', "description": 'School name'},
                {"name": 'field_of_study', "type": 'TEXT', "description": 'Field of study'},
                {"name": 'degree', "type": 'TEXT', "description": 'Degree obtained'},
                {"name": 'description', "type": 'TEXT', "description": 'Description of the education'},
                {"name": 'start_date', "type": 'DATE', "description": 'Start date of education (mm/dd/yyyy)'},
                {"name": 'end_date', "type": 'DATE', "description": 'End date of education (mm/dd/yyyy)'}
            ]
        },
    }

    
    return schema



