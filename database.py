import sqlite3
from datetime import datetime

def init_db():
    """Initialize the database with tables if they don't exist"""
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    
    # Create predictions table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        area TEXT NOT NULL,
        item TEXT NOT NULL,
        year INTEGER NOT NULL,
        rainfall REAL NOT NULL,
        pesticides REAL NOT NULL,
        temperature REAL NOT NULL,
        yield_value REAL NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def get_user_id(username):
    """Get user ID from username"""
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    c.execute('SELECT id FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    
    conn.close()
    
    if result:
        return result[0]
    else:
        return None

def save_prediction(username, area, item, year, rainfall, pesticides, temperature, yield_value):
    """Save a prediction to the database"""
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    # Get user ID
    user_id = get_user_id(username)
    
    if user_id:
        # Get current date and time
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Insert prediction
        c.execute('''
        INSERT INTO predictions (user_id, date, area, item, year, rainfall, pesticides, temperature, yield_value)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, current_date, area, item, year, rainfall, pesticides, temperature, yield_value))
        
        conn.commit()
    
    conn.close()

def get_user_predictions(username):
    """Get all predictions for a user"""
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    # Get user ID
    user_id = get_user_id(username)
    
    if user_id:
        # Get all predictions for this user
        c.execute('''
        SELECT id, date, area, item, year, rainfall, pesticides, temperature, yield_value
        FROM predictions
        WHERE user_id = ?
        ORDER BY date DESC
        ''', (user_id,))
        
        results = c.fetchall()
    else:
        results = []
    
    conn.close()
    
    return results
