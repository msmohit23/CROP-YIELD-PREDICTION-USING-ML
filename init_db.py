import sqlite3

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
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()