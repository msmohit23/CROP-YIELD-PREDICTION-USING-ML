import streamlit as st
import sqlite3
import hashlib
import secrets

def hash_password(password, salt=None):
    """Hash a password for storing."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Create the hash
    pwdhash = hashlib.pbkdf2_hmac('sha256', 
                                  password.encode('utf-8'), 
                                  salt.encode('utf-8'), 
                                  100000)
    
    pwdhash = pwdhash.hex()
    
    return salt + ':' + pwdhash

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password.split(':')[0]
    stored_hash = stored_password.split(':')[1]
    
    # Hash the provided password with the same salt
    pwdhash = hashlib.pbkdf2_hmac('sha256', 
                                 provided_password.encode('utf-8'), 
                                 salt.encode('utf-8'), 
                                 100000)
    
    pwdhash = pwdhash.hex()
    
    return pwdhash == stored_hash

def login(username, password):
    """Check if username/password combination is valid"""
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    # Check if user exists
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    
    # No user found with that username
    if result is None:
        return False
    
    # Check password
    stored_password = result[0]
    return verify_password(stored_password, password)

def signup(username, password):
    """Add a new user to the database"""
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    # Check if username already exists
    c.execute('SELECT username FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    
    # Username already exists
    if result is not None:
        conn.close()
        return False
    
    # Hash password
    hashed_password = hash_password(password)
    
    # Insert new user
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
              (username, hashed_password))
    
    conn.commit()
    conn.close()
    
    return True

def check_authenticated():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def logout():
    """Log out the current user"""
    if 'authenticated' in st.session_state:
        del st.session_state['authenticated']
    
    if 'username' in st.session_state:
        del st.session_state['username']
