import os
import psycopg2

try:
    db_password = os.getenv("DB_PASSWORD", "your_super_secret_password")
    
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password=db_password
    )
    
    print("Database connection successful.")
    
    # Close the connection
    conn.close()

except Exception as e:
    print(f"Database connection failed: {e}")