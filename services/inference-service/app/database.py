import os
import psycopg2
from psycopg2.extras import RealDictCursor
 
def get_connection():
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("şifre tanımlı değil, .env dosyasını kontroledin")
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        database=os.getenv("POSTGRES_DB", "dronedb"),
        user=os.getenv("POSTGRES_USER", "drone"),
        password=password,
    )
def init_db():
    conn =get_connection()
    cur=conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id SERIAL PRIMARY KEY,
            frame_id VARCHAR(64),
            model_name VARCHAR(64),          -- hangi model tespit etti
            label VARCHAR(64),               -- car, person, bicycle, bus...
            confidence FLOAT,                -- 0.0 - 1.0
            bbox JSONB,                      -- {x1, y1, x2, y2}
            drone_id VARCHAR(64),            -- hangi drone kamerasından
            stream_url TEXT,                 -- RTSP stream adresi
            detected_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()