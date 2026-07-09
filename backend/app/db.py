import sqlite3
import os
from datetime import datetime, timedelta

class Database:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, 'helioscast.db')
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    predicted_speed REAL,
                    actual_speed REAL,
                    error REAL,
                    risk_class TEXT
                )
            ''')
            conn.commit()

    def log_prediction(self, predicted_speed, risk_class):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            timestamp = datetime.utcnow().isoformat() + "Z"
            cursor.execute('''
                INSERT INTO predictions (timestamp, predicted_speed, risk_class)
                VALUES (?, ?, ?)
            ''', (timestamp, predicted_speed, risk_class))
            
            # Keep database lightweight for deployment: retain only latest 500 records
            cursor.execute('''
                DELETE FROM predictions 
                WHERE id NOT IN (
                    SELECT id FROM predictions 
                    ORDER BY id DESC 
                    LIMIT 500
                )
            ''')
            conn.commit()

    def update_actuals(self, history_data):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all predictions that haven't been resolved yet
            cursor.execute('''
                SELECT id, timestamp, predicted_speed FROM predictions 
                WHERE actual_speed IS NULL 
            ''')
            unresolved = cursor.fetchall()
            
            for pred_id, pred_ts, pred_speed in unresolved:
                try:
                    # Parse prediction timestamp
                    clean_ts = pred_ts.replace('Z', '+00:00')
                    if '+' not in clean_ts and '-' not in clean_ts[10:]:
                        clean_ts += '+00:00'
                    target_dt = datetime.fromisoformat(clean_ts) + timedelta(minutes=60)
                    
                    # Find the closest historical point that matches the target time (within 10 minutes)
                    best_match = None
                    min_diff = 601
                    for point in history_data:
                        point_ts = point['timestamp'].replace('Z', '+00:00')
                        if '+' not in point_ts and '-' not in point_ts[10:]:
                            point_ts += '+00:00'
                        point_dt = datetime.fromisoformat(point_ts)
                        
                        time_diff = abs((point_dt - target_dt).total_seconds())
                        if time_diff < min_diff:
                            min_diff = time_diff
                            best_match = point
                    
                    if best_match is not None:
                        actual_speed = best_match['speed']
                        error = abs(pred_speed - actual_speed)
                        
                        cursor.execute('''
                            UPDATE predictions 
                            SET actual_speed = ?, error = ? 
                            WHERE id = ?
                        ''', (actual_speed, error, pred_id))
                except Exception as e:
                    print(f"Error in update_actuals: {e}")
                    pass
            conn.commit()

    def get_prediction_history(self, limit=50):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, predicted_speed, actual_speed, error, risk_class 
                FROM predictions 
                WHERE actual_speed IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            
        history = []
        for row in rows:
            history.append({
                "timestamp": row[0],
                "predicted_speed": row[1],
                "actual_speed": row[2] if row[2] is not None else None,
                "error": row[3] if row[3] is not None else None,
                "risk_class": row[4]
            })
        return history

db = Database()
