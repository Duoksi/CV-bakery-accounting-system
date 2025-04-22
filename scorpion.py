import cv2
from ultralytics import YOLO
import random
import ntplib
from time import ctime, sleep, time
import datetime
import mysql.connector
import threading
import os
import sched


def write_time_to_file():
    c = ntplib.NTPClient()
    while True:
        try:
            with open('C:\\Users\\video-ai\\Desktop\\backup_{}.txt'.format(datetime.datetime.now().strftime("%Y-%m-%d")), 'a') as f:
                response = c.request('censored')
                current_time = ctime(response.tx_time)
                f.write(current_time + '\n')
                f.flush()  # Сбрасываем буфер вывода в файл
                os.fsync(f.fileno())  # Принудительно сбрасываем данные на диск
                sleep(1)
        except Exception as e:
            # print(datetime.datetime.now(), "Error:", e)
            pass


def remove_file_daily():
    scheduler = sched.scheduler()
    def remove_file():
        os.remove('C:\\Users\\video-ai\\Desktop\\backup_{}.txt'.format(datetime.datetime.now().strftime("%Y-%m-%d")))
        scheduler.enter(172800, 1, remove_file)
    scheduler.enter(172800, 1, remove_file)
    scheduler.run()

def create_table():
    try:
        connection = mysql.connector.connect(
            host="censored",
            user="root",
            password="",
            database="database"
        )

        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id INT AUTO_INCREMENT PRIMARY KEY,
                machine VARCHAR(255),
                current_label VARCHAR(255),
                time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        print("Table created successfully")

    except mysql.connector.Error as error:
        print("Failed to create table in MySQL:", error)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def write_to_database(machine, current_label, current_time):
    try:
        connection = mysql.connector.connect(
            host="censored",
            user="root",
            password="",
            database="database"
        )

        cursor = connection.cursor()

        sql = "INSERT INTO labels (machine, current_label, time) VALUES (%s, %s, %s)"
        current_time = datetime.datetime.strptime(current_time, "%a %b %d %H:%M:%S %Y")
        
        current_time_formatted = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        val = (machine, current_label, current_time_formatted)
        cursor.execute(sql, val)

        connection.commit()
        print("Record inserted successfully into database")

    except mysql.connector.Error as error:
        print("Failed to insert record into MySQL table:", error)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def process_video_with_tracking(model, input_video_path, show_video=True):
    c = ntplib.NTPClient()
    current_label = None
    machine = 'scorpion'
    while True:
        try:
            cap = cv2.VideoCapture(input_video_path)
        except Exception as e:
            print(datetime.datetime.now(),"Error:", e)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            
            frame = frame[380:980, 530:1300]
            try:
                results = model.track(frame, iou=0.4, conf=0.89, verbose=False, tracker="botsort.yaml")
                if results[0].boxes.id != None: 
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    cls = results[0].boxes.cls.cpu().numpy().astype(int)

                    for box, cl in zip(boxes, cls):
                        name = results[0].names[cl]
                        if current_label != name:
                            current_label = name
                            response = c.request('censored')
                            print(ctime(response.tx_time), name)
                            current_time = ctime(response.tx_time)
                            write_to_database(machine, current_label, current_time)
                            
                        random.seed(int(cl))
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 2)
                        cv2.putText(
                            frame,
                            f"{name}",
                            (box[0], box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2
                        )

                if show_video:
                    cv2.imshow("frame", frame)

            except Exception as e:
                print(datetime.datetime.now(),"Error:", e)
                cap = cv2.VideoCapture(input_video_path)

            key = cv2.waitKey(1)
            if key == 27:  # Press ESC to exit/close each window.
                break
            
        cap.release()
        cv2.destroyAllWindows()




if __name__ == '__main__':
    # model = YOLO('yolov8n.pt')
    # results = model.train(data='scorpion_dataset/data.yaml', epochs=100, imgsz=640)
    create_table()
    time_thread = threading.Thread(target=write_time_to_file)
    time_thread.start()
    remove_thread = threading.Thread(target=remove_file_daily)
    remove_thread.start()
    model = YOLO('runs/detect/train5/weights/best.pt')
    model.fuse()
    process_video_with_tracking(model, "camera_rtps_ip", show_video=True)