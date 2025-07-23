# main.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, ImageMessage, StickerMessage, FlexSendMessage, ImageSendMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError

# --- ตั้งค่า Line Bot API Keys (สำคัญมาก!) ---
# คุณต้องนำ Channel Access Token และ Channel Secret ของคุณมาใส่ตรงนี้
# **ไปที่ Line Developers Console ของคุณเพื่อคัดลอกค่าเหล่านี้**
# **สำคัญมาก: Reissue Channel Access Token ใหม่และคัดลอกค่าทั้งหมดที่ถูกต้องมาใส่**
LINE_CHANNEL_ACCESS_TOKEN = "YOUR_LINE_CHANNEL_ACCESS_TOKEN"  ## << แก้ไขตรงนี้! (นำค่าจริงมาใส่)
LINE_CHANNEL_SECRET = "YOUR_LINE_CHANNEL_SECRET"      ## << แก้ไขตรงนี้! (นำค่าจริงมาใส่)

# ตรวจสอบว่าได้ตั้งค่า API Keys แล้ว
if LINE_CHANNEL_ACCESS_TOKEN == "YOUR_LINE_CHANNEL_ACCESS_TOKEN" or \
   LINE_CHANNEL_SECRET == "YOUR_LINE_CHANNEL_SECRET":
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! WARNING: Please update LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET in main.py !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # สามารถ raise ValueError("กรุณาตั้งค่า Line API Keys ใน main.py") เพื่อให้โปรแกรมหยุดทำงานถ้ายังไม่ได้ตั้งค่า

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

app = FastAPI()

# --- โหลดโมเดล YOLO ที่ฝึกแล้ว (best.pt) ---
# ตรวจสอบให้แน่ใจว่าไฟล์ best.pt อยู่ในโฟลเดอร์เดียวกันกับ main.py
MODEL_PATH = 'best.pt'
try:
    YOLO_MODEL = YOLO(MODEL_PATH)
    # ตั้งค่าให้โมเดลใช้ GPU ถ้ามี (เร็วกว่า) หรือใช้ CPU (ช้ากว่า)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    YOLO_MODEL.to(DEVICE)
    print(f"YOLO model loaded on {DEVICE} from {MODEL_PATH}") #

    # กำหนดชื่อคลาสตามลำดับที่คุณใช้ใน Roboflow และในไฟล์ data.yaml ของคุณตอนเทรน
    # *** สำคัญมาก! ต้องตรงเป๊ะกับลำดับคลาสใน data.yaml ที่คุณใช้เทรน (รวมถึงตัวพิมพ์เล็ก-ใหญ่) ***
    # ตัวอย่าง: ถ้าใน data.yaml ของคุณ class 0 คือ 'leaf_miner', class 1 คือ 'canker', class 2 คือ 'normal_leaf'
    # คุณจะต้องใส่ ['leaf_miner', 'canker', 'normal_leaf']
    CLASS_NAMES_FOR_MODEL = ['Canker', 'Leaf Miner', 'Normal Leaf'] ## << แก้ไขตรงนี้! (ใส่ชื่อคลาสทั้งหมดตามลำดับ ID ใน data.yaml)

except Exception as e:
    print(f"Error loading YOLO model: {e}")
    YOLO_MODEL = None # หากโหลดโมเดลไม่ได้ ระบบจะไม่ทำงานต่อ

# ฟังก์ชันช่วยแปลงชื่อคลาสจากโมเดล (ภาษาอังกฤษ) เป็นภาษาไทย
def disease_name_thai(english_name): #
    mapping = {
        'Canker': 'แคงเกอร์',
        'Leaf Miner' : 'หนอนชอนใบ' ,
        'Normal Leaf': 'ใบปกติ',
        # << เพิ่มคลาสอื่นๆ ที่คุณเทรนโมเดลที่นี่ให้ครบถ้วน >>  ## << แก้ไขตรงนี้! (เพิ่ม mapping สำหรับคลาสของคุณ)
    }
    return mapping.get(english_name, english_name) # คืนชื่อภาษาอังกฤษถ้าไม่พบใน mapping

# นี่คือส่วนที่ Line ส่งข้อมูลเข้ามาที่ Backend ของเรา
@app.post("/webhook")
async def callback(request: Request): #
    signature = request.headers['X-Line-Signature']
    body = await request.body()
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"

# เมื่อ Line ส่งข้อความรูปภาพมา
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event): #
    if not YOLO_MODEL: # ถ้าโมเดลโหลดไม่สำเร็จ
        line_bot_api.reply_message(event.reply_token, TextMessage(text="ขออภัยค่ะ ระบบวิเคราะห์ภาพยังไม่พร้อมใช้งาน (โมเดลไม่โหลด)"))
        return

    # 1. ดึงรูปภาพจาก Line
    message_content = line_bot_api.get_message_content(event.message.id) #
    image_bytes = b''
    for chunk in message_content.iter_content():
        image_bytes += chunk

    # 2. แปลง bytes ของรูปภาพเป็นรูปแบบที่ OpenCV อ่านได้
    np_arr = np.frombuffer(image_bytes, np.uint8) #
    img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_cv2 is None: #
        line_bot_api.reply_message(event.reply_token, TextMessage(text="ไม่สามารถประมวลผลรูปภาพได้ กรุณาส่งรูปภาพที่ชัดเจน"))
        return

    # 3. แปลงจาก OpenCV (BGR) เป็น PIL (RGB) เพื่อให้ YOLO อ่านได้
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)) #

    # 4. ทำการอนุมาน (inference) ด้วยโมเดล YOLO
    results = YOLO_MODEL(img_pil, imgsz=640) # imgsz ควรเท่ากับขนาดที่คุณใช้เทรนโมเดล (ปกติ 640)

    # ใช้ list ของ dictionaries เพื่อเก็บโรคที่ตรวจพบพร้อมค่าความมั่นใจ
    detected_diseases_with_conf = []
    
    # 5. ประมวลผลผลลัพธ์จากโมเดล
    for r in results: # r คือ Results object สำหรับแต่ละรูปภาพ (ในกรณีนี้มี 1 รูป)
        for box in r.boxes: # วนลูปผ่าน bounding boxes ที่โมเดลตรวจพบ
            class_id = int(box.cls) # Class ID ของสิ่งที่ตรวจพบ
            conf = float(box.conf)  # Confidence score (ความมั่นใจของโมเดล)
            
            # ตรวจสอบ confidence score (ความมั่นใจ) ถ้าต่ำเกินไปอาจจะไม่นับเป็นโรค
            if conf > 0.5: # กำหนดเกณฑ์ความมั่นใจ เช่น 0.5 หมายถึง 50%
                class_name = CLASS_NAMES_FOR_MODEL[class_id]
                # เพิ่มข้อมูลโรคและความมั่นใจลงใน list
                detected_diseases_with_conf.append({
                    'name': class_name,
                    'confidence': conf
                })

    # กรองเอาเฉพาะโรคที่ไม่ซ้ำกัน และเก็บค่าความมั่นใจสูงสุดของแต่ละโรค
    unique_diseases_info = {}
    for item in detected_diseases_with_conf:
        name = item['name']
        conf = item['confidence']
        if name not in unique_diseases_info or conf > unique_diseases_info[name]['confidence']:
            unique_diseases_info[name] = {'name': name, 'confidence': conf}
            
    final_detected_diseases = list(unique_diseases_info.values())
    
    # 6. สร้างข้อความตอบกลับ Line
    reply_text = ""
    if len(final_detected_diseases) == 0:
        # ถ้าไม่พบโรคใดๆ เลย
        reply_text = "ไม่พบอาการของโรคในรูปภาพค่ะ ต้นส้มโอของคุณอาจจะ**ใบปกติ** หรืออาการยังไม่ชัดเจน\n\nข้อควรระวัง: นี่เป็นการวิเคราะห์เบื้องต้นเท่านั้น หากมีข้อสงสัย ควรปรึกษาผู้เชี่ยวชาญด้านพืชหรือเกษตรอำเภอค่ะ"
    else:
        # ถ้าพบโรค
        if len(final_detected_diseases) > 1:
            reply_text += "**พบหลายโรค**ในรูปภาพของคุณค่ะ:\n"
        else:
            reply_text += "ผลการวิเคราะห์เบื้องต้นพบอาการของโรคดังนี้ค่ะ:\n"

        for disease_info in final_detected_diseases:
            disease_name_en = disease_info['name']
            confidence_percentage = int(disease_info['confidence'] * 100) # แปลงเป็นเปอร์เซ็นต์
            reply_text += f"- **{disease_name_thai(disease_name_en)}** ({confidence_percentage}%)\n" #

        reply_text += "\n**คำแนะนำเบื้องต้น:**\n"
        reply_text += "โปรดจำไว้ว่านี่เป็นการวิเคราะห์จาก AI ซึ่งเป็นเพียงข้อมูลเบื้องต้นเท่านั้น เพื่อการวินิจฉัยที่แม่นยำและคำแนะนำในการจัดการโรคที่ถูกต้อง ควรปรึกษาผู้เชี่ยวชาญด้านพืชหรือเกษตรอำเภอในพื้นที่ของคุณค่ะ"

    # 7. ส่งข้อความตอบกลับไปที่ Line
    line_bot_api.reply_message( #
        event.reply_token,
        TextMessage(text=reply_text)
    )

# เมื่อ Line ส่งข้อความตัวอักษรมา (สำหรับคำทักทายหรือช่วยเหลือ)
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event): #
    user_message = event.message.text.lower() # ทำให้เป็นตัวเล็กทั้งหมด
    if "สวัสดี" in user_message or "hi" in user_message or "hello" in user_message:
        reply_text = """
สวัสดีครับ/ค่ะ! ยินดีต้อนรับสู่ระบบ Doo Pomelo!
ผม/ฉันคือแชทบอท AI อัจฉริยะที่ปรึกษาด้านสุขภาพส้มโอทับทิมสยาม 🤖✨
สามารถวิเคราะห์โรคจากรูปภาพที่คุณส่งมาได้
เพียงแค่:
1. ส่งรูปภาพอาการของส้มโอ เช่น ใบ กิ่ง หรือผล
2. รอรับผลการวิเคราะห์เบื้องต้นและคำแนะนำ
คุณสามารถส่งรูปภาพมาให้ผม/ฉันวิเคราะห์ได้เลยครับ/ค่ะ! 🌿🍊
"""
        # หากต้องการเพิ่มรูปภาพประกอบเหมือนภาพตัวอย่าง Line Official Account
        # คุณจะต้องมี URL ของรูปภาพนั้นที่สามารถเข้าถึงได้จากภายนอก
        # และส่งเป็นหลาย Message ใน Reply
        # ตัวอย่าง:
        # messages_to_reply = [
        #     TextMessage(text=reply_text),
        #     ImageSendMessage(
        #         original_content_url="https://yourdomain.com/path/to/full_image.jpg", # URL ของรูปภาพขนาดเต็ม
        #         preview_image_url="https://yourdomain.com/path/to/preview_image.jpg" # URL ของรูปภาพขนาดย่อ
        #     )
        # ]
        # line_bot_api.reply_message(event.reply_token, messages_to_reply)
        # หากใช้ ImageSendMessage, ต้องมั่นใจว่า URL ของรูปภาพนั้นเป็น Public URL ที่ Line เข้าถึงได้
        # และต้องเปลี่ยน line_bot_api.reply_message ให้รับ list ของ messages
        
    elif "ช่วยเหลือ" in user_message or "help" in user_message:
        reply_text = "ฉันคือระบบวิเคราะห์โรคส้มโอทับทิมสยามเบื้องต้นค่ะ เพียงส่งรูปภาพส่วนที่เป็นอาการ เช่น ใบ กิ่ง หรือผล มาให้ฉัน ก็จะได้รับผลวิเคราะห์ค่ะ"
    else:
        reply_text = "ฉันเข้าใจว่าคุณอาจมีคำถามเพิ่มเติม แต่ฉันสามารถวิเคราะห์ได้จากรูปภาพเท่านั้นค่ะ ลองส่งรูปภาพมาดูนะคะ"
    
    line_bot_api.reply_message( #
        event.reply_token,
        TextMessage(text=reply_text)
    )
