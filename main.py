# main.py (ฉบับสมบูรณ์สำหรับ LINE Bot)

import io
import os
from fastapi import FastAPI, Request, HTTPException
from PIL import Image
from ultralytics import YOLO

# นำเข้า Library ของ LINE
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    ImageMessageContent
)

# --------------------------------------------------------------------------
# ส่วนที่ 1: การตั้งค่าและโหลดโมเดล
# --------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.50

app = FastAPI(
    title="API วิเคราะห์โรคส้มโอ",
    description="API สำหรับรับ Webhook จาก LINE และทำนายโรค",
    version="1.0"
)

# ดึงค่า Channel Access Token และ Channel Secret จาก Environment Variables
# **คุณต้องไปตั้งค่านี้ในหน้า Settings ของ Render**
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)

if channel_access_token is None or channel_secret is None:
    print("❌ Error: LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set.")
    # ในกรณีที่ไม่ได้ตั้งค่า ให้ใช้ค่าสมมติเพื่อไม่ให้แอปพัง แต่จะทำงานไม่ถูกต้อง
    channel_access_token = "vC4zbb62rQF0MhtikvV1iDjLg/Aa29UP52aixFuzcLtrfCmQLvW0kLxQmR4107TaMvl/O9dkljSh07cEUm/c9uVV+S5zgf2LPj6YzP+AxsdJEGzJM/j2JKhnxIpFDvognn+vG2o6w+kXMp9+5r66rAdB04t89/1O/w1cDnyilFU=" 
    channel_secret = "a26b5ed7c45d1aecda85643c6e92d25a"

configuration = Configuration(access_token=channel_access_token)
handler = WebhookHandler(channel_secret)

try:
    model = YOLO('best.pt')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --------------------------------------------------------------------------
# ส่วนที่ 2: สร้าง Webhook Endpoint สำหรับ LINE
# --------------------------------------------------------------------------

@app.post("/webhook")
async def line_webhook(request: Request):
    """
    Endpoint ใหม่สำหรับรับข้อมูลจาก LINE โดยเฉพาะ
    """
    signature = request.headers.get('X-Line-Signature')
    body = await request.body()

    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    return 'OK'


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    """
    ฟังก์ชันที่จะทำงานเมื่อได้รับข้อความที่เป็น "รูปภาพ"
    """
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        
        # 1. ดึง ID ของรูปภาพเพื่อไปดาวน์โหลด
        message_id = event.message.id
        message_content = line_bot_api.get_message_content(message_id=message_id)
        
        # 2. นำรูปภาพมาวิเคราะห์ด้วยโมเดล
        image = Image.open(io.BytesIO(message_content))
        results = model(image)
        
        # 3. จัดรูปแบบผลลัพธ์
        detections = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                if confidence >= CONFIDENCE_THRESHOLD:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    detections.append(f"{class_name} (ความมั่นใจ: {confidence:.0%})")
        
        # 4. สร้างข้อความตอบกลับ
        if not detections:
            reply_text = "ไม่พบร่องรอยของโรคในภาพ หรือความมั่นใจต่ำกว่าเกณฑ์"
        else:
            reply_text = "ผลการวิเคราะห์:\n- " + "\n- ".join(detections)

        # 5. ส่งข้อความตอบกลับไปหาผู้ใช้
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

# --------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"status": "API is running!", "model_loaded": "Yes" if model else "No"}
