# Library
import io
import os
from fastapi import FastAPI, Request, HTTPException
from PIL import Image
from ultralytics import YOLO


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

CONFIDENCE_THRESHOLD = 0.50
app = FastAPI(
    title="API วิเคราะห์โรคส้มโอ",
    description="API สำหรับรับ Webhook จาก LINE และทำนายโรค",
    version="1.0"
)


channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)

# **(ส่วนที่แก้ไข)** ตรวจสอบว่ามีการตั้งค่าบน Render หรือไม่ แต่ไม่ต้องใส่ค่าจริงลงในโค้ด
if channel_access_token is None or channel_secret is None:
    print("❌ Error: LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set on Render's Environment Variables.")
    # หยุดการทำงานส่วนนี้ถ้าไม่มีค่า เพื่อป้องกันข้อผิดพลาด
    # เราจะปล่อยให้ตัวแปรเป็น None และปล่อยให้เกิด Error ตอนที่ LINE พยายามเชื่อมต่อ
    # เพื่อให้เรารู้ว่าลืมตั้งค่าบน Render

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
    # ตรวจสอบว่า Channel Secret ถูกตั้งค่าหรือไม่ก่อนใช้งาน
    if channel_secret is None:
        raise HTTPException(status_code=500, detail="LINE Channel Secret is not configured on the server.")

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
    # ตรวจสอบว่าโมเดลและ Access Token พร้อมใช้งานหรือไม่
    if model is None or channel_access_token is None:
        print("Error: Model or Access Token not available.")
        return # ไม่ทำอะไรต่อถ้าโมเดลหรือ token ไม่พร้อม

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
