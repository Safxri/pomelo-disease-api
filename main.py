# main.py (เพิ่มฟังก์ชันทักทาย)

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
    TextMessage,
    MessagingApiBlob
)
from linebot.v3.webhooks import (
    MessageEvent,
    ImageMessageContent,
    TextMessageContent  # **(เพิ่ม import ใหม่)**
)

# --- ส่วนตั้งค่าและโหลดโมเดล (เหมือนเดิม) ---

CONFIDENCE_THRESHOLD = 0.50
app = FastAPI(title="API วิเคราะห์โรคส้มโอ")

channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('LINE_CHANNEL_SECRET')

if not channel_access_token or not channel_secret:
    print("❌ CRITICAL ERROR: Environment variables are missing or empty!")

configuration = Configuration(access_token=channel_access_token)
handler = WebhookHandler(channel_secret)

try:
    model = YOLO('best.pt')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- ส่วน Webhook Endpoint (เหมือนเดิม) ---

@app.post("/webhook")
async def line_webhook(request: Request):
    signature = request.headers.get('X-Line-Signature')
    body = await request.body()
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        print(f"❌ Error in webhook handler: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return 'OK'

# --- ส่วนจัดการข้อความ (ส่วนที่เพิ่มเติม) ---

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    """
    ฟังก์ชันใหม่! ที่จะทำงานเมื่อได้รับข้อความที่เป็น "ตัวอักษร"
    """
    text = event.message.text.strip() # .strip() เพื่อตัดช่องว่างหน้า-หลัง

    # ตรวจสอบว่าข้อความที่ส่งมาคือ "สวัสดี" หรือไม่
    if text == "สวัสดี":
        reply_text = (
            "สวัสดีครับ 🙏\n\n"
            "ผมคือแชทบอทวิเคราะห์โรคส้มโอทับทิมสยาม "
            "เพียงส่งรูปภาพใบหรือผลส้มโอที่สงสัยเข้ามาได้เลยครับ"
        )
        
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    """
    ฟังก์ชันเดิม ที่ทำงานเมื่อได้รับ "รูปภาพ"
    """
    if model is None:
        print("Error: Model not available.")
        return

    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        
        message_id = event.message.id
        message_content = line_bot_blob_api.get_message_content(message_id=message_id)
        
        image = Image.open(io.BytesIO(message_content))
        results = model(image)
        
        detections = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                if confidence >= CONFIDENCE_THRESHOLD:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    detections.append(f"{class_name} (ความมั่นใจ: {confidence:.0%})")
        
        if not detections:
            reply_text = "ไม่พบร่องรอยของโรคในภาพ หรือความมั่นใจต่ำกว่าเกณฑ์ครับ"
        else:
            reply_text = "ผลการวิเคราะห์:\n- " + "\n- ".join(detections)

        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

@app.get("/")
def read_root():
    return {"status": "API is running!", "model_loaded": "Yes" if model else "No"}
