# main.py (แยกการตอบกลับ "สวัสดี" และ "วิธีใช้")

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
    TextMessageContent
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

# --- ส่วนจัดการข้อความ (ส่วนที่ปรับปรุงใหม่) ---

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    """
    ฟังก์ชันที่ทำงานเมื่อได้รับข้อความที่เป็น "ตัวอักษร"
    """
    text = event.message.text.strip().lower()
    reply_text = "" # สร้างตัวแปรว่างไว้ก่อน

    # เงื่อนไขที่ 1: ถ้าผู้ใช้พิมพ์ "สวัสดี"
    if text == "สวัสดี":
        reply_text = (
            "สวัสดีครับ 🙏\n\n"
            "ยินดีต้อนรับสู่แชทบอทวิเคราะห์โรคส้มโอทับทิมสยามครับ\n\n"
            "หากต้องการดูวิธีใช้งาน พิมพ์ 'วิธีใช้' ได้เลยครับ"
        )
    
    # เงื่อนไขที่ 2: ถ้าผู้ใช้พิมพ์ "วิธีใช้"
    elif text == "วิธีใช้":
        reply_text = (
            "**วิธีใช้งาน:**\n"
            "1. ถ่ายรูปใบหรือผลส้มโอที่สงสัยว่าเป็นโรคให้ชัดเจน\n"
            "2. ส่งรูปภาพนั้นเข้ามาในแชทนี้ได้เลย\n"
            "3. รอสักครู่... ผมจะวิเคราะห์และส่งผลลัพธ์กลับไปให้ครับ"
        )
    
    # ถ้ามีข้อความที่ต้องตอบกลับ (reply_text ไม่ใช่ค่าว่าง)
    if reply_text:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )

# --- ส่วนจัดการรูปภาพ (เหมือนเดิม) ---

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    if model is None:
        print("Error: Model not available.")
        return

    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        message_id = event.message.id
        message_content = line_bot_blob_api.get_message_content(message_id=message_id)
        
        image = Image.open(io.BytesIO(message_content))
        results = model(image)
        
        unique_diseases = {}
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                if confidence >= CONFIDENCE_THRESHOLD:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    
                    if class_name in unique_diseases:
                        if confidence > unique_diseases[class_name]:
                            unique_diseases[class_name] = confidence
                    else:
                        unique_diseases[class_name] = confidence
        
        if not unique_diseases:
            reply_text = "ไม่พบร่องรอยของโรคในภาพ หรือความมั่นใจต่ำกว่าเกณฑ์ครับ"
        else:
            detection_texts = []
            for disease, conf in unique_diseases.items():
                detection_texts.append(f"{disease} (ความมั่นใจสูงสุด: {conf:.0%})")
            
            reply_text = "ผลการวิเคราะห์:\n- " + "\n- ".join(detection_texts)

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
