# main.py (Corrected for latest library versions)

import io
import os
from fastapi import FastAPI, Request, HTTPException
from PIL import Image
from ultralytics import YOLO

# Corrected LINE SDK imports
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    ImageSendMessage,
    MessagingApiBlob
)
from linebot.v3.webhooks import (
    MessageEvent,
    ImageMessageContent,
    TextMessageContent,
    FollowEvent
)

# --- App Setup ---
CONFIDENCE_THRESHOLD = 0.50
app = FastAPI(title="API วิเคราะห์โรคส้มโอ")

# --- Environment Variables & LINE SDK Configuration ---
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('LINE_CHANNEL_SECRET')

if not channel_access_token or not channel_secret:
    print("❌ CRITICAL ERROR: Environment variables are missing or empty!")

configuration = Configuration(access_token=channel_access_token)
handler = WebhookHandler(channel_secret)

# --- Model Loading ---
try:
    model = YOLO('best.pt')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Webhook Endpoint ---
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

# --- Event Handlers ---

@handler.add(FollowEvent)
def handle_follow_event(event):
    """Handles when a user adds the bot as a friend."""
    IMAGE_URL = "https://raw.githubusercontent.com/Safxri/pomelo-disease-api/main/S__43040785.jpg" # Your image URL
    reply_text = (
        "สวัสดีครับ ขอบคุณที่เพิ่ม Pomelo Bot เป็นเพื่อน 🙏\n\n"
        "ผมคือผู้ช่วยวิเคราะห์โรคส้มโอทับทิมสยามเบื้องต้นครับ\n\n"
        "หากต้องการเริ่มต้นใช้งาน สามารถพิมพ์ 'วิธีใช้' เพื่อดูคำแนะนำ หรือส่งรูปภาพเข้ามาได้เลยครับ!"
    )
    image_message = ImageSendMessage(
        original_content_url=IMAGE_URL,
        preview_image_url=IMAGE_URL
    )
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[image_message, TextMessage(text=reply_text)]
            )
        )

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    """Handles text messages from users."""
    text = event.message.text.strip().lower()
    reply_text = ""
    if text == "สวัสดี":
        reply_text = "สวัสดีครับ 🙏\n\nหากต้องการดูวิธีใช้งาน พิมพ์ 'วิธีใช้' ได้เลยครับ"
    elif text == "วิธีใช้":
        reply_text = (
            "**วิธีใช้งาน:**\n"
            "1. ถ่ายรูปใบหรือผลส้มโอที่สงสัยว่าเป็นโรคให้ชัดเจน\n"
            "2. ส่งรูปภาพนั้นเข้ามาในแชทนี้ได้เลย\n"
            "3. รอสักครู่... ผมจะวิเคราะห์และส่งผลลัพธ์กลับไปให้ครับ"
        )
    if reply_text:
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
    """Handles image messages from users."""
    if model is None: return
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
