# main.py (เวอร์ชันนักสืบ)

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

# =================================================================================
# ส่วนที่ 1: การตั้งค่าและโหลดโมเดล (ส่วนนักสืบ)
# =================================================================================

print("🕵️  Starting Debug Mode...")

# ดึงค่าจาก Environment Variables
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('LINE_CHANNEL_SECRET')

# **(สำคัญ)** พิมพ์ค่าที่ได้รับออกมาใน Log เพื่อตรวจสอบ
print(f"🕵️  Loaded Access Token: {channel_access_token}")
print(f"🕵️  Loaded Channel Secret: {channel_secret}")


# ตรวจสอบว่าค่าที่ได้รับมาถูกต้องหรือไม่
if not channel_access_token or not channel_secret:
    print("❌ CRITICAL ERROR: Environment variables are missing or empty!")
    # ในกรณีนี้ เราจะปล่อยให้โปรแกรมหยุดทำงานไปเลย เพื่อให้เห็น Error ชัดๆ

# ถ้าค่าถูกต้อง ให้ทำงานต่อ
print("✅ Environment variables seem to be loaded. Initializing LINE SDK...")
configuration = Configuration(access_token=channel_access_token)
handler = WebhookHandler(channel_secret)

# --- ส่วนที่เหลือของโค้ดเหมือนเดิม ---

CONFIDENCE_THRESHOLD = 0.50
app = FastAPI(title="API วิเคราะห์โรคส้มโอ")

try:
    model = YOLO('best.pt')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

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

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    if model is None:
        print("Error: Model not available.")
        return

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        message_id = event.message.id
        message_content = line_bot_api.get_message_content(message_id=message_id)
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
            reply_text = "ไม่พบร่องรอยของโรคในภาพ หรือความมั่นใจต่ำกว่าเกณฑ์"
        else:
            reply_text = "ผลการวิเคราะห์:\n- " + "\n- ".join(detections)

        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

@app.get("/")
def read_root():
    return {"status": "API is running!", "model_loaded": "Yes" if model else "No"}
