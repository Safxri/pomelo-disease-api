# main.py (ตัด ImageSendMessage ออก)

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
    TextMessageContent,
    FollowEvent
)
DISEASE_INFO = {
    "Canker": {
        "th_name": "โรคแคงเกอร์",
        "advice": "ตัดแต่งกิ่งที่เป็นโรคไปเผาทำลาย และพ่นสารคอปเปอร์ออกซีคลอไรด์"
    },
    "Thrips": {
        "th_name": "เพลี้ยไฟ",
        "advice": "พ่นสารกลุ่มสไปนีโทแรม (Spinetoram) ในช่วงใบอ่อน"
    }
}

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

# --- ส่วนจัดการ Event จาก LINE (ส่วนที่ปรับปรุง) ---

@handler.add(FollowEvent)
def handle_follow_event(event):
    """ทำงานเมื่อมีคนแอดเพื่อนเข้ามาครั้งแรก (ส่งข้อความอย่างเดียว)"""
    reply_text = (
        "สวัสดีครับ ขอบคุณที่เพิ่ม Pomelo Bot เป็นเพื่อน 🙏\n\n"
        "ผมคือผู้ช่วยวิเคราะห์โรคส้มโอทับทิมสยามเบื้องต้นครับ\n\n"
        "หากต้องการเริ่มต้นใช้งาน สามารถพิมพ์ 'วิธีใช้' เพื่อดูคำแนะนำ หรือส่งรูปภาพเข้ามาได้เลยครับ!"
    )
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)] # ส่งข้อความอย่างเดียว
            )
        )

# --- โค้ดส่วนที่เหลือเหมือนเดิมทั้งหมด ---
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
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
    if model is None: return
    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        message_id = event.message.id
        message_content = line_bot_blob_api.get_message_content(message_id=message_id)
        image = Image.open(io.BytesIO(message_content))
        image.thumbnail((640, 640))
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
            reply_text = "ไม่พบร่องรอยของโรคในภาพครับ ลองถ่ายให้ชัดขึ้นนะครับ"
        else:
            reply_text = "✅ ผลการวิเคราะห์:\n"
            for disease, conf in unique_diseases.items():
                info = DISEASE_INFO.get(disease, {"th_name": disease, "advice": "โปรดปรึกษาผู้เชี่ยวชาญ"})
                reply_text += f"\n📌 {info['th_name']} (มั่นใจ {conf:.0%})\n💡 วิธีแก้: {info['advice']}\n"
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
