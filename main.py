# main.py (ฉบับสมบูรณ์)

import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------
# ส่วนที่ 1: การตั้งค่าและโหลดโมเดล
# --------------------------------------------------------------------------

# กำหนดเกณฑ์ความมั่นใจขั้นต่ำที่เราจะยอมรับ (เช่น 50%)
CONFIDENCE_THRESHOLD = 0.50

# สร้าง Application ด้วย FastAPI
app = FastAPI(
    title="API วิเคราะห์โรคส้มโอ",
    description="รับรูปภาพใบส้มโอและทำนายโรคด้วยโมเดล YOLO",
    version="1.0"
)

# โหลดโมเดล AI (จะทำงานแค่ครั้งเดียวตอนเซิร์ฟเวอร์เริ่มทำงาน)
try:
    model = YOLO('best.pt')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --------------------------------------------------------------------------
# ส่วนที่ 2: สร้าง API Endpoints (ช่องทางการสื่อสาร)
# --------------------------------------------------------------------------

@app.get("/")
def read_root():
    """
    Endpoint หลักสำหรับตรวจสอบว่า API ทำงานอยู่หรือไม่
    """
    return {
        "status": "API is running!",
        "model_loaded": "Yes" if model else "No, there was an error"
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Endpoint สำหรับรับไฟล์รูปภาพและวิเคราะห์โรค
    """
    # ตรวจสอบว่าโมเดลพร้อมใช้งานหรือไม่
    if not model:
        return {"error": "Model is not loaded, please check server logs."}

    # 1. อ่านข้อมูลรูปภาพที่อัปโหลดมา
    image_bytes = await file.read()

    # 2. แปลงข้อมูลรูปภาพให้โมเดลเข้าใจ
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Invalid image file: {e}"}

    # 3. ส่งรูปภาพเข้าโมเดลเพื่อวิเคราะห์ผล
    results = model(image)

    # 4. จัดรูปแบบและกรองผลลัพธ์
    detections = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf)
            # กรองผลลัพธ์ที่มีความมั่นใจต่ำกว่าเกณฑ์ออก
            if confidence >= CONFIDENCE_THRESHOLD:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                detections.append({
                    "disease_name": class_name,
                    "confidence": round(confidence, 2)
                })

    # 5. คืนค่าผลลัพธ์ที่ดีที่สุดเพียงหนึ่งเดียว
    if not detections:
        # กรณีไม่เจออะไรเลย หรือที่เจอมีความมั่นใจต่ำกว่าเกณฑ์
        return {
            "status": "healthy",
            "disease_name": "ไม่พบร่องรอยของโรค",
            "confidence": 1.0
        }

    # ถ้าเจอ ให้เรียงลำดับผลลัพธ์ตามความมั่นใจและเลือกอันที่ดีที่สุด
    best_detection = sorted(detections, key=lambda x: x['confidence'], reverse=True)[0]

    return {
        "status": "disease_found",
        "disease_name": best_detection['disease_name'],
        "confidence": best_detection['confidence']
    }