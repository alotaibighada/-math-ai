import io, re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import cv2
import pytesseract
from sympy import symbols, Eq, solve, simplify, sympify

app = FastAPI(title="Math AI API")

# السماح بالوصول من أي مصدر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# خدمة الملفات الثابتة (index.html + frontend)
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# -----------------------------
# دوال المعالجة وحل المعادلات
# -----------------------------
def preprocess_image(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def fix_implied_mul(expr: str) -> str:
    expr = expr.replace(" ", "")
    expr = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z\)])([a-zA-Z\(])', r'\1*\2', expr)
    expr = re.sub(r'(\))(\d|\()', r'\1*\2', expr)
    return expr

def solve_expression_text(text: str):
    cleaned = text.strip().replace("−","-").replace("^","**")
    cleaned = fix_implied_mul(cleaned)
    x = symbols("x")
    try:
        if cleaned.count("=") > 1:
            return {"error": "أكثر من علامة مساواة"}
        if "=" in cleaned:
            left, right = cleaned.split("=", maxsplit=1)
            left_s = sympify(left)
            right_s = sympify(right)
            vars_in_eq = list(left_s.free_symbols.union(right_s.free_symbols))
            if vars_in_eq:
                eq = Eq(left_s, right_s)
                sol = solve(eq, vars_in_eq)
                return {"type": "equation", "fixed": f"{simplify(left_s)} = {simplify(right_s)}", "solution": sol}
            else:
                return {"type": "boolean", "fixed": f"{simplify(left_s)} = {simplify(right_s)}", "equal": left_s==right_s}
        else:
            val = sympify(cleaned).evalf()
            return {"type": "expression", "value": float(val)}
    except Exception as e:
        return {"error": f"تعذر الحل: {str(e)}"}

# -----------------------------
# API حل الصور
# -----------------------------
@app.post("/solve-image")
async def solve_image(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="الملف يجب أن يكون صورة")
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="تعذر فتح الصورة")
    img_proc = preprocess_image(pil_img)
    pil_for_ocr = Image.fromarray(img_proc)
    ocr_text = pytesseract.image_to_string(pil_for_ocr, config='--psm 6').strip()
    if not ocr_text:
        return {"ok": False, "message": "لم يُستخرج نص من الصورة", "extracted": ""}
    result = solve_expression_text(ocr_text)
    return {"ok": True, "extracted": ocr_text, "result": result}

@app.get("/api-status")
def read_root():
    return {"msg":"Math AI API running"}
