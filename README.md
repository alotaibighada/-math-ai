# Math AI - Web App

## تشغيل المشروع على جهاز واحد (شبكة محلية)

1. أنشئ بيئة افتراضية:
```bash
python -m venv venv
source venv/bin/activate   # على ويندوز: venv\Scripts\activate
```

2. ثبّت المكتبات:
```bash
pip install -r requirements.txt
```

3. شغّل السيرفر:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

4. افتح `index.html` في أي متصفح على نفس الجهاز أو الجوال.
