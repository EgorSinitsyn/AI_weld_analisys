from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse, HTMLResponse
import app.schemas as schemas
import app.models as models
from dotenv import load_dotenv
from app.database import engine, get_db, Base
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from pathlib import Path
import httpx
from app.visualize_predictions import PanoramaProcessor

# Загружаем переменные окружения из файла .env
load_dotenv()

# Базовый URL ML-сервиса, по умолчанию для локальной разработки
ML_SERVICE_BASE_URL: str = os.getenv("ML_SERVICE_URL", "http://localhost:8080")
ML_SERVICE_DETECT_ENDPOINT: str = f"{ML_SERVICE_BASE_URL}/detect"

# Инициализация FastAPI-приложения
application = FastAPI()

processor = PanoramaProcessor()


# Настройка CORS (при необходимости)
application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка статических файлов и шаблонов
RESULTS_DIR = Path("static/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
application.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Создание таблиц в базе данных при старте
Base.metadata.create_all(bind=engine)

@application.get("/", response_class=HTMLResponse)
def read_root(request: Request) -> HTMLResponse:
    """
    Возвращает HTML-шаблон главной страницы приложения.

    Параметры:
        request (Request): Объект запроса FastAPI.

    Возвращает:
        HTMLResponse: Отрендеренный шаблон index.html.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@application.post("/upload")
async def upload_image(file: UploadFile) -> dict[str, str]:
    """
    Принимает загруженное изображение и сохраняет его во временную папку.

    Параметры:
        file (UploadFile): Загруженный файл изображения.

    Возвращает:
        dict: Словарь с URL результата обработки.
    """
    # upload_dir = Path("temp_uploads")
    # upload_dir.mkdir(exist_ok=True)
    # temp_path = upload_dir / file.filename
    # content = await file.read()
    # if not content:
    #     raise HTTPException(status_code=400, detail="Пустой файл")
    # temp_path.write_bytes(content)
    # # Здесь можно добавить логику PanoramaProcessor
    # return {"result_url": f"/static/results/processed_{file.filename}"}

    # 1) Сохраняем во временную папку
    upload_dir = Path("temp_uploads")
    upload_dir.mkdir(exist_ok=True)
    temp_path = upload_dir / file.filename
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Пустой файл")
    temp_path.write_bytes(content)

    # 2) Обрабатываем панораму и получаем путь к результату
    processor = PanoramaProcessor()
    result_path_str = processor.process_image(str(temp_path))
    result_path = Path(result_path_str)

    if not result_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Процессор вернул несуществующий файл: {result_path}"
        )

    # 3) Формируем URL относительно /static
    #    (предположим, что он лежит в static/results)
    filename = result_path.name
    return {"result_url": f"/static/results/{filename}"}

@application.post('/api/predict', status_code=status.HTTP_201_CREATED)
async def predict_defect(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> list[dict[str, object]]:
    """
    Обрабатывает загруженный файл панорамы: режет на тайлы, отправляет каждый тайл в ML-сервис,
    собирает ответы и возвращает список результатов.

    Параметры:
        file (UploadFile): Загруженный файл панорамы.
        db (Session): Сессия базы данных для хранения записей (не используется на текущем этапе).

    Возвращает:
        list[dict]: Список словарей с полем 'status' и 'defects' для каждого тайла.
    """
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"message": "Файл должен быть изображением"}
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Пустой файл")

    # Декодируем изображение через OpenCV
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Не удалось прочитать изображение")

    # Нарезаем панораму на тайлы
    def slice_panorama(image: np.ndarray) -> list[np.ndarray]:
        """
        Возвращает список растров numpy, представляющих тайлы панорамы.

        Параметры:
            image (np.ndarray): Исходное изображение-панорама.

        Возвращает:
            list[np.ndarray]: Список разделённых тайлов.
        """
        size_map = {
            (31920, 1152): 28,
            (30780, 1152): 27,
            (18144, 1142): 16,
        }
        h, w = image.shape[:2]
        tiles_count = size_map.get((w, h))
        if tiles_count is None:
            raise HTTPException(
                status_code=400,
                detail=f"Неизвестный размер панорамы {w}×{h}"
            )
        tile_width = w // tiles_count
        return [image[:, i * tile_width:(i + 1) * tile_width] for i in range(tiles_count)]

    tiles = slice_panorama(img)
    combined_results: list[dict[str, object]] = []

    async with httpx.AsyncClient() as client:
        for tile in tiles:
            success, buf = cv2.imencode('.png', tile)
            if not success:
                continue
            files = {'file': ('tile.png', buf.tobytes(), 'image/png')}
            resp = await client.post(ML_SERVICE_DETECT_ENDPOINT, files=files)
            if resp.status_code != status.HTTP_201_CREATED:
                raise HTTPException(status_code=502, detail="Ошибка ML-сервиса")
            ml_json = resp.json()
            combined_results.append({
                "status": ml_json.get("status"),
                "defects": [
                    {"class": d['class'], "confidence": f"{d['confidence'] * 100:.2f}%"}
                    for d in ml_json.get("detections", [])
                ]
            })

    return combined_results

@application.get('/api/image/{id}', response_model=schemas.GetImage)
def get_image(id: int, db: Session = Depends(get_db)) -> schemas.GetImage:
    """
    Возвращает запись изображения из БД по его идентификатору.

    Параметры:
        id (int): Идентификатор записи изображения.
        db (Session): Сессия БД.

    Возвращает:
        GetImage: Pydantic-схема с данными изображения.
    """
    record = db.query(models.Images).filter(models.Images.id == id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    return record

@application.delete('/api/delete/image/{id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_image(id: int, db: Session = Depends(get_db)) -> None:
    """
    Удаляет запись изображения и связанные с ней детекции из БД.

    Параметры:
        id (int): Идентификатор записи изображения.
        db (Session): Сессия БД.
    """
    record = db.query(models.Images).filter(models.Images.id == id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    db.delete(record)
    db.commit()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="0.0.0.0", port=8000)

