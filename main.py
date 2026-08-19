from __future__ import annotations

import asyncio
import io
import gc
import logging

from functools import lru_cache
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from rembg import new_session, remove # type: ignore
from PIL import Image, ImageOps

MODEL_NAME = "u2net"
SUPPORTED_MODELS = {"u2netp", "silueta", "u2net", "isnet-general-use"}
MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
READ_CHUNK_SIZE = 1024 * 1024
MAX_CONCURRENT_INFERENCE = 1 
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)


logger = logging.getLogger("background-removal")

def get_client_ip(request: Request) -> str:
    # Получение реального IP клиента
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

def check_image_signature(data: bytes) -> bool:
    # Проверка магических байтов (сигнатур) для защиты от подделки Content-Type.
    if data.startswith(b'\xff\xd8\xff'): return True  # JPEG
    if data.startswith(b'\x89PNG'): return True       # PNG
    if data.startswith(b'BM'): return True            # BMP
    if len(data) > 12 and data.startswith(b'RIFF') and data[8:12] == b'WEBP': return True
    return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_session(MODEL_NAME)
    logger.info("Model preloaded successfully.")
    yield


app = FastAPI(
    title="Background Removal Service",
    description="Удаление фона с фотографии и возврат PNG с прозрачностью.",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


app.add_middleware(SlowAPIMiddleware)


limiter = Limiter(key_func=get_client_ip)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Слишком много запросов. Лимит: {exc.detail}"}
    )


@lru_cache(maxsize=len(SUPPORTED_MODELS))
def get_session(model_name: str = MODEL_NAME):
    return new_session(model_name)


def remove_background_bytes(
    image_bytes: bytes,
    *,
    model_name: str = MODEL_NAME,
    alpha_matting: bool = False,
    foreground_threshold: int = 240,
    background_threshold: int = 10,
    post_process_mask: bool = False,
) -> bytes:
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load() 
        
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Разрешение изображения слишком большое")
        
        image = ImageOps.exif_transpose(image) # type: ignore

        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB") # type: ignore
            
    except ValueError:
        raise 
    except Exception as exc:
        raise ValueError("Файл не является корректным изображением") from exc


    result = remove(
        image,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=foreground_threshold,
        alpha_matting_background_threshold=background_threshold,
        post_process_mask=post_process_mask,
        session=get_session(model_name),
    )

    if isinstance(result, Image.Image):
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()
    elif isinstance(result, bytes):
        return result
    
    raise TypeError(f"Unexpected result type from rembg: {type(result)!r}")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/api/remove-background")
@limiter.limit("10/minute")
async def remove_background(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(MODEL_NAME),
    alpha_matting: bool = Form(False),
    foreground_threshold: int = Form(240),
    background_threshold: int = Form(10),
    post_process_mask: bool = Form(False),
) -> Response:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Поддерживаются только JPEG, PNG, WEBP и BMP")

    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Выбрана неподдерживаемая модель")

    if not 0 <= foreground_threshold <= 255 or not 0 <= background_threshold <= 255:
        raise HTTPException(status_code=400, detail="Пороги должны быть в диапазоне от 0 до 255")

    buffer = io.BytesIO()
    size = 0

    while chunk := await file.read(READ_CHUNK_SIZE):
        size += len(chunk)
        if size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Файл больше 15 MB")
        buffer.write(chunk)

    image_bytes = buffer.getvalue()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Файл пуст")

# ML-инференс ограничен одним запуском на процесс, чтобы входящие запросы не конкурировали за память во время выполнения ONNX-модели.
    try:
        async with inference_semaphore:
            result = await run_in_threadpool(
                remove_background_bytes,
                image_bytes,
                model_name=model_name,
                alpha_matting=alpha_matting,
                foreground_threshold=foreground_threshold,
                background_threshold=background_threshold,
                post_process_mask=post_process_mask,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Background removal failed")
        raise HTTPException(status_code=500, detail="Ошибка обработки изображения") from exc

    return Response(content=result, media_type="image/png")


app.mount(
    "/samples",
    StaticFiles(directory=Path(__file__).parent / "benchmark" / "images"),
    name="samples",
)
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="web",
)
