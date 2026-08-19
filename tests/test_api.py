import io

from fastapi.testclient import TestClient
from PIL import Image

import main


client = TestClient(main.app)


def image_bytes(format_name="JPEG") -> bytes:
    image = Image.new("RGB", (32, 32), "red")
    buffer = io.BytesIO()
    image.save(buffer, format=format_name)
    return buffer.getvalue()


def rgba_png_bytes() -> bytes:
    image = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": main.MODEL_NAME}


def test_remove_background(monkeypatch):
    monkeypatch.setattr(main, "remove_background_bytes", lambda _, **__: rgba_png_bytes())

    response = client.post(
        "/api/remove-background",
        files={"file": ("photo.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).mode == "RGBA"


def test_uses_selected_model(monkeypatch):
    options = {}

    def fake_remove_background(_, **kwargs):
        options.update(kwargs)
        return rgba_png_bytes()

    monkeypatch.setattr(main, "remove_background_bytes", fake_remove_background)

    response = client.post(
        "/api/remove-background",
        data={"model_name": "u2netp"},
        files={"file": ("photo.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert options["model_name"] == "u2netp"


def test_rejects_unsupported_model():
    response = client.post(
        "/api/remove-background",
        data={"model_name": "not-a-model"},
        files={"file": ("photo.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 400


def test_rejects_unsupported_content_type():
    response = client.post(
        "/api/remove-background",
        files={"file": ("file.txt", b"text", "text/plain")},
    )

    assert response.status_code == 415


def test_rejects_empty_file():
    response = client.post(
        "/api/remove-background",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400


def test_rejects_file_over_size_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_FILE_SIZE", 4)

    response = client.post(
        "/api/remove-background",
        files={"file": ("photo.jpg", b"12345", "image/jpeg")},
    )

    assert response.status_code == 413


def test_rejects_image_over_pixel_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_IMAGE_PIXELS", 1)

    response = client.post(
        "/api/remove-background",
        files={"file": ("photo.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 400


def test_rejects_corrupted_image():
    response = client.post(
        "/api/remove-background",
        files={"file": ("photo.jpg", b"not a jpg", "image/jpeg")},
    )

    assert response.status_code == 400
