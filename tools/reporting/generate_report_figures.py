from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
REPORT_ASSETS = ROOT / "deliverables" / "informe_final" / "figuras_fuente"
OUT = ROOT / "deliverables" / "informe_final" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#17324d"
BLUE = "#2f75b5"
LIGHT_BLUE = "#dceaf7"
GREEN = "#4b9b4a"
LIGHT_GREEN = "#e4f2e4"
ORANGE = "#d46f3d"
YELLOW = "#d9ad2b"
RED = "#b94747"
GRAY = "#65717c"
LIGHT_GRAY = "#eef1f3"
DARK = "#1d2730"
WHITE = "#ffffff"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, size: tuple[int, int], background=WHITE) -> Image.Image:
    target_w, target_h = size
    source = image.convert("RGB")
    scale = min(target_w / source.width, target_h / source.height)
    resized = source.resize(
        (max(1, int(source.width * scale)), max(1, int(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, background)
    canvas.paste(resized, ((target_w - resized.width) // 2, (target_h - resized.height) // 2))
    return canvas


def cover_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    source = image.convert("RGB")
    scale = max(target_w / source.width, target_h / source.height)
    resized = source.resize(
        (max(1, int(source.width * scale)), max(1, int(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: str,
    outline: str = "#c7d0d8",
    radius: int = 18,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color=NAVY, width=5) -> None:
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    angle = np.arctan2(y2 - y1, x2 - x1)
    length = 16
    spread = 0.55
    p1 = (
        int(x2 - length * np.cos(angle - spread)),
        int(y2 - length * np.sin(angle - spread)),
    )
    p2 = (
        int(x2 - length * np.cos(angle + spread)),
        int(y2 - length * np.sin(angle + spread)),
    )
    draw.polygon([end, p1, p2], fill=color)


def centered_text(draw: ImageDraw.ImageDraw, xy, text, text_font, fill=NAVY) -> None:
    x1, y1, x2, y2 = xy
    bbox = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=5, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.multiline_text(
        ((x1 + x2 - width) / 2, (y1 + y2 - height) / 2),
        text,
        font=text_font,
        fill=fill,
        spacing=5,
        align="center",
    )


def save_jpeg(image: Image.Image, name: str, quality: int = 94) -> Path:
    path = OUT / name
    image.convert("RGB").save(path, quality=quality, subsampling=0)
    return path


def figure_c1() -> None:
    source = REPORT_ASSETS / "prueba_tecnica" / "centrado_original.jpg"
    image = Image.open(source).convert("RGB")
    canvas = Image.new("RGB", (1600, 980), WHITE)
    canvas.paste(cover_image(image, (1600, 900)), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 900, 1600, 980), fill=WHITE)
    draw.text((50, 922), "Vista posterior de un camión CAEX con carga visible", font=font(36, True), fill=NAVY)
    draw.text((1180, 930), "Imagen de faena", font=font(25), fill=GRAY)
    save_jpeg(canvas, "Figura_C1_contexto_captura.jpg")


def figure_c2() -> None:
    canvas = Image.new("RGB", (1800, 900), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 45), "Arquitectura general del sistema", font=font(48, True), fill=NAVY)
    draw.text(
        (72, 110),
        "Flujo desde el video de entrada hasta los resultados revisables",
        font=font(27),
        fill=GRAY,
    )

    boxes = [
        ((80, 285, 320, 470), "Video\nMP4", LIGHT_GRAY, NAVY),
        ((380, 285, 650, 470), "Extracción\nde frames", LIGHT_BLUE, BLUE),
        ((710, 285, 1010, 470), "YOLO\nsegmentación", LIGHT_GREEN, GREEN),
        ((1070, 285, 1310, 470), "Máscaras RGB\nCarga", "#eef8ee", GREEN),
        ((1370, 285, 1710, 470), "Analítico\ngeométrico", "#fff4e8", ORANGE),
    ]
    for xy, text, fill, outline in boxes:
        rounded_box(draw, xy, fill=fill, outline=outline, radius=20, width=3)
        centered_text(draw, xy, text, font(31, True), fill=NAVY)

    for left, right in zip(boxes[:-1], boxes[1:]):
        arrow(draw, (left[0][2] + 12, 377), (right[0][0] - 12, 377), color=NAVY, width=5)

    outputs = (1120, 610, 1710, 790)
    rounded_box(draw, outputs, fill="#f5f7f9", outline=NAVY, radius=20, width=3)
    centered_text(
        draw,
        outputs,
        "Resultados\nCSV · overlays · veredicto por video",
        font(30, True),
        fill=NAVY,
    )
    arrow(draw, (1540, 482), (1540, 595), color=NAVY, width=5)

    web = (380, 610, 940, 790)
    rounded_box(draw, web, fill="#e8f0f7", outline=BLUE, radius=20, width=3)
    centered_text(
        draw,
        web,
        "Aplicación web local\nsubida · ejecución · revisión · descarga",
        font(29, True),
        fill=NAVY,
    )
    arrow(draw, (955, 700), (1105, 700), color=BLUE, width=5)
    draw.text((515, 825), "Procesos locales", font=font(23, True), fill=BLUE)
    draw.text((790, 825), "Docker + GPU", font=font(23, True), fill=GREEN)
    draw.text((1115, 825), "Archivos intercambiados", font=font(23, True), fill=ORANGE)
    save_jpeg(canvas, "Figura_C2_arquitectura.jpg")


def figure_c3() -> None:
    image_path = REPORT_ASSETS / "cvat" / "imagen_ejemplo.jpg"
    mask_path = REPORT_ASSETS / "cvat" / "mascara_ejemplo.png"
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")
    image_np = np.asarray(image).astype(np.float32)
    mask_np = np.asarray(mask)
    selected = np.all(mask_np == np.array([102, 255, 102], dtype=np.uint8), axis=2)
    annotated = image_np.copy()
    green = np.array([102, 255, 102], dtype=np.float32)
    annotated[selected] = 0.52 * annotated[selected] + 0.48 * green
    annotated = Image.fromarray(np.clip(annotated, 0, 255).astype(np.uint8))

    panels = [image, annotated, mask]
    labels = ["Imagen original", "Anotación visual", "Máscara RGB exportada"]
    canvas = Image.new("RGB", (1800, 790), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 35), "Preparación de datos desde CVAT", font=font(46, True), fill=NAVY)
    draw.text((60, 95), "La clase Carga se identifica mediante el color RGB [102, 255, 102].", font=font(26), fill=GRAY)

    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = 50 + index * 585
        rounded_box(draw, (x, 155, x + 550, 720), fill="#f7f8f9", outline="#c7d0d8", radius=16, width=2)
        panel_img = cover_image(panel, (520, 455))
        canvas.paste(panel_img, (x + 15, 175))
        centered_text(draw, (x + 15, 640, x + 535, 705), label, font(27, True), fill=NAVY)
    save_jpeg(canvas, "Figura_C3_cvat_mascaras.jpg")


def rolling_mean(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def draw_chart(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, np.ndarray, str]],
    y_min: float,
    y_max: float,
    x_values: np.ndarray,
) -> None:
    x1, y1, x2, y2 = xy
    rounded_box(draw, xy, fill=WHITE, outline="#cbd3da", radius=16, width=2)
    draw.text((x1 + 28, y1 + 22), title, font=font(28, True), fill=NAVY)
    plot_left, plot_top = x1 + 75, y1 + 120
    plot_right, plot_bottom = x2 - 28, y2 - 58

    for fraction in np.linspace(0, 1, 5):
        y = int(plot_bottom - fraction * (plot_bottom - plot_top))
        value = y_min + fraction * (y_max - y_min)
        draw.line((plot_left, y, plot_right, y), fill="#e5e9ed", width=2)
        draw.text((x1 + 12, y - 12), f"{value:.1f}", font=font(17), fill=GRAY)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#697680", width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#697680", width=3)
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    for epoch in [0, 25, 50, 75, 100]:
        x = int(plot_left + (epoch - x_min) / max(1.0, x_max - x_min) * (plot_right - plot_left))
        draw.line((x, plot_bottom, x, plot_bottom + 7), fill="#697680", width=2)
        draw.text((x - 12, plot_bottom + 12), str(epoch), font=font(17), fill=GRAY)

    for _, values, color in series:
        clipped = np.clip(values, y_min, y_max)
        points = []
        for x_value, y_value in zip(x_values, clipped):
            x = int(plot_left + (x_value - x_min) / max(1.0, x_max - x_min) * (plot_right - plot_left))
            y = int(plot_bottom - (y_value - y_min) / max(1e-9, y_max - y_min) * (plot_bottom - plot_top))
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=5, joint="curve")

    legend_start_x = x2 - 365
    legend_start_y = y1 + 58
    legend_column_width = 180
    for index, (label, _, color) in enumerate(series):
        column = index % 2
        row = index // 2
        legend_x = legend_start_x + column * legend_column_width
        legend_y = legend_start_y + row * 30
        draw.line(
            (legend_x, legend_y + 10, legend_x + 28, legend_y + 10),
            fill=color,
            width=5,
        )
        draw.text(
            (legend_x + 38, legend_y - 2),
            label,
            font=font(18, True),
            fill=DARK,
        )


def figure_c4() -> None:
    csv_path = REPORT_ASSETS / "entrenamiento" / "results.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    def values(key: str) -> np.ndarray:
        return np.array([float(row[key]) for row in rows], dtype=float)

    epochs = values("epoch")
    train_seg = values("train/seg_loss")
    val_seg = values("val/seg_loss")
    precision = values("metrics/precision(M)")
    recall = values("metrics/recall(M)")
    map50 = values("metrics/mAP50(M)")
    map5095 = values("metrics/mAP50-95(M)")

    canvas = Image.new("RGB", (1800, 1050), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 30), "Entrenamiento y trazabilidad del checkpoint alternativo", font=font(43, True), fill=NAVY)
    draw.text(
        (55, 88),
        "Evolución durante 100 épocas; se conserva como respaldo del modelo principal",
        font=font(25),
        fill=GRAY,
    )

    draw_chart(
        draw,
        (45, 145, 880, 625),
        "Pérdida de segmentación",
        [
            ("Entrenamiento", rolling_mean(train_seg), BLUE),
            ("Validación", rolling_mean(np.minimum(val_seg, 8)), ORANGE),
        ],
        0.0,
        8.0,
        epochs,
    )
    draw_chart(
        draw,
        (920, 145, 1755, 625),
        "Métricas de máscara",
        [
            ("Precisión", rolling_mean(precision), BLUE),
            ("Recall", rolling_mean(recall), GREEN),
            ("mAP@0,50", rolling_mean(map50), ORANGE),
            ("mAP@0,50–0,95", rolling_mean(map5095), RED),
        ],
        0.0,
        1.0,
        epochs,
    )

    rounded_box(draw, (45, 665, 1755, 995), fill="#f5f7f9", outline="#cbd3da", radius=18, width=2)
    draw.text((80, 700), "Registro del experimento en MLflow", font=font(31, True), fill=NAVY)
    fields = [
        ("Experimento", "cono_carga_yolo_seg"),
        ("Arquitectura", "YOLOv8n-seg"),
        ("Datos", "180 entrenamiento\n36 validación · 24 prueba"),
        ("Configuración", "100 épocas · 640 px\nbatch 8"),
        ("Registro en MLflow", "parámetros · métricas\nbest.pt · last.pt"),
    ]
    starts = [80, 420, 735, 1080, 1405]
    for (label, value), x in zip(fields, starts):
        draw.text((x, 760), label, font=font(20), fill=GRAY)
        draw.multiline_text(
            (x, 800),
            value,
            font=font(23, True),
            fill=DARK,
            spacing=7,
        )
    save_jpeg(canvas, "Figura_C4_entrenamiento_mlflow.png")


def figure_c5() -> None:
    overlay_path = REPORT_ASSETS / "prueba_tecnica" / "cerca_umbral_overlay.jpg"
    overlay = Image.open(overlay_path).convert("RGB")
    canvas = Image.new("RGB", (1800, 930), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 30), "Elementos del analítico geométrico", font=font(45, True), fill=NAVY)
    image_panel = fit_image(overlay, (1320, 760), background="#f5f6f7")
    canvas.paste(image_panel, (35, 110))
    draw.rectangle((35, 110, 1355, 870), outline="#c7d0d8", width=3)

    rounded_box(draw, (1390, 110, 1765, 870), fill="#f5f7f9", outline="#c7d0d8", radius=18, width=2)
    draw.text((1430, 145), "Leyenda", font=font(34, True), fill=NAVY)
    entries = [
        ("Máscara de carga", "#66ff66", "Área visible segmentada"),
        ("Bounding box", "#d7d700", "Referencia geométrica interna"),
        ("Eje interno", "#28b9d8", "Centro horizontal del bbox"),
        ("Centroide", "#ff22cb", "Centroide de la máscara"),
        ("Perfil superior", "#e31a1c", "Contorno suavizado"),
        ("Ápice", "#31c94d", "Pico principal relativo"),
        ("Picos", "#ffffff", "Máximos diagnósticos"),
    ]
    y = 220
    for title, color, detail in entries:
        draw.ellipse((1430, y + 6, 1460, y + 36), fill=color, outline="#555555", width=2)
        draw.text((1480, y), title, font=font(25, True), fill=DARK)
        draw.text((1480, y + 34), detail, font=font(20), fill=GRAY)
        y += 86
    save_jpeg(canvas, "Figura_C5_elementos_geometricos.jpg")


def figure_c6() -> None:
    canvas = Image.new("RGB", (1800, 1050), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 35), "Secuencia de ejecución del pipeline", font=font(47, True), fill=NAVY)
    draw.text((60, 95), "Procesos locales, contenedor de percepción y archivos intercambiados", font=font(27), fill=GRAY)

    lanes = [
        ("Usuario", 150, LIGHT_GRAY, NAVY),
        ("Web app", 450, LIGHT_BLUE, BLUE),
        ("Orquestador local", 780, "#eef2f5", NAVY),
        ("Docker · YOLO", 1130, LIGHT_GREEN, GREEN),
        ("Analítico", 1450, "#fff4e8", ORANGE),
        ("Resultados", 1680, "#f4f5f6", GRAY),
    ]
    for label, x, fill, outline in lanes:
        rounded_box(draw, (x - 105, 155, x + 105, 225), fill=fill, outline=outline, radius=15, width=3)
        centered_text(draw, (x - 100, 160, x + 100, 220), label, font(23, True), fill=NAVY)
        draw.line((x, 225, x, 985), fill="#aab4bd", width=2)

    steps = [
        (280, 150, 450, "1. Carga video"),
        (365, 450, 780, "2. Solicita corrida"),
        (450, 780, 780, "3. Extrae frames"),
        (535, 780, 1130, "4. Ejecuta inferencia"),
        (620, 1130, 780, "5. Escribe máscaras RGB"),
        (705, 780, 1450, "6. Ejecuta analítico"),
        (790, 1450, 1680, "7. CSV + overlays"),
        (875, 1680, 450, "8. Veredicto y estadísticas"),
        (960, 450, 150, "9. Revisión y descarga"),
    ]
    for y, source_x, target_x, label in steps:
        arrow(draw, (source_x, y), (target_x, y), color=NAVY if source_x < target_x else BLUE, width=4)
        mid_x = (source_x + target_x) // 2
        label_box = draw.textbbox((0, 0), label, font=font(22, True))
        label_w = label_box[2] - label_box[0]
        draw.rectangle((mid_x - label_w // 2 - 9, y - 35, mid_x + label_w // 2 + 9, y - 5), fill=WHITE)
        draw.text((mid_x - label_w // 2, y - 33), label, font=font(22, True), fill=DARK)
    save_jpeg(canvas, "Figura_C6_secuencia_pipeline.jpg")


def figure_c7() -> None:
    overlay_paths = [
        REPORT_ASSETS / "prueba_tecnica" / "izquierda_secundaria_overlay.jpg",
        REPORT_ASSETS / "prueba_tecnica" / "centrado_overlay.jpg",
        REPORT_ASSETS / "prueba_tecnica" / "derecha_overlay.jpg",
    ]
    canvas = Image.new("RGB", (1800, 1120), "#0f141a")
    draw = ImageDraw.Draw(canvas)
    draw.text((45, 30), "Cono de Carga", font=font(50, True), fill=WHITE)
    draw.text((45, 95), "Camera_8_20240907_160001.mp4", font=font(29, True), fill="#cbd5df")

    rounded_box(draw, (45, 145, 1755, 310), fill="#16212d", outline="#33485a", radius=16, width=2)
    draw.text((70, 165), "Consola", font=font(24, True), fill=WHITE)
    console_lines = [
        "[1/1] Extrayendo frames: 11 guardados",
        "YOLO: 11 máscaras generadas con GPU",
        "Analítico: 10/11 frames válidos",
        "Pipeline completado · resultados disponibles",
    ]
    y = 205
    for line in console_lines:
        draw.text((75, y), line, font=font(20), fill="#b9d8f2")
        y += 27

    draw.text((45, 345), "Overlays", font=font(31, True), fill=WHITE)
    for index, path in enumerate(overlay_paths):
        x = 45 + index * 575
        image = cover_image(Image.open(path), (540, 304))
        canvas.paste(image, (x, 395))
        draw.rounded_rectangle((x, 395, x + 540, 699), radius=12, outline="#3b4a57", width=3)

    draw.text((45, 735), "Veredicto", font=font(31, True), fill=WHITE)
    cards = [
        ("Resultado", "Tendencia izquierda", "#d46f3d"),
        ("Confianza", "0,80", "#2f75b5"),
        ("Frames válidos", "10 de 11", "#4b9b4a"),
        ("Tiempo", "20,2 s", "#65717c"),
    ]
    for index, (title, value, accent) in enumerate(cards):
        x = 45 + index * 425
        rounded_box(draw, (x, 785, x + 390, 955), fill="#1a252e", outline="#384955", radius=15, width=2)
        draw.rectangle((x, 785, x + 390, 793), fill=accent)
        draw.text((x + 24, 815), title, font=font(21, True), fill="#aab8c5")
        draw.text((x + 24, 862), value, font=font(31, True), fill=WHITE)

    draw.text((45, 995), "Descargas", font=font(27, True), fill=WHITE)
    buttons = ["video_summary.csv", "features.csv", "pipeline_summary.json"]
    for index, label in enumerate(buttons):
        x = 250 + index * 455
        rounded_box(draw, (x, 985, x + 410, 1060), fill="#1d2b36", outline="#557083", radius=12, width=2)
        centered_text(draw, (x, 985, x + 410, 1060), label, font(22, True), fill="#dbe8f2")
    save_jpeg(canvas, "Figura_C7_web_app.jpg")


def figure_c8() -> None:
    canvas = Image.new("RGB", (1800, 1620), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 32), "Prueba técnica – Cono de Carga", font=font(47, True), fill=NAVY)
    draw.text(
        (60, 94),
        "Comparación entre selección visual y resultado del pipeline",
        font=font(25),
        fill=GRAY,
    )

    cards = [
        ("Imágenes", "90", DARK),
        ("Evaluables", "100,0 %", BLUE),
        ("Acierto total", "88,9 %", GREEN),
        ("Acierto evaluables", "88,9 %", GREEN),
    ]
    for index, (label, value, color) in enumerate(cards):
        x1 = 60 + index * 425
        rounded_box(
            draw,
            (x1, 145, x1 + 395, 290),
            fill="#f7f8f9",
            outline="#cbd3da",
            radius=13,
            width=2,
        )
        draw.text((x1 + 24, 170), label, font=font(21, True), fill=GRAY)
        draw.text((x1 + 24, 213), value, font=font(38, True), fill=color)

    draw.text((60, 340), "Matriz de confusión", font=font(31, True), fill=NAVY)
    draw.text(
        (60, 388),
        "Filas: clase esperada | Columnas: predicción",
        font=font(19),
        fill=GRAY,
    )

    matrix = [
        ("Centrado", [22, 6, 2, 0]),
        ("Izquierda", [2, 28, 0, 0]),
        ("Derecha", [0, 0, 30, 0]),
    ]
    column_labels = ["Centrado", "Izquierda", "Derecha", "No evaluable"]
    matrix_left = 260
    matrix_top = 470
    cell_width = 190
    cell_height = 100

    for column_index, label in enumerate(column_labels):
        centered_text(
            draw,
            (
                matrix_left + column_index * cell_width,
                420,
                matrix_left + (column_index + 1) * cell_width,
                465,
            ),
            label,
            font(18, True),
            fill=GRAY,
        )

    for row_index, (row_label, values_row) in enumerate(matrix):
        row_y = matrix_top + row_index * cell_height
        draw.text((60, row_y + 33), row_label, font=font(22, True), fill=DARK)
        for column_index, value in enumerate(values_row):
            cell_x = matrix_left + column_index * cell_width
            if row_index == column_index:
                fill = "#b8dfbf"
            elif value > 0:
                fill = "#f8e5e9"
            else:
                fill = WHITE
            draw.rectangle(
                (
                    cell_x,
                    row_y,
                    cell_x + cell_width,
                    row_y + cell_height,
                ),
                fill=fill,
                outline="#d4d9de",
                width=2,
            )
            centered_text(
                draw,
                (
                    cell_x,
                    row_y,
                    cell_x + cell_width,
                    row_y + cell_height,
                ),
                str(value),
                font(31, True),
                fill=DARK,
            )

    draw.text((1210, 340), "Acierto por clase", font=font(31, True), fill=NAVY)
    chart_left = 1200
    chart_right = 1710
    chart_top = 455
    chart_bottom = 770
    draw.line(
        (chart_left, chart_bottom, chart_right, chart_bottom),
        fill="#596570",
        width=3,
    )
    class_scores = [
        ("Centrado", 73.3),
        ("Izquierda", 93.3),
        ("Derecha", 100.0),
    ]
    bar_width = 105
    bar_gap = 70
    first_bar_x = 1240
    for index, (label, score) in enumerate(class_scores):
        x1 = first_bar_x + index * (bar_width + bar_gap)
        x2 = x1 + bar_width
        y1 = int(chart_bottom - (score / 100) * (chart_bottom - chart_top))
        draw.rounded_rectangle(
            (x1, y1, x2, chart_bottom),
            radius=8,
            fill=GREEN,
        )
        score_text = f"{score:.1f} %".replace(".", ",")
        centered_text(
            draw,
            (x1 - 25, y1 - 48, x2 + 25, y1 - 6),
            score_text,
            font(20, True),
            fill=DARK,
        )
        centered_text(
            draw,
            (x1 - 35, chart_bottom + 8, x2 + 35, chart_bottom + 55),
            label,
            font(18, True),
            fill=GRAY,
        )

    rounded_box(
        draw,
        (60, 850, 1740, 1060),
        fill="#f7f8f9",
        outline="#cbd3da",
        radius=13,
        width=2,
    )
    draw.text((90, 875), "Resumen por clase", font=font(26, True), fill=NAVY)
    summaries = [
        "Centrado: 22/30 correctas | 30/30 evaluables",
        "Izquierda: 28/30 correctas | 30/30 evaluables",
        "Derecha: 30/30 correctas | 30/30 evaluables",
    ]
    for index, text in enumerate(summaries):
        draw.text((100, 925 + index * 38), text, font=font(21, True), fill=DARK)
    draw.text(
        (1010, 1000),
        "El acierto total considera las imágenes no evaluables como error.",
        font=font(18),
        fill=RED,
    )

    draw.text(
        (60, 1100),
        "Indicadores geométricos por clase esperada",
        font=font(31, True),
        fill=NAVY,
    )
    draw.text(
        (60, 1145),
        "Cada valor corresponde a la mediana de los frames válidos de la clase.",
        font=font(19),
        fill=GRAY,
    )

    table_left = 60
    table_top = 1190
    column_widths = [270, 450, 480, 480]
    table_headers = [
        "Clase esperada",
        "Pico mediano",
        "Simetría mediana",
        "Centroide mediano",
    ]
    table_rows = [
        ("Centrado", "−0,004", "0,024", "0,007"),
        ("Izquierda", "−0,138", "−0,179", "−0,051"),
        ("Derecha", "0,109", "0,172", "0,050"),
    ]
    row_height = 70
    current_x = table_left
    for header, width in zip(table_headers, column_widths):
        draw.rectangle(
            (current_x, table_top, current_x + width, table_top + row_height),
            fill="#e9eef2",
            outline="#cbd3da",
            width=2,
        )
        draw.text(
            (current_x + 18, table_top + 23),
            header,
            font=font(20, True),
            fill=NAVY,
        )
        current_x += width

    for row_index, row in enumerate(table_rows):
        row_y = table_top + (row_index + 1) * row_height
        current_x = table_left
        for column_index, (value, width) in enumerate(zip(row, column_widths)):
            draw.rectangle(
                (current_x, row_y, current_x + width, row_y + row_height),
                fill=WHITE if row_index % 2 == 0 else "#f7f8f9",
                outline="#d4d9de",
                width=2,
            )
            draw.text(
                (current_x + 18, row_y + 23),
                value,
                font=font(20, column_index == 0),
                fill=DARK,
            )
            current_x += width

    draw.text(
        (60, 1488),
        "Referencias: pico < −0,06 izquierda y > 0,06 derecha | simetría < −0,07 izquierda y > 0,07 derecha.",
        font=font(18),
        fill=GRAY,
    )
    draw.text(
        (60, 1522),
        "El centroide relativo es descriptivo: negativo izquierda, cero centro, positivo derecha.",
        font=font(18),
        fill=GRAY,
    )
    canvas.save(OUT / "Figura_C8_prueba_tecnica.png")


def figure_c9() -> None:
    base = REPORT_ASSETS / "prueba_tecnica"
    examples = [
        (
            "IZQUIERDA",
            base / "izquierda_original.jpg",
            base / "izquierda_overlay.jpg",
            ORANGE,
            (236, 100, 958, 426),
        ),
        (
            "CENTRADO",
            base / "centrado_original.jpg",
            base / "centrado_overlay.jpg",
            GREEN,
            (324, 44, 954, 386),
        ),
        (
            "DERECHA",
            base / "derecha_original.jpg",
            base / "derecha_overlay.jpg",
            YELLOW,
            (214, 68, 948, 466),
        ),
    ]

    def crop_load(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
        xmin, ymin, xmax, ymax = bbox
        margin_x = max(70, int((xmax - xmin) * 0.10))
        margin_y = max(55, int((ymax - ymin) * 0.15))
        crop_box = (
            max(0, xmin - margin_x),
            max(0, ymin - margin_y),
            min(image.width, xmax + margin_x),
            min(image.height, ymax + margin_y),
        )
        return image.crop(crop_box)

    canvas = Image.new("RGB", (1800, 1430), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 28), "Comparación entre imagen original y resultado del analítico", font=font(43, True), fill=NAVY)
    draw.text((365, 100), "Imagen original", font=font(27, True), fill=GRAY)
    draw.text((1135, 100), "Overlay del pipeline", font=font(27, True), fill=GRAY)

    panel_size = (750, 355)
    for index, (label, original_path, overlay_path, color, bbox) in enumerate(examples):
        y = 150 + index * 420
        draw.rounded_rectangle(
            (35, y, 1765, y + 390),
            radius=12,
            fill="#f7f8f9",
            outline="#c9d0d7",
            width=2,
        )
        draw.rectangle((35, y, 49, y + 390), fill=color)
        draw.rounded_rectangle((65, y + 145, 190, y + 220), radius=8, fill=color)
        label_box = draw.textbbox((0, 0), label, font=font(21, True))
        label_width = label_box[2] - label_box[0]
        draw.text(
            (127 - label_width / 2, y + 169),
            label,
            font=font(21, True),
            fill=WHITE if color != YELLOW else NAVY,
        )

        original_source = crop_load(Image.open(original_path).convert("RGB"), bbox)
        overlay_source = crop_load(Image.open(overlay_path).convert("RGB"), bbox)
        original = cover_image(original_source, panel_size)
        overlay = cover_image(overlay_source, panel_size)
        canvas.paste(original, (220, y + 17))
        canvas.paste(overlay, (990, y + 17))

    save_jpeg(canvas, "Figura_C9_comparaciones_clases.jpg")


def figure_c10() -> None:
    total = 1248
    evaluable = 1183
    results = [
        ("Izquierda", 177, ORANGE),
        ("Centrado", 889, GREEN),
        ("Derecha", 117, BLUE),
        ("No evaluable", 65, "#8a9096"),
    ]

    canvas = Image.new("RGB", (1800, 1050), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (55, 30),
        "Distribución de resultados de la muestra operacional",
        font=font(45, True),
        fill=NAVY,
    )
    draw.text(
        (55, 92),
        "Clasificación de la forma visible de la carga en 1.248 imágenes",
        font=font(25),
        fill=GRAY,
    )

    cards = [
        ("Imágenes analizadas", "1.248", "Total procesado"),
        ("Imágenes evaluables", "1.183", "94,8 % de la muestra"),
        ("No evaluables", "65", "5,2 % de la muestra"),
    ]
    card_width = 540
    for index, (label, value, detail) in enumerate(cards):
        x1 = 55 + index * 575
        x2 = x1 + card_width
        rounded_box(
            draw,
            (x1, 150, x2, 300),
            fill="#f5f7f9",
            outline="#cbd3da",
            radius=15,
            width=2,
        )
        draw.text((x1 + 25, 172), label, font=font(22, True), fill=NAVY)
        draw.text((x1 + 25, 208), value, font=font(40, True), fill=DARK)
        draw.text((x1 + 170, 229), detail, font=font(20), fill=GRAY)

    rounded_box(
        draw,
        (55, 335, 1745, 995),
        fill=WHITE,
        outline="#cbd3da",
        radius=15,
        width=2,
    )
    draw.text((85, 365), "Resultado", font=font(23, True), fill=NAVY)
    draw.text(
        (360, 365),
        "Proporción sobre el total de imágenes",
        font=font(23, True),
        fill=NAVY,
    )

    plot_left = 360
    plot_right = 1450
    plot_width = plot_right - plot_left
    axis_y = 425
    for percent in (0, 25, 50, 75, 100):
        x = int(plot_left + plot_width * percent / 100)
        draw.line((x, axis_y, x, 945), fill="#e1e6ea", width=2)
        label = f"{percent} %"
        label_box = draw.textbbox((0, 0), label, font=font(18))
        draw.text(
            (x - (label_box[2] - label_box[0]) / 2, 397),
            label,
            font=font(18),
            fill=GRAY,
        )

    for index, (label, count, color) in enumerate(results):
        center_y = 490 + index * 125
        percentage = count / total * 100
        draw.text((85, center_y - 18), label, font=font(27, True), fill=DARK)

        track = (plot_left, center_y - 30, plot_right, center_y + 30)
        draw.rounded_rectangle(track, radius=12, fill=LIGHT_GRAY)
        bar_right = int(plot_left + plot_width * percentage / 100)
        draw.rounded_rectangle(
            (plot_left, center_y - 30, max(plot_left + 12, bar_right), center_y + 30),
            radius=12,
            fill=color,
        )

        percent_label = f"{percentage:.1f} %".replace(".", ",")
        draw.text(
            (bar_right + 18, center_y - 16),
            percent_label,
            font=font(24, True),
            fill=DARK,
        )
        draw.text(
            (1545, center_y - 29),
            f"{count}",
            font=font(31, True),
            fill=DARK,
        )
        draw.text(
            (1545, center_y + 8),
            "imágenes",
            font=font(19),
            fill=GRAY,
        )

    canvas.save(OUT / "Figura_C10_distribucion_1248.png")


def figure_c11() -> None:
    incomplete = REPORT_ASSETS / "pipeline" / "mascara_incompleta_overlay.jpg"
    border = REPORT_ASSETS / "muestra_1248" / "carga_truncada_overlay.jpg"
    empty = REPORT_ASSETS / "muestra_1248" / "sin_deteccion_overlay.jpg"
    threshold = REPORT_ASSETS / "prueba_tecnica" / "cerca_umbral_overlay.jpg"
    cases = [
        (
            "Máscara incompleta y contaminada",
            "Omisión, fuga exterior y contacto con el borde",
            incomplete,
        ),
        ("Carga truncada", "La máscara toca el borde de la imagen", border),
        ("Sin detección", "Máscara vacía producida por percepción", empty),
        ("Cerca del umbral", "Señales geométricas de baja magnitud", threshold),
    ]

    canvas = Image.new("RGB", (1800, 1120), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((55, 30), "Casos de error y condiciones no evaluables", font=font(45, True), fill=NAVY)
    for index, (title, subtitle, path) in enumerate(cases):
        col = index % 2
        row = index // 2
        x = 45 + col * 880
        y = 115 + row * 490
        rounded_box(draw, (x, y, x + 835, y + 455), fill="#f7f8f9", outline="#ccd3d9", radius=15, width=2)
        source = Image.open(path).convert("RGB")
        if title == "Máscara incompleta y contaminada":
            scale = 805 / source.width
            resized = source.resize(
                (805, max(350, int(source.height * scale))),
                Image.Resampling.LANCZOS,
            )
            panel = resized.crop(
                (0, resized.height - 350, 805, resized.height)
            )
            panel_draw = ImageDraw.Draw(panel)
            panel_draw.rounded_rectangle(
                (48, 18, 170, 338),
                radius=14,
                outline=RED,
                width=7,
            )
            panel_draw.rounded_rectangle(
                (180, 26, 405, 76),
                radius=9,
                fill=RED,
            )
            centered_text(
                panel_draw,
                (180, 26, 405, 76),
                "OMISIÓN Y FUGA",
                font(20, True),
                fill=WHITE,
            )
            arrow(panel_draw, (180, 52), (155, 90), color=RED, width=6)
            panel_draw.rounded_rectangle(
                (62, 302, 525, 348),
                radius=10,
                outline=RED,
                width=7,
            )
            panel_draw.rounded_rectangle(
                (520, 282, 798, 332),
                radius=9,
                fill=RED,
            )
            centered_text(
                panel_draw,
                (520, 282, 798, 332),
                "CONTACTO CON BORDE",
                font(19, True),
                fill=WHITE,
            )
            arrow(panel_draw, (535, 315), (510, 330), color=RED, width=6)
        else:
            panel = cover_image(source, (805, 350))
        canvas.paste(panel, (x + 15, y + 15))
        draw.text((x + 22, y + 378), title, font=font(27, True), fill=RED)
        draw.text((x + 22, y + 415), subtitle, font=font(21), fill=GRAY)
    save_jpeg(canvas, "Figura_C11_casos_error.jpg")


def build_contact_sheet() -> None:
    paths = sorted(OUT.glob("Figura_C*.*"))
    thumb_size = (520, 330)
    margin = 24
    label_height = 42
    columns = 3
    rows = int(np.ceil(len(paths) / columns))
    canvas = Image.new(
        "RGB",
        (
            margin + columns * (thumb_size[0] + margin),
            margin + rows * (thumb_size[1] + label_height + margin),
        ),
        WHITE,
    )
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths):
        column = index % columns
        row = index // columns
        x = margin + column * (thumb_size[0] + margin)
        y = margin + row * (thumb_size[1] + label_height + margin)
        image = fit_image(Image.open(path), thumb_size, background="#f1f3f5")
        canvas.paste(image, (x, y))
        draw.text((x, y + thumb_size[1] + 7), path.stem, font=font(18, True), fill=NAVY)
    canvas.save(OUT / "_contact_sheet.jpg", quality=90)


def main() -> None:
    builders = [
        figure_c1,
        figure_c2,
        figure_c3,
        figure_c4,
        figure_c5,
        figure_c6,
        figure_c7,
        figure_c8,
        figure_c9,
        figure_c10,
        figure_c11,
    ]
    for builder in builders:
        builder()
        print(f"OK {builder.__name__}")
    print(f"Figuras generadas en: {OUT}")


if __name__ == "__main__":
    main()
