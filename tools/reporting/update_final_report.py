from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.shared import Inches


ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "deliverables"
    / "informe_final"
    / "Parte_Cono_de_Carga_Informe_Final.docx"
)
TRAINING_FIGURE = (
    ROOT
    / "deliverables"
    / "informe_final"
    / "figuras"
    / "Figura_C4_entrenamiento_mlflow.png"
)
REPORT_FIGURES = {
    "image2.jpg": (
        ROOT
        / "deliverables"
        / "informe_final"
        / "figuras"
        / "Figura_C2_arquitectura.jpg"
    ),
    "image4.png": TRAINING_FIGURE,
    "image6.jpg": (
        ROOT
        / "deliverables"
        / "informe_final"
        / "figuras"
        / "Figura_C6_secuencia_pipeline.jpg"
    ),
    "image7.jpg": (
        ROOT
        / "deliverables"
        / "informe_final"
        / "figuras"
        / "Figura_C7_web_app.jpg"
    ),
}


PARAGRAPH_REPLACEMENTS = {
    "UNIVERSIDAD DE CONCEPCIÓN\nFACULTAD DE INGENIERÍA": (
        "UNIVERSIDAD DE CONCEPCIÓN\n"
        "FACULTAD DE INGENIERÍA\n"
        "DEPARTAMENTO DE INGENIERÍA INFORMÁTICA Y CS. DE LA COMPUTACIÓN\n"
        "CARRERA DE INGENIERÍA CIVIL INFORMÁTICA\n"
    ),
    "La Memoria se desarrolla en la Facultad de Ingeniería": (
        "La Memoria se desarrolla en la carrera de Ingeniería Civil Informática "
        "de la Facultad de Ingeniería de la Universidad de Concepción, "
        "institución académica responsable de la formación y evaluación del "
        "trabajo. PSINet participa como contraparte técnica a través de su "
        "departamento de I+D y vincula el desarrollo con una aplicación "
        "industrial de visión computacional en minería. Los registros utilizados "
        "provienen de la operación de Compañía Minera Doña Inés de Collahuasi, "
        "ubicada en la Región de Tarapacá. Collahuasi se considera origen de los "
        "datos y no institución colaboradora formal de la Memoria. El permiso y "
        "la realización del trabajo fueron consultados mediante PSINet en "
        "reuniones con gerencias de ambas organizaciones [15]-[17]."
    ),
    "La implementación integra Python, OpenCV": (
        "La implementación integra Python, OpenCV, NumPy, SciPy, Pandas, "
        "Ultralytics, NVIDIA CUDA, MLflow y Streamlit. Docker encapsula el "
        "entrenamiento, mientras la inferencia y el pipeline se ejecutan en un "
        "entorno Python local. El repositorio "
        "de percepción registra un entrenamiento reproducible con 240 imágenes "
        "y conserva su best.pt como checkpoint alternativo. Para la operación "
        "se seleccionó el modelo principal PSINet. En una prueba técnica "
        "balanceada de 90 imágenes, el pipeline obtiene 88,89 % de exactitud "
        "global: 73,33 % para centrado, 93,33 % para izquierda y 100 % para "
        "derecha. Las medianas de pico, simetría y centroide son cercanas a "
        "cero para centrado, negativas para izquierda y positivas para derecha, "
        "lo que aporta una explicación geométrica consistente. Una muestra "
        "descriptiva de 1.248 imágenes contiene 1.183 casos evaluables y 65 no "
        "evaluables. Los resultados confirman que la segmentación condiciona la "
        "clasificación. Las salidas representan asimetría visual interna, no "
        "balance físico, distribución real del material ni desplazamiento "
        "respecto de la tolva."
    ),
    "The implementation combines Python, OpenCV": (
        "The implementation combines Python, OpenCV, NumPy, SciPy, Pandas, "
        "Ultralytics, NVIDIA CUDA, MLflow, and Streamlit. Docker encapsulates "
        "training, while inference and the operational pipeline run in a local "
        "Python environment. The repository "
        "records a reproducible training run with 240 images and retains its "
        "best.pt as an alternative checkpoint, while PSINet is used as the "
        "primary operational segmenter. In a balanced technical evaluation with "
        "90 visually selected images, the complete pipeline reaches 88.89% "
        "overall accuracy. Median apex, area-symmetry, and centroid indicators "
        "remain close to zero for centered loads, negative for left cases, and "
        "positive for right cases. A separate descriptive sample of 1,248 images "
        "contains 1,183 evaluable and 65 non-evaluable cases."
    ),
    "5.3 Entrenamiento reproducible": "5.3 Entrenamiento reproducible del checkpoint alternativo",
    "El entrenamiento parte desde la definición yolov8n-seg.yaml": (
        "El entrenamiento reproducible parte desde yolov8n-seg.yaml, sin pesos "
        "preentrenados. El uso de pretrained=false demuestra que el repositorio "
        "entrena un segmentador y no se limita a recibir pesos terminados. "
        "MLflow registra la configuración, el conjunto, los parámetros, las "
        "métricas y los artefactos best.pt y last.pt en el servidor definido "
        "mediante .env.mlflow [11]. Tras la evaluación extremo a extremo, "
        "best.pt se conserva como alternativa y PSINet queda como modelo "
        "principal de segmentación."
    ),
    "Tabla C.5. Configuración del entrenamiento.": (
        "Tabla C.5. Configuración del entrenamiento del checkpoint alternativo."
    ),
    "Figura C.4. Seguimiento del entrenamiento": (
        "Figura C.4. Seguimiento y trazabilidad del entrenamiento del checkpoint alternativo."
    ),
    "5.4 Resultados del modelo de percepción": "5.4 Modelo de percepción operativo y referencia de entrenamiento",
    "El mejor punto del entrenamiento alcanza un mAP": (
        "En el entrenamiento del checkpoint alternativo, el mejor punto alcanza "
        "un mAP de máscara entre IoU 0,50 y 0,95 de 0,94142 en la época 87. Al "
        "finalizar se registra mAP50 de máscara igual a 0,995 y mAP50-95 igual "
        "a 0,91647. Estas métricas muestran buen ajuste sobre la validación "
        "disponible, pero las 36 imágenes y su posible relación temporal impiden "
        "interpretarlas como una estimación definitiva de generalización. La "
        "selección operativa se basó además en el desempeño del pipeline "
        "completo, donde PSINet obtuvo los mejores resultados."
    ),
    "El modelo fue entrenado para identificar la carga visible": (
        "El pipeline utiliza como segmentador principal "
        "perception/models/primary/psinet_carga_visible.pt y selecciona su "
        "clase material (class_id=1). El checkpoint "
        "perception/models/alternative/best.pt se mantiene para respaldo y "
        "comparaciones controladas con su clase Carga (class_id=0). Durante la "
        "inferencia se aplica una confianza de 0,35 y el modo largest conserva "
        "la instancia de carga de mayor área. La predicción se rasteriza como "
        "una única máscara RGB, contrato de entrada del analítico geométrico."
    ),
    "Los umbrales se mantienen en el archivo de configuración": (
        "Los umbrales se mantienen en el archivo de configuración para "
        "conservar trazabilidad y permitir recalibración ante nuevas cámaras o "
        "condiciones. El analítico interpreta la máscara entregada por el modelo "
        "principal de percepción y no intenta reconstruir regiones omitidas ni "
        "corregir contornos erróneos."
    ),
    "La inferencia de YOLO utiliza el modelo entrenado solo cuando el contenedor Docker está disponible||La inferencia operacional de YOLO se ejecuta": (
        "La inferencia operacional de YOLO se ejecuta con PyTorch y Ultralytics "
        "en el entorno Python local del proyecto. El orquestador invoca el "
        "adaptador de máscaras mediante el mismo intérprete que usa la "
        "aplicación web, por lo que no requiere Docker ni conversiones de rutas "
        "durante el análisis. Docker se reserva para el entrenamiento y la "
        "validación reproducibles con GPU."
    ),
    "La evaluación se organiza en tres niveles.||La evaluación considera tres niveles:": (
        "La evaluación considera tres niveles: métricas internas del "
        "entrenamiento; una prueba técnica balanceada de 90 imágenes, con 30 "
        "casos visuales por clase, para medir la clasificación de extremo a "
        "extremo; y una muestra de 1.248 imágenes para estudiar el "
        "comportamiento operacional, los descartes y la distribución de salidas "
        "en una cantidad mayor de escenarios."
    ),
    "Las etiquetas esperadas de la prueba técnica||Las 90 imágenes proceden": (
        "Las 90 imágenes proceden del banco de evidencias CAEX del proyecto de "
        "Granulometría (data/raw/evidencias_caex/images), una base distinta del "
        "entrenamiento y formada por registros de fechas posteriores. Se "
        "seleccionaron visualmente 30 casos centrados, 30 a izquierda y 30 a "
        "derecha. Las etiquetas representan apariencia visual, no mediciones "
        "físicas. La separación temporal reduce el riesgo de reutilizar imágenes "
        "de entrenamiento, aunque el tamaño y la selección dirigida mantienen el "
        "resultado como validación técnica local."
    ),
    "El modelo de percepción entrenado se evalúa junto": (
        "El modelo principal PSINet se evalúa junto con el analítico sobre las "
        "90 imágenes de la prueba técnica. Todas resultan evaluables y 80 se "
        "clasifican correctamente, equivalentes a 88,89 % de exactitud global. "
        "Este valor representa el mejor resultado extremo a extremo obtenido y "
        "corresponde a la configuración utilizada por el pipeline."
    ),
    "Fuente: elaboración propia a partir de la prueba técnica": (
        "Fuente: elaboración propia a partir de la prueba técnica balanceada de "
        "90 imágenes procesada con el modelo principal de percepción."
    ),
    "La muestra operacional se procesa con confianza 0,35": (
        "Las 1.248 imágenes también proceden del banco de evidencias CAEX del "
        "proyecto de Granulometría (data/raw/evidencias_caex/images). Se "
        "seleccionaron para observar el comportamiento del pipeline sobre un "
        "volumen mayor y obtener una referencia descriptiva de cómo se "
        "distribuyen las cargas en diversos escenarios. La muestra se procesa "
        "con confianza 0,35 y selección de la instancia de mayor área. Se "
        "obtienen 1.183 imágenes válidas y 65 no evaluables. De estas últimas, "
        "42 corresponden a máscaras vacías y 23 a máscaras que tocan algún "
        "borde de la imagen. El descarte por borde evita que una carga "
        "incompleta genere picos falsos o una simetría artificial."
    ),
    "La predominancia de centrado describe": (
        "La predominancia de centrado describe las salidas del sistema sobre "
        "esta muestra amplia y permite observar su distribución en distintos "
        "escenarios. No constituye una medición de exactitud ni, por sí sola, "
        "una evaluación de la calidad del carguío, porque las 1.248 imágenes no "
        "poseen una etiqueta esperada independiente para cada caso y pueden "
        "contener múltiples fotogramas del mismo evento."
    ),
    "Tabla C.9. Métricas por clase del modelo de percepción.": (
        "Tabla C.9. Métricas de clasificación por clase."
    ),
    "La prueba técnica confirma que la combinación": (
        "La prueba técnica confirma que la combinación de segmentación y "
        "geometría puede discriminar tendencias visuales laterales en el "
        "conjunto evaluado. El modelo principal PSINet, integrado como "
        "segmentador operativo, alcanza 88,89 % de exactitud global. La revisión "
        "de errores muestra que la percepción sigue siendo un factor limitante, "
        "pues las reglas geométricas no recuperan regiones omitidas ni remueven "
        "regiones incorrectas sin información adicional."
    ),
    "La prueba técnica tiene 90 imágenes seleccionadas": (
        "La prueba técnica se limita a 90 imágenes seleccionadas visualmente. "
        "Aunque proceden de una base distinta y de fechas posteriores al "
        "conjunto usado para entrenar el segmentador, su tamaño y selección "
        "balanceada no reemplazan una evaluación prospectiva por secuencias "
        "completas, eventos independientes y condiciones operacionales más "
        "diversas."
    ),
    "La aplicación depende de Docker y GPU para la inferencia encapsulada;||La aplicación depende del entorno local de PyTorch": (
        "La aplicación depende del entorno local de PyTorch, Ultralytics y del "
        "controlador NVIDIA para acelerar la inferencia. Docker no es un "
        "requisito operacional, pero no se evalúan todavía la latencia "
        "sostenida ni la disponibilidad del sistema en producción."
    ),
    "El repositorio mantiene configuración y código": (
        "El repositorio mantiene configuración y código separados de datos y "
        "resultados. Los umbrales del analítico residen en analytic_config.yaml; "
        "los parámetros de percepción, en YAML; y las credenciales de MLflow, "
        "en un archivo de entorno excluido de Git. El modelo principal PSINet y "
        "el checkpoint alternativo best.pt se conservan en rutas explícitas, "
        "mientras MLflow mantiene la trazabilidad del entrenamiento."
    ),
    "La evaluación técnica obtiene 88,89 %": (
        "La evaluación técnica obtiene 88,89 % de exactitud global con el modelo "
        "principal PSINet. El análisis de errores confirma que la calidad y "
        "cobertura de la máscara condicionan el resultado extremo a extremo. Los "
        "65 descartes en la muestra de 1.248 imágenes también muestran la "
        "utilidad de declarar no evaluables los casos sin evidencia suficiente."
    ),
    "Versionar explícitamente el modelo de percepción": (
        "Versionar explícitamente el modelo principal y el checkpoint "
        "alternativo, y registrar junto a cada corrida la ruta del peso, la "
        "clase seleccionada, el umbral de confianza y el conjunto de evaluación."
    ),
    "ANEXO B. OBSERVACIONES EDITORIALES PARA LA INTEGRACIÓN": (
        "ANEXO B. PENDIENTES EDITORIALES PARA LA INTEGRACIÓN"
    ),
}


APPENDIX_PENDING_ITEMS = (
    "Definir el título definitivo de la Memoria y reemplazar el título provisional de la portada.",
    "Ajustar la numeración final de capítulos, tablas, figuras y anexos al fusionar esta parte con Granulometría; el prefijo C se mantiene provisionalmente.",
    "Actualizar las fechas de consulta de las referencias y aplicar el estilo bibliográfico exigido por la carrera al cerrar el documento completo.",
)


def replace_paragraph_text(paragraph, replacement: str) -> None:
    if paragraph.runs:
        paragraph.runs[0].text = replacement
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(replacement)


def remove_paragraph(paragraph) -> None:
    element = paragraph._element
    element.getparent().remove(element)
    paragraph._p = paragraph._element = None


def update_appendix(document: Document) -> None:
    paragraphs = document.paragraphs
    heading_index = next(
        i
        for i, paragraph in enumerate(paragraphs)
        if paragraph.text == "ANEXO B. PENDIENTES EDITORIALES PARA LA INTEGRACIÓN"
    )
    item_paragraphs = paragraphs[heading_index + 2 :]
    if len(item_paragraphs) < len(APPENDIX_PENDING_ITEMS):
        raise RuntimeError("El Anexo B no contiene suficientes párrafos editables.")

    for paragraph, text in zip(item_paragraphs, APPENDIX_PENDING_ITEMS):
        replace_paragraph_text(paragraph, text)
    for paragraph in item_paragraphs[len(APPENDIX_PENDING_ITEMS) :]:
        remove_paragraph(paragraph)


def update_text(document: Document) -> None:
    found = set()
    for prefix_group, replacement in PARAGRAPH_REPLACEMENTS.items():
        prefixes = prefix_group.split("||")
        for paragraph in document.paragraphs:
            if any(paragraph.text.startswith(prefix) for prefix in prefixes):
                replace_paragraph_text(paragraph, replacement)
                found.add(prefix_group)
                break
            if paragraph.text == replacement:
                found.add(prefix_group)
                break

    missing = set(PARAGRAPH_REPLACEMENTS) - found
    if missing:
        raise RuntimeError(f"No se encontraron párrafos esperados: {sorted(missing)}")

    alternatives_table = document.tables[0]
    alternatives_table.cell(4, 3).text = "Solución implementada."

    requirements_table = document.tables[1]
    requirements_table.cell(8, 2).text = (
        "El entorno de entrenamiento se construye con Docker y dependencias fijadas."
    )

    components_table = document.tables[2]
    components_table.cell(7, 1).text = "Python local"
    components_table.cell(7, 2).text = (
        "Orquestar las etapas y construir el resumen por video."
    )

    training_table = document.tables[4]
    training_table.cell(1, 1).text = "yolov8n-seg (alternativo)"
    training_table.cell(1, 2).text = "Checkpoint reproducible conservado como respaldo."
    update_appendix(document)


def replace_embedded_figures(
    docx_path: Path,
    figures: dict[str, Path],
) -> None:
    with tempfile.TemporaryDirectory(prefix="cone_report_") as temp_dir:
        temp = Path(temp_dir)
        unpacked = temp / "unpacked"
        unpacked.mkdir()
        with zipfile.ZipFile(docx_path) as archive:
            archive.extractall(unpacked)

        for embedded_name, figure_path in figures.items():
            embedded = unpacked / "word" / "media" / embedded_name
            if not embedded.exists():
                raise FileNotFoundError(f"No se encontró la figura embebida: {embedded}")
            if not figure_path.exists():
                raise FileNotFoundError(f"No se encontró la figura fuente: {figure_path}")
            shutil.copy2(figure_path, embedded)

        rebuilt = temp / docx_path.name
        with zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(unpacked.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(unpacked).as_posix())
        shutil.copy2(rebuilt, docx_path)


def resize_architecture_figure(document: Document) -> None:
    for shape in document.inline_shapes:
        relationship_id = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        target = document.part.rels[relationship_id].target_ref
        if target.endswith("image2.jpg"):
            shape.width = Inches(5.7)
            shape.height = Inches(2.85)
            return
    raise RuntimeError("No se encontro la figura de arquitectura en el informe.")


def main() -> None:
    if not REPORT.exists():
        raise FileNotFoundError(REPORT)
    for figure_path in REPORT_FIGURES.values():
        if not figure_path.exists():
            raise FileNotFoundError(figure_path)

    document = Document(REPORT)
    update_text(document)
    resize_architecture_figure(document)
    document.save(REPORT)
    replace_embedded_figures(REPORT, REPORT_FIGURES)
    print(f"Informe actualizado: {REPORT}")


if __name__ == "__main__":
    main()
