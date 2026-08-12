import warnings
from pathlib import Path

import fitz  # PyMuPDF
import gradio as gr
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import BoundingBox, Cluster, Page, Size
from docling.utils.layout_postprocessor import LayoutPostprocessor
from PIL import Image, ImageDraw, ImageFont

from docling_pp_doc_layout.model import PPDocLayoutV3Model
from docling_pp_doc_layout.options import PPDocLayoutV3Options

# Global model instance
model = None
current_options = None


def load_model(options):
    global model, current_options
    # Re-load model only if model name changed or if it's the first time
    if model is None or current_options.model_name != options.model_name:
        accelerator_options = AcceleratorOptions()
        model = PPDocLayoutV3Model(artifacts_path=None, accelerator_options=accelerator_options, options=options)
        current_options = options
    else:
        # Update options on existing model
        model.options = options
    return model


# Color map for labels
COLORS = {
    "text": "red",
    "table": "blue",
    "picture": "green",
    "caption": "orange",
    "section_header": "purple",
    "title": "brown",
    "page_header": "gray",
    "page_footer": "gray",
    "footnote": "pink",
    "formula": "cyan",
    "code": "magenta",
}


def get_color(label):
    if hasattr(label, "value"):
        label_str = label.value.lower()
    else:
        label_str = str(label).lower()
    return COLORS.get(label_str, "red")


def draw_detections(image, detections, draw_polygons=True):
    draw = ImageDraw.Draw(image)
    try:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "DejaVuSans.ttf",
        ]
        font = None
        for p in font_paths:
            if Path(p).exists():
                font = ImageFont.truetype(p, 18)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        label = det["label"]
        score = det["confidence"]
        box = [det["l"], det["t"], det["r"], det["b"]]
        poly = det.get("polygon")

        color = get_color(label)

        # Draw Polygon if available
        if draw_polygons and poly is not None:
            # Poly can be [[x,y], [x,y]] or [x,y,x,y]
            if isinstance(poly[0], list):
                flat_poly = [item for sublist in poly for item in sublist]
            else:
                flat_poly = poly

            if len(flat_poly) >= 4:
                draw.polygon(flat_poly, outline=color, width=2)

        # Draw Bounding Box
        draw.rectangle(box, outline=color, width=3)

        text = f"{label.value if hasattr(label, 'value') else label}: {score:.2f}"

        # Draw label background
        if hasattr(draw, "textbbox"):
            text_box = draw.textbbox((box[0], box[1]), text, font=font)
            draw.rectangle(text_box, fill=color)
        else:
            tw, th = draw.textsize(text, font=font)
            draw.rectangle([box[0], box[1], box[0] + tw, box[1] + th], fill=color)

        draw.text((box[0], box[1]), text, fill="white", font=font)

    return image


class MockBackend:
    def is_valid(self):
        return True


class MockPage(Page):
    image: Image.Image
    orig_image_scale: float

    def get_image(self, scale=1.0, **kwargs):
        # Scale back to original resolution, then apply new scale
        new_size = (
            int(self.image.width * scale / self.orig_image_scale),
            int(self.image.height * scale / self.orig_image_scale),
        )
        return self.image.resize(new_size, Image.Resampling.LANCZOS)


def process_file(file_path, threshold, unclip_ratio, image_scale, apply_postprocessing):
    if file_path is None:
        return []

    # Handle FileData object or string path
    if hasattr(file_path, "name"):
        path = Path(file_path.name)
    else:
        path = Path(file_path)

    options = PPDocLayoutV3Options(confidence_threshold=threshold, unclip_ratio=unclip_ratio, image_scale=image_scale)

    model = load_model(options)

    images = []
    page_sizes = []
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_sizes.append((page.rect.width, page.rect.height))
            pix = page.get_pixmap(matrix=fitz.Matrix(image_scale, image_scale))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    else:
        try:
            img = Image.open(path).convert("RGB")
            orig_size = img.size
            page_sizes.append(orig_size)
            if image_scale != 1.0:
                new_size = (int(img.width * image_scale), int(img.height * image_scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            images.append(img)
        except Exception as e:
            print(f"Error opening image: {e}")
            return []

    # Get raw detections
    batch_detections = model._run_inference(images)

    output_images = []

    for i, (img, detections) in enumerate(zip(images, batch_detections)):
        if apply_postprocessing:
            # Use MockPage to override get_image
            mock_page = MockPage(
                page_no=i,
                size=Size(width=page_sizes[i][0], height=page_sizes[i][1]),
                image=img,
                orig_image_scale=image_scale,
            )
            mock_page._backend = MockBackend()

            clusters = []
            for ix, det in enumerate(detections):
                clusters.append(
                    Cluster(
                        id=ix,
                        label=det["label"],
                        confidence=det["confidence"],
                        bbox=BoundingBox(
                            l=det["l"] / image_scale,
                            t=det["t"] / image_scale,
                            r=det["r"] / image_scale,
                            b=det["b"] / image_scale,
                        ),
                        cells=[],
                    )
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                processed_clusters, _ = LayoutPostprocessor(mock_page, clusters, options).postprocess()

            # Map back to image coordinates for drawing
            final_detections = []
            for c in processed_clusters:
                final_detections.append({
                    "label": c.label,
                    "confidence": c.confidence,
                    "l": c.bbox.l * image_scale,
                    "t": c.bbox.t * image_scale,
                    "r": c.bbox.r * image_scale,
                    "b": c.bbox.b * image_scale,
                })
            processed_img = draw_detections(img.copy(), final_detections, draw_polygons=False)
        else:
            processed_img = draw_detections(img.copy(), detections, draw_polygons=True)

        output_images.append(processed_img)

    return output_images


with gr.Blocks(title="PP-DocLayout-V3 Debug Demo") as demo:
    gr.Markdown("# PP-DocLayout-V3 Layout Analysis (Debug)")
    gr.Markdown("Compare raw model output (with polygons) vs Docling post-processed result.")

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(
                label="Upload PDF or Image", file_types=[".pdf", ".png", ".jpg", ".jpeg"], type="filepath"
            )

            with gr.Accordion("Analysis Options", open=True):
                apply_postprocessing = gr.Checkbox(
                    label="Apply Docling Post-processing (Merging/Filtering)", value=False
                )
                threshold = gr.Slider(0.0, 1.0, value=0.4, label="Confidence Threshold")
                unclip_ratio = gr.Slider(1.0, 2.0, value=1.0, label="Unclip Ratio")
                image_scale = gr.Slider(1.0, 4.0, value=2.0, label="Image Scale")

            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=2):
            output_gallery = gr.Gallery(label="Output", columns=1, height="auto", preview=True)

    submit_btn.click(
        fn=process_file,
        inputs=[input_file, threshold, unclip_ratio, image_scale, apply_postprocessing],
        outputs=[output_gallery],
    )

if __name__ == "__main__":
    demo.launch()
