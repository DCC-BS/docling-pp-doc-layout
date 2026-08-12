from PIL import Image, ImageDraw
from transformers import pipeline

COLORS = {
    "header": "red",
    "text": "blue",
    "footer": "green",
    "table": "orange",
    "figure": "purple",
}


def main():
    image_path = "test.jpg"
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path)
    print(f"Image size: {image.size}")

    print("\nLoading PP-DocLayoutV3 model...")
    layout_detector = pipeline("object-detection", model="PaddlePaddle/PP-DocLayoutV3_safetensors", threshold=0.3)

    print("\nRunning detection...")
    results = layout_detector(image)

    print(f"\nDetected {len(results)} elements:")
    for idx, res in enumerate(results):
        print(f"Order {idx + 1}: {res}")

    draw = ImageDraw.Draw(image)
    for res in results:
        box = res["box"]
        coords = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
        color = COLORS.get(res["label"], "yellow")
        draw.rectangle(coords, outline=color, width=3)
        label_text = f"{res['label']} ({res['score']:.2f})"
        draw.text((box["xmin"], box["ymin"] - 15), label_text, fill=color)

    output_path = "test_output.jpg"
    image.save(output_path)
    print(f"\nSaved annotated image to: {output_path}")


if __name__ == "__main__":
    main()
