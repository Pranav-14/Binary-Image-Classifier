import os
import sys
import argparse
from src.config import config
from src.predict import ImagePredictor

def main():
    parser = argparse.ArgumentParser(
        description="Binary Image Classifier CLI - Clean vs Garbage Sanitation AI"
    )
    parser.add_argument(
        "action", choices=["predict", "batch", "info"],
        help="Command to run: 'predict' for single image, 'batch' for directory scan, 'info' for model info"
    )
    parser.add_argument(
        "--path", "-p", type=str, default=None,
        help="Filepath or directory path for prediction"
    )

    args = parser.parse_args()
    predictor = ImagePredictor()

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        console = Console()
    except ImportError:
        console = None

    if args.action == "info":
        info_text = f"Model Backend: {predictor.backend.upper()}\n"
        info_text += f"Target Resolution: {config.IMG_SIZE}\n"
        info_text += f"Classes: {config.CLASS_LABELS}"
        if console:
            console.print(Panel(info_text, title="Binary Image Classifier System Info", border_style="cyan"))
        else:
            print(f"=== Binary Image Classifier System Info ===\n{info_text}")
        return

    if args.action == "predict":
        path = args.path
        if not path and os.path.exists(config.SAMPLES_DIR):
            sample_files = [f for f in os.listdir(config.SAMPLES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if sample_files:
                path = os.path.join(config.SAMPLES_DIR, sample_files[0])

        if not path or not os.path.exists(path):
            print("Error: Please specify a valid image file path using --path <filename>")
            sys.exit(1)

        result = predictor.predict(path)
        filename = os.path.basename(path)

        if console:
            color = "red" if result["class_id"] == 1 else "green"
            table = Table(title=f"Classification Result: {filename}", border_style=color)
            table.add_column("Property", style="bold cyan")
            table.add_column("Value", style="bold white")
            table.add_row("Classification", f"[{color}]{result['label']}[/{color}]")
            table.add_row("Status", result["status"])
            table.add_row("Confidence Score", f"{result['confidence']}%")
            table.add_row("Raw Probability", str(result["raw_probability"]))
            table.add_row("Inference Engine", predictor.backend.upper())
            console.print(table)
        else:
            print(f"\n--- Classification Result: {filename} ---")
            print(f"Label: {result['label']}")
            print(f"Status: {result['status']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Backend: {predictor.backend}")

    elif args.action == "batch":
        dir_path = args.path or config.SAMPLES_DIR
        if not os.path.exists(dir_path):
            print(f"Error: Directory path {dir_path} does not exist.")
            sys.exit(1)

        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not files:
            print(f"No valid image files found in {dir_path}")
            return

        if console:
            table = Table(title=f"Batch Classification ({len(files)} images)", border_style="blue")
            table.add_column("Filename", style="cyan")
            table.add_column("Predicted Class", style="bold")
            table.add_column("Confidence", style="magenta")
            table.add_column("Raw Prob", style="dim")

            for f in files:
                res = predictor.predict(f)
                color = "red" if res["class_id"] == 1 else "green"
                table.add_row(
                    os.path.basename(f),
                    f"[{color}]{res['label']}[/{color}]",
                    f"{res['confidence']}%",
                    str(res['raw_probability'])
                )
            console.print(table)
        else:
            print(f"\n=== Batch Classification ({len(files)} images) ===")
            for f in files:
                res = predictor.predict(f)
                print(f"{os.path.basename(f)} -> {res['label']} ({res['confidence']}%)")

if __name__ == "__main__":
    main()
