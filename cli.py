#!/usr/bin/env python3

import argparse
from tqdm import tqdm
from vanonymizer.processor import VideoProcessor
import cv2


def main():

    class CzechArgumentParser(argparse.ArgumentParser):
    
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    
        def format_help(self):
            help_text = super().format_help()
    
            return (help_text
                .replace("usage:", "\nPoužití:")
                .replace("options:", "Možnosti:")
                .replace("optional arguments:", "Volitelné argumenty:")
                .replace("positional arguments:", "Povinné položky:")
                .replace("show this help message and exi:", "Zobrazí tuto nápovědu.")
            )

    parser = CzechArgumentParser(
        prog="vanonymizer",
        description="vanonymizer – anonymizace obličejů, postav a SPZ ve videu by @psimonek.",
        epilog="Příklad: vanonymizer input.mp4 output.mp4 --people --pixelate --detect-interval 2\n\u200b",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("input", help="Vstupní video pro anonymizaci.")
    parser.add_argument("output", help="Název souboru pro uložení hotového videa.")
    parser.add_argument(
            "--version",
            action="version",
            version="vanonymizer 1.0",
            help="Zobrazení verze programu."
        )

    detekce = parser.add_argument_group("Detekce objektů")

    detekce.add_argument("--people",help="Detekovat celé postavy lidí.", action="store_true")
    detekce.add_argument("--no-faces", help="Nedetekovat obličeje.", action="store_true")
    detekce.add_argument("--no-plates", help="Nedetekovat registrační značky.", action="store_true")

    anonymizace = parser.add_argument_group("Způsob anonymizace")

    anonymizace.add_argument("--blur", help="Použít rozmazání  (výchozí).", action="store_true")
    anonymizace.add_argument("--pixelate", help="Použít pixelizaci namísto rozmazání.", action="store_true")
    anonymizace.add_argument("--blackbox", help="Použít černé překrytí namísto rozmazání.", action="store_true")

    pokrocile = parser.add_argument_group("Pokročilé nastavení")

    pokrocile.add_argument("--detect-interval", help="Detekce každých N snímků. Výchozí je 5.", type=int, default=5)
    pokrocile.add_argument("--track-buffer", help="Počet snímků, kdy má zachovat detekci při ztrátě objektu. Výchozí 40.", type=int, default=40)
    pokrocile.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Volba výpočetního zařízení (výchozí je cpu)."
    )

    args = parser.parse_args()

    config = {
        "PEOPLE": args.people,
        "FACES": not args.no_faces,
        "PLATES": not args.no_plates,
        "BLUR_TYPE": (
            "pixelate" if args.pixelate
            else "blackbox" if args.blackbox
            else "blur"
        ),
        "DETECT_INTERVAL": args.detect_interval,
        "TRACK_BUFFER": args.track_buffer,
        "DEVICE": args.device
    }

    cap = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    bar = tqdm(total=total_frames, unit="frame")

    def progress(frame_idx, eta):
        bar.n = frame_idx
        bar.refresh()
        bar.set_postfix_str(f"ETA {int(eta)}s")

    processor = VideoProcessor(config=config)

    processor.process_video(
        args.input,
        args.output,
        progress_callback=progress
    )

    bar.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
