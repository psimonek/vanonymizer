# vanonymizer

CLI nástroj pro anonymizaci videa.

Detekuje a anonymizuje:

- obličeje
- osoby
- registrační značky

## Funkce

- detekce obličejů
- detekce osob
- detekce SPZ
- blur / pixelate / blackbox
- sledování objektů mezi snímky
- podpora CPU / Apple Silicon (MPS) / CUDA

## Instalace v Pythonu

pip install -r requirements.txt

## Spuštění CLI v Pythonu

python cli.py --help

## Build binárního souboru

pyinstaller vanonymizer.spec

## Binární soubory

V Realeases jsou k dispozici ke stažení kompilované soubory pro:

- macOS s architekturou Silicon. Intel není kompilován.
- Ubuntu Linux.
- Windows.

## Licence

MIT
