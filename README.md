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

## Instalace

pip install -r requirements.txt

## CLI

python cli.py --help

## Build binárního souboru

pyinstaller vanonymizer.spec

## Licence

MIT
