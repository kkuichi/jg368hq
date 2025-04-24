**Systémová príručka k diplomovej práci _Využitie hlbokého učenia pre segmentáciu a analýzu vybraných typov galaxií_**

Tento projekt sa zameriava na detekciu a analýzu edge-on galaxií pomocou hlbokého učenia a spracovania astronomických dát. Kombinuje modely YOLOv5 a SCSS-Net na detekciu a segmentáciu galaxií z dát SDSS, pričom následne vypočítava zakrivenie galaktického disku (warp).

## Obsah repozitára

- **preprocessing/**: Skripty na predspracovanie dát (napr. konverzia FITS súborov, kopírovanie súborov).
- **postprocessing/**: Nástroje na analýzu výstupov, výpočet warpu a generovanie vizualizácií,tvorbu masiek, výpočet galaktických parametrov, proces bez transformácie galaxie do veľkého formátu.
- **yolov5/**: Implementácia a trénovania detekčného modelu YOLOv5.
- **scss-net/**: Segmentačný model SCSS-Net pre extrakciu tvaru galaxií.
- **launching_scripts.ipynb**: Jupyter notebook na spustenie pipeline.
- **metrics.py**: Výpočet metrík ako precision, recall pre výstup z yolov5.
- **requirements.txt**: Zoznam požadovaných Python knižníc.

 ## Požiadavky

- Python 3.8.3
- Jupyter Notebook
- FITS súbory z SDSS (alebo vlastné galaktické dáta)
  
## Inštalácia

1. Naklonuj repozitár:
   ```bash
   git clone https://github.com/kkuichi/jg368hq.git
   cd jg368hq
   ```

2. Vytvor a aktivuj virtuálne prostredie:
   ```bash
   python -m venv venv
   source venv/bin/activate  # alebo 'venv\Scripts\activate' na Windows
   ```

3. Nainštaluj requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Použitie
1. Z preprocessingu prekonvertuj galaxie na jpg.
2. Použi yolov5 na nájdenie galaxie
3. Spusti `launching_scripts.ipynb` v Jupyter Notebooku.
4. Postupuj podľa jednotlivých buniek.
5. Spusti Jupyter Notebook `warp_calculation.ipynb`

## Výstupy

- Textové súbory s vypočítanými hodnotami warpu pre každú galaxiu.
- PDF vizualizácie galaxií s vyznačeným zakrivením disku.
- Metadáta a metriky hodnotiace výkon modelov.
