# PCG projekt 1
- autor: xkrato61

## Měření výkonu (čas / 100 kroků simulace)

### Průběžné
|   N   | CPU [s]   | Step 0 [s] | Step 1 [s] | Step 2 [s] |
|:-----:|-----------|------------|------------|------------|
|  4096 |   0.49213 |  0.203007  |  0.106965  |  0.091065  |
|  8192 |   1.47132 |  0.407360  |  0.212823  |  0.181292  |
| 12288 |   2.47894 |  0.612802  |  0.318243  |  0.271505  |
| 16384 |   3.38680 |  0.818658  |  0.424326  |  0.361717  |
| 20480 |   5.05924 |  1.019877  |  0.530424  |  0.451936  |
| 24576 |   7.11217 |  1.224882  |  0.636555  |  0.542110  |
| 28672 |   9.89285 |  1.428629  |  0.742706  |  0.632310  |
| 32768 |  12.59829 |  1.632473  |  0.848957  |  0.722580  |
| 36864 |  15.54297 |  1.836832  |  0.955091  |  0.812797  |
| 40960 |  19.36099 |  2.041330  |  1.061407  |  0.902983  |
| 45056 |  23.48723 |  2.244779  |  1.167693  |  0.993277  |
| 49152 |  27.69359 |  2.448539  |  1.273562  |  1.083469  |
| 53248 |  32.63063 |  2.652461  |  1.379569  |  1.173780  |
| 57344 |  37.43660 |  4.411817  |  2.596695  |  2.182826  |
| 61440 |  42.85863 |  4.745105  |  2.784927  |  2.338922  |
| 65536 |  49.46104 |  5.076002  |  2.974576  |  2.494674  |
| 69632 |  55.14939 |  5.401757  |  3.160765  |  2.650666  |
| 73728 |  62.04446 |  5.726504  |  3.347031  |  2.806445  |
| 77824 |  69.26138 |  6.052894  |  3.533109  |  2.962367  |
| 81920 |  76.60071 |  6.377802  |  3.719534  |  3.118157  |

### Závěrečné
|    N   |  CPU [s] | GPU [s]  | Zrychlení | Propustnost [GiB/s] | Výkon [GFLOPS] |
|:------:|:--------:|:--------:|:---------:|:-------------------:|:--------------:|
|   1024 |   1.0928 | 0.025488 | 42.875078 |       0.136         |      126.42    |
|   2048 |   0.5958 | 0.046942 | 12.692258 |       0.119         |      256.14    |
|   4096 |   0.6652 | 0.091921 |  7.236648 |       0.111         |      516.17    |
|   8192 |   1.6599 | 0.182133 |  9.113669 |       0.107         |     1035.62    |
|  16384 |   3.3655 | 0.362584 |  9.281987 |       0.105         |     2073.64    |
|  32768 |  12.7233 | 0.724270 | 17.567067 |       0.103         |     4146.67    |
|  65536 |  48.9732 | 2.495646 | 19.623456 |       0.060         |     4797.16    |
| 131072 | 195.9965 | 7.435132 | 26.360863 |       0.040         |     6468.37    |

## Otázky

### Krok 0: Základní implementace
**Vyskytla se nějaká anomále v naměřených časech? Pokud ano, vysvětlete:**
Ano, pre N=53248 a N=57344 dochadza skoro k vyse 1.7 navyseniu casu potrebneho
na vypocet. Kedze sa benchmark spusta v konfiguracii 512 threads/block, pre N=53248
potrebujeme 104 blokov a pre N=57344 potrebujeme az 112 blokov. Kedze blok sa
mapuje na SM procesor a na Karoline mame A100 40GB s 108 SM procesormi, prekrocime
hranicu mapovania 1:1 (blok:SM) a teda niektore SM musia spracovat viac blokov.
To vysvetluje skoro dvojnasobny cas potrebny na vypocet.

### Krok 1: Sloučení kernelů
**Došlo ke zrychlení?**
Ano, doslo k nezanedbatelnemu zrychleniu.

**Popište hlavní důvody:**
- Nizsia rezia spustania kernelov.
- Nemusime pouzivat prechodne ulozisko na vypocet (tmpVel)
- Znovupouzitie registrov.
- Niektore smycky a cast vypoctu je rovnaka (napr. vypocet vzdialenosti dvoch telies), teda
  po zluceni zbytocne nevykonavame rovnaky vypocet dva krat.
- Lepsi instruction-level parallelism.

### Krok 2: Sdílená paměť
**Došlo ke zrychlení?**
Ano, znova doslo k nezanedbatelnemu zrychleniu.

**Popište hlavní důvody:**
Dochadza k prednacitaniu dat jednotlivych castic kazdym vlaknom do zdielanej pamate.
Nasledne sa pri vypocte jednotlive vlakne nedotazuju do globalnej pamate ale vyuzivaju
namiesto toho pamat zdielanu, teda je tam podstatne nizsia latencia a vyssia priepustnost.
Zvysila sa znovupouzitelnost pamate. Taktiez som naimplementoval pouzitie zdielanej pamate tak,
ze susedne vlakna pristupuju do susednych lokacii, co vyrazne zrychlilo beh programu. 

### Krok 5: Měření výkonu
**Jakých jste dosáhli výsledků?**
Pre systemy s mensim poctom telies dochadza az k 42x zrychleniu (N=1024).

**Lze v datech pozorovat nějaké anomálie?**
Zrychlenie potom klesa pre vacsie systemy a pre velke zacina znovu narastat. Vysvetlujem si to
podobne ako pre anomaliu v kroku 0, kedze znova po prekonani "limitu" mapovania 1:1 (blok-SM proc.)
sa zacne vypocet rovnomerne rozkladat medzi SM procesory.

Za dalsiu anomaliu by som oznacil pomerne rychlo klesajucu priepustnost pamate. Tu vsak mozno vysvetlit
zvysujucim sa vykonom a drasticky sa zvysujucou aritmetickou intenzitou vypoctu systemov s vela telesami.
Toto sa potvrdilo aj pri profilingu v nastroji Nsight Compute, kde som zistil, ze vysledny nbody kod je
compute bound a ani sa neblizil teoretickemu maximu vytazenia pamate.
