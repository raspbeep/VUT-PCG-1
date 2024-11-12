# PCG projekt 1
- autor: xkrato61

## Měření výkonu (čas / 100 kroků simulace)

### Průběžné
|   N   | CPU [s]   | Step 0 [s] | Step 1 [s] | Step 2 [s] |
|:-----:|-----------|------------|------------|------------|
|  4096 |   0.49213 |  0.180416  |  0.106965  |  0.091065  |
|  8192 |   1.47132 |  0.361037  |  0.212823  |  0.181292  |
| 12288 |   2.47894 |  0.542200  |  0.318243  |  0.271505  |
| 16384 |   3.38680 |  0.722740  |  0.424326  |  0.361717  |
| 20480 |   5.05924 |  0.903371  |  0.530424  |  0.451936  |
| 24576 |   7.11217 |  1.084767  |  0.636555  |  0.542110  |
| 28672 |   9.89285 |  1.265644  |  0.742706  |  0.632310  |
| 32768 |  12.59829 |  1.446420  |  0.848957  |  0.722580  |
| 36864 |  15.54297 |  1.628652  |  0.955091  |  0.812797  |
| 40960 |  19.36099 |  1.810337  |  1.061407  |  0.902983  |
| 45056 |  23.48723 |  1.991745  |  1.167693  |  0.993277  |
| 49152 |  27.69359 |  2.174474  |  1.273562  |  1.083469  |
| 53248 |  32.63063 |  2.357250  |  1.379569  |  1.173780  |
| 57344 |  37.43660 |  4.015800  |  2.596695  |  2.182826  |
| 61440 |  42.85863 |  4.320521  |  2.784927  |  2.338922  |
| 65536 |  49.46104 |  4.623569  |  2.974576  |  2.494674  |
| 69632 |  55.14939 |  4.927458  |  3.160765  |  2.650666  |
| 73728 |  62.04446 |  5.231956  |  3.347031  |  2.806445  |
| 77824 |  69.26138 |  5.536249  |  3.533109  |  2.962367  |
| 81920 |  76.60071 |  5.840641  |  3.719534  |  3.118157  |


### Závěrečné
|    N   |  CPU [s] | GPU [s]  | Zrychlení | Propustnost [GiB/s] | Výkon [GFLOPS] |
|:------:|:--------:|:--------:|:---------:|:-------------------:|:--------------:|
|   1024 |   1.0928 | 0.025488 | 42.875078 |                     |                |
|   2048 |   0.5958 | 0.046942 | 12.692258 |                     |                |
|   4096 |   0.6652 | 0.091921 |  7.236648 |                     |                |
|   8192 |   1.6599 | 0.182133 |  9.113669 |                     |                |
|  16384 |   3.3655 | 0.362584 |  9.281987 |                     |                |
|  32768 |  12.7233 | 0.724270 | 17.567067 |                     |                |
|  65536 |  48.9732 | 2.495646 | 19.623456 |                     |                |
| 131072 | 195.9965 | 7.435132 | 26.360863 |                     |                |

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
- Znovupouzitie registrov.
- Niektore smycky a cast vypoctu je rovnaka (napr. vypocet vzdialenosti dvoch telies), teda
  po zluceni zbytocne nevykonavame rovnaky vypocet dva krat.
- Lepsi instruction-level parallelism.

### Krok 2: Sdílená paměť
**Došlo ke zrychlení?**
Ano.

**Popište hlavní důvody:**
Dochadza k prednacitaniu dat jednotlivych castic kazdym vlaknom do zdielanej pamate.
Nasledne sa pri vypocte jednotlive vlakne nedotazuju do globalnej pamate ale vyuzivaju
namiesto toho pamat zdielanu, teda je tam podstatne nizsia latencia a vyssia priepustnost.

### Krok 5: Měření výkonu
**Jakých jste dosáhli výsledků?**
Pre systemy s mensim poctom telies dochadza az k 42x zrychleniu (N=1024). Zrychlenie potom
klesa pre vacsie systemy a pre velke zacina znovu narastat. Vysvetlujem si to 

**Lze v datech pozorovat nějaké anomálie?**

