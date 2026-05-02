# Vis_per_comp — Segmentación ósea 3D en CTs PENGWIN

Pipeline de Visión por Computador para segmentar hueso pélvico en volúmenes CT del dataset **PENGWIN**, usando una U-Net 3D (MONAI) con pérdida combinada Dice + Cross-Entropy.

## Dataset

- **PENGWIN — Pelvic Bone Fracture Segmentation Challenge**
- Fuente: https://zenodo.org/records/10927452
- Archivos consumidos:
  - `PENGWIN_CT_train_images_part1.zip` → CTs originales (Hounsfield Units).
  - `PENGWIN_CT_train_labels.zip` → máscaras multi-clase de hueso/fragmentos.

El notebook **descarga y extrae automáticamente** los `.zip` si las carpetas locales correspondientes están vacías (ver `ensure_dataset()` en la primera celda de utilidades). No hace falta bajar nada a mano.

## Estructura

```
Vis_per_comp/
├── data_processing.ipynb            # Pipeline completo (download → preproc → train → vis)
├── best_pengwin_model.pth            # Checkpoint por mejor Dice
├── best_pengwin_model_min_loss.pth   # Checkpoint por menor val loss
└── Data/
    ├── PENGWIN_CT_train_images_part1/      # CTs originales (auto-descarga)
    ├── PENGWIN_CT_train_images_normalized/ # CTs clippeadas [50, 1000] HU y escaladas a [0,1]
    ├── PENGWIN_CT_train_labels/            # Labels multi-clase originales (auto-descarga)
    └── PENGWIN_CT_train_labels_binary/     # Labels binarizadas (hueso vs fondo)
```

## Requisitos

- **Python 3.11+** (probado en 3.14.4 / Homebrew, macOS arm64).
- GPU CUDA recomendada (el notebook usa `cuda` si está disponible, fallback a `cpu`).
- Dependencias:

```bash
pip install --user \
  torch monai SimpleITK itk requests tqdm \
  numpy scikit-learn matplotlib pyvista scikit-image
```

> `itk` es **obligatorio** además de `SimpleITK`: MONAI sólo registra `ITKReader` para leer `.mha` si encuentra el paquete `itk` instalado. Sin él, `LoadImaged` falla con *"cannot find a suitable reader"*.

## Pipeline (orden de ejecución de las celdas)

1. **Imports** — librerías base.
2. **`ensure_dataset()`** — utilidad de descarga + extracción segura (con protección contra Zip Slip / CWE-22).
3. **Binarización de labels** — cualquier valor > 0 se mapea a 1, el resto a 0; preserva `spacing`/`origin`.
4. **Normalización de CTs** — clip a `[50, 1000] HU` (rango óseo) y escalado min-max a `[0, 1]`.
5. **Verificación visual** — comparación corte-a-corte original vs. normalizada.
6. **Setup del modelo** — U-Net 3D con `DiceCELoss(sigmoid=True)`, Adam (lr=1e-4), `roi_size=[128,128,128]`.
7. **Entrenamiento** — 50 épocas, validación cada 2, sliding-window inference para volumen completo, guarda checkpoint con mínima val loss.
8. **Visualización 3D** — extracción de superficie (`marching_cubes`) y render PyVista comparativo (GT vs. predicción).

## Decisiones de preprocesamiento

- **Clipping a `[50, 1000] HU`**: el hueso esponjoso/cortical vive ~200–1000 HU; el piso en 50 HU mantiene el borde tejido-blando/hueso. Todo lo fuera de rango se satura.
- **Normalización min-max a `[0, 1]`**: estabiliza los gradientes y unifica brillo entre pacientes/instituciones.
- **Binarización**: el problema se reduce a `hueso vs. fondo` — descarta la distinción de fragmentos individuales del label original.

## Notas

- `num_workers=0` en los DataLoaders: más lento pero evita errores de pickling/multiprocessing en notebooks (especialmente Windows).
- El loss reportado en `tqdm.set_postfix` durante el entrenamiento es la loss del **último batch**, no el promedio de la época. La métrica comparable es la val loss impresa cada 2 épocas.
- Los `.pth` están versionados — pesan ~19 MB cada uno. Si crece el repo, mover a Releases o LFS.
