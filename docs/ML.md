# Auditoría de modelos (Fase 1)

Código: `backend/ml/`. Reproducir: `python -m backend.ml.audit` (≈22 s). API: `GET /api/model-audit`
(se calcula en la primera consulta y se guarda en memoria).

## Protocolo

- Partición temporal por semana: entrenamiento 1-5 (5,604 jugadas), validación 6 (1,004), prueba 7-8 (1,949).
  Ningún partido aparece en dos conjuntos (se verifica en código).
- La validación elige modelo, hiperparámetros y umbral de decisión. La prueba se evalúa una vez con el elegido.
- Toda comparación incluye un modelo base: tasa de entrenamiento (clasificación) o media (regresión).
- Ante empate técnico en validación (log loss ±0.002, RMSE ±0.02) gana el modelo más simple.
- Intervalos de confianza al 95% por remuestreo (300 réplicas) de las jugadas de prueba.
- Solo numpy/pandas, como exige el stack del hackathon. Se implementaron regresión logística, ridge,
  bosque aleatorio y boosting de árboles. XGBoost no se usa: no está en el stack.

## Variables

Cada variable declara cuándo se conoce (`backend/ml/features.py`):

| Momento | Variables |
|---|---|
| Antes del snap | down, yardsToGo, quarter, tiempo en el cuarto y en la mitad, diferencia de marcador (ofensiva), yardas a la zona de anotación, formación, personal ofensivo (RB/TE/WR), defensores en la caja, personal defensivo (DL/LB/DB) |
| Decisión defensiva evaluada | cobertura (`pff_passCoverage`), número de pass rushers |
| Solo después de la jugada (no se usan) | dropBackType, pff_playAction, passResult, hits/hurries/sacks |

Cobertura y número de rushers vienen etiquetados por PFF después de la jugada. Son las variables de decisión
del What-If, no información que el rival o el cuerpo técnico conozca antes del snap. El código antiguo
llamaba a la cobertura `expectedCoverage` aunque era la observada (`actualCoverage` era la misma columna).

## Resultados en prueba (semanas 7-8)

| Objetivo | Modelo elegido | Métrica | Elegido | Base | Estado |
|---|---|---|---|---|---|
| Presión (con decisión defensiva) | Gradient boosting, profundidad 1, 100 árboles | Brier | 0.2224 | 0.2304 | limited |
| | | Log loss | 0.6363 | 0.6535 | |
| | | ROC-AUC (IC95%) | 0.618 (0.592-0.646) | 0.500 | |
| | | PR-AUC | 0.474 | 0.359 | |
| | | ECE | 0.047 | 0.016 | |
| Presión (solo variables previas al snap) | Random forest, profundidad 4 | ROC-AUC (IC95%) | 0.585 (0.558-0.612) | 0.500 | limited |
| | | Brier | 0.2253 | 0.2304 | |
| Sack | Regresión logística, l2=100 | ROC-AUC (IC95%) | 0.609 (0.553-0.660) | 0.500 | limited |
| | | PR-AUC | 0.115 | 0.063 | |
| Yardas netas (`playResult`) | Random forest, profundidad 5 | MAE / RMSE / R² | 7.26 / 10.18 / 0.025 | 7.44 / 10.32 / -0.002 | limited |
| Yardas por intento (sin sacks) | Random forest, profundidad 5 | MAE / RMSE / R² | 7.04 / 9.88 / 0.030 | 7.25 / 10.05 / -0.003 | limited |

Estados (política del producto, constantes en `validation.py`):

- `validated`: mejora a la base con IC95% > 0, límite inferior del ROC-AUC ≥ 0.65 (R² ≥ 0.10 en regresión) y ECE ≤ 0.05.
- `limited`: mejora a la base con IC95% > 0, pero con poca capacidad de discriminar. Se puede mostrar solo con advertencia.
- `not_reliable`: no mejora a la base. No debe mostrarse como estimación.

## Conclusión

1. Ningún modelo alcanza `validated`. La señal es real pero débil: en presión, conocer la decisión defensiva
   sube el ROC-AUC de 0.585 a 0.618, y ambos intervalos quedan lejos de 0.65.
2. Las yardas por jugada casi no son predecibles con estas variables (R² ≈ 0.03). Las tarjetas de What-If deben
   mostrar frecuencias históricas con su tamaño de muestra y presentar la probabilidad modelada con advertencia.
3. Las cifras del prototipo (R² 0.047 y 0.003) usaban partición aleatoria y regresión lineal sobre un 0/1.
   No son comparables con las de esta tabla porque cambian el protocolo y la métrica.
4. El recall de 0.91 con precisión 0.37 en presión es el umbral que maximiza F1 en validación. Con una tasa
   base de 0.36 equivale a marcar casi todo como presión, no a detectar jugadas concretas.
5. Desbalance: presión 36-37%, sack 6%. No se aplicó remuestreo. Se usan métricas de probabilidad (Brier, log
   loss, PR-AUC) y se verifica la calibración, que remuestrear la distorsionaría.

## Pendiente

- La app aún usa los modelos lineales antiguos en `/api/recommend`. La Fase 2 los reemplaza por el campeón
  validado, con la etiqueta de estado y la comparación basada en frecuencias históricas.
- El tracking no se usa (decisión del equipo), así que Time to Pressure queda fuera.
