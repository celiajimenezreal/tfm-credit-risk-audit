# Informe de Robustez Adversarial

## 1. Resumen de Robustez

| model   | attack   |   eps |   success_rate |   avg_distortion |
|:--------|:---------|------:|---------------:|-----------------:|
| logreg  | fgsm     |  0.01 |       0.136739 |        0.0489897 |
| logreg  | fgsm     |  0.05 |       0.138055 |        0.244948  |
| logreg  | fgsm     |  0.1  |       0.140864 |        0.489897  |
| logreg  | fgsm     |  0.2  |       0.154248 |        0.979794  |

## 2. Variables Más Vulnerables

### Modelo: **logreg**, Ataque: **fgsm**, ε = 0.01

| Variable | Veces modificada |
|:---------|-----------------:|
| Term | 3116 |
| Occupation | 3116 |
| EmploymentStatusDuration | 3116 |
| CurrentCreditLines | 3116 |
| OpenCreditLines | 3116 |

### Modelo: **logreg**, Ataque: **fgsm**, ε = 0.05

| Variable | Veces modificada |
|:---------|-----------------:|
| Term | 3146 |
| Occupation | 3146 |
| EmploymentStatusDuration | 3146 |
| CurrentCreditLines | 3146 |
| OpenCreditLines | 3146 |

### Modelo: **logreg**, Ataque: **fgsm**, ε = 0.1

| Variable | Veces modificada |
|:---------|-----------------:|
| Term | 3210 |
| Occupation | 3210 |
| EmploymentStatusDuration | 3210 |
| CurrentCreditLines | 3210 |
| OpenCreditLines | 3210 |

### Modelo: **logreg**, Ataque: **fgsm**, ε = 0.2

| Variable | Veces modificada |
|:---------|-----------------:|
| Term | 3515 |
| Occupation | 3515 |
| EmploymentStatusDuration | 3515 |
| CurrentCreditLines | 3515 |
| OpenCreditLines | 3515 |

## 3. Recomendaciones Generales

- **Adversarial training**: incluye ejemplos adversariales en el entrenamiento.
- **Regularización de gradiente**: penaliza altas sensibilidades.
- **Detectores de anomalías**: filtra inputs con cambios atípicos en variables críticas.
