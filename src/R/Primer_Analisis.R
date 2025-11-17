# MONITORIZACIÓN Y ANÁLISIS DE DATOS QKD (TFG)
# Autor: Miguel López Ferreiro
# Tutor: Alberto Juan Sebastián Lombraña
# Fecha: 31/10/2025
#
# Descripción:
# Este script realiza el primer análisis exploratorio 
# de tres datasets cuánticos (QTI y Toshiba 2024–2025).
# Se limpian, unifican, generan estadísticas descriptivas,
# correlaciones, métricas de estabilidad y detección de
# anomalías, con salida estructurada para posteriores
# análisis predictivos (Python, Grafana, ML, etc.).
invisible({
# --- LIBRERÍAS NECESARIAS ---
library(tidyverse)
library(lubridate)
library(ggplot2)
library(scales)
library(corrplot)

# 1. CARGA Y PREPARACIÓN DE LOS DATOS

# Ruta base de datos
ruta <- file.path(getwd(), "data")

# Carga de los archivos CSV originales
qti <- read_csv(file.path(ruta, "raw/QTI.csv"))
toshiba_2024 <- read_csv(file.path(ruta, "raw/Toshiba-2024-W25.csv"))
toshiba_2025 <- read_csv(file.path(ruta, "raw/Toshiba-2025-W27.csv"))

# Conversión de cadenas de texto a formato datetime
qti <- qti %>% mutate(datetime = ymd_hms(datetime))
toshiba_2024 <- toshiba_2024 %>% mutate(Time = ymd_hms(Time))
toshiba_2025 <- toshiba_2025 %>% mutate(Time = ymd_hms(Time))

# Renombrado homogéneo de columnas
toshiba_2024 <- toshiba_2024 %>% rename(datetime = Time,
                                        qber = QBER,
                                        secure_key_rate = `SecureKeyRate(bps)`)
toshiba_2025 <- toshiba_2025 %>% rename(datetime = Time,
                                        qber = QBER,
                                        secure_key_rate = `SecureKeyRate(bps)`)

# Añadimos identificadores de origen
qti$source <- "QTI"
toshiba_2024$source <- "Toshiba_2024"
toshiba_2025$source <- "Toshiba_2025"

# Añadimos columna vacía de pérdidas ópticas (solo existe en QTI)
toshiba_2024$channel_loss <- NA
toshiba_2025$channel_loss <- NA

# Unificación de todos los datasets en un único DataFrame
data_all <- bind_rows(
  qti %>% select(datetime, qber, secure_key_rate, channel_loss, source),
  toshiba_2024 %>% select(datetime, qber, secure_key_rate, channel_loss, source),
  toshiba_2025 %>% select(datetime, qber, secure_key_rate, channel_loss, source)
)

# 2. ESTADÍSTICAS DESCRIPTIVAS

#cat("\n=== ESTADÍSTICAS DESCRIPTIVAS ===\n")

summary_stats <- data_all %>%
  group_by(source) %>%
  summarise(
    registros = n(),
    qber_promedio = mean(qber, na.rm = TRUE),
    qber_sd = sd(qber, na.rm = TRUE),
    qber_min = min(qber, na.rm = TRUE),
    qber_max = max(qber, na.rm = TRUE),
    keyrate_promedio = mean(secure_key_rate, na.rm = TRUE),
    keyrate_sd = sd(secure_key_rate, na.rm = TRUE),
    keyrate_min = min(secure_key_rate, na.rm = TRUE),
    keyrate_max = max(secure_key_rate, na.rm = TRUE),
    perdida_promedio = mean(channel_loss, na.rm = TRUE),
    perdida_sd = sd(channel_loss, na.rm = TRUE)
  )

#print(summary_stats)

#cat("\nInterpretación:\n")
#cat("Las métricas anteriores permiten cuantificar la estabilidad y el rendimiento medio de cada sistema.\n")
#cat("El QBER representa la proporción de errores en el canal cuántico, mientras que el SecureKeyRate indica la velocidad de generación de claves seguras.\n")
#cat("La desviación estándar permite evaluar la regularidad de la conexión: valores más bajos implican una red más estable.\n")

# 3. CORRELACIONES ENTRE VARIABLES PRINCIPALES

#cat("\n=== CORRELACIONES ENTRE VARIABLES ===\n")

corr_qti <- cor(qti$qber, qti$secure_key_rate)
corr_toshiba_2024 <- cor(toshiba_2024$qber, toshiba_2024$secure_key_rate)
corr_toshiba_2025 <- cor(toshiba_2025$qber, toshiba_2025$secure_key_rate)

#cat(paste("QTI → Corr(QBER, Rate):", round(corr_qti, 4), "\n"))
#cat(paste("Toshiba 2024 → Corr(QBER, Rate):", round(corr_toshiba_2024, 4), "\n"))
#cat(paste("Toshiba 2025 → Corr(QBER, Rate):", round(corr_toshiba_2025, 4), "\n"))

#cat("\nInterpretación:\n")
#cat("Una correlación negativa indica que un aumento en el error cuántico (QBER) se traduce en una disminución de la tasa de generación de claves (Key Rate).\n")
#cat("Este resultado valida la dependencia inversa entre la pureza del canal y la productividad del sistema cuántico.\n")

# 4. DETECCIÓN DE VALORES ATÍPICOS (OUTLIERS)

#cat("\n=== DETECCIÓN DE VALORES ATÍPICOS ===\n")

outliers <- data_all %>%
  group_by(source) %>%
  mutate(
    z_qber = (qber - mean(qber, na.rm = TRUE)) / sd(qber, na.rm = TRUE),
    z_rate = (secure_key_rate - mean(secure_key_rate, na.rm = TRUE)) / sd(secure_key_rate, na.rm = TRUE)
  ) %>%
  filter(abs(z_qber) > 3 | abs(z_rate) > 3)

#cat("Número total de valores atípicos detectados:", nrow(outliers), "\n")

#cat("\nInterpretación:\n")
#cat("Los valores detectados corresponden a condiciones de operación anómalas, como interferencias ópticas, ruido térmico o inestabilidades en la fuente.\n")
#cat("Estos puntos pueden utilizarse posteriormente para entrenar modelos de detección automática de anomalías mediante técnicas de Machine Learning.\n")

# 5. ESTABILIDAD TEMPORAL Y VARIABILIDAD ENTRE MEDICIONES

#cat("\n=== ESTABILIDAD TEMPORAL ===\n")

data_all <- data_all %>%
  group_by(source) %>%
  arrange(datetime) %>%
  mutate(
    delta_qber = qber - lag(qber),
    delta_rate = secure_key_rate - lag(secure_key_rate)
  )

variacion_media <- data_all %>%
  group_by(source) %>%
  summarise(
    media_abs_delta_qber = mean(abs(delta_qber), na.rm = TRUE),
    media_abs_delta_rate = mean(abs(delta_rate), na.rm = TRUE)
  )

#print(variacion_media)

#cat("\nInterpretación:\n")
#cat("Esta métrica permite cuantificar la estabilidad de la red a lo largo del tiempo. Una variación media reducida indica que el enlace mantiene una calidad constante.\n")
#cat("Valores altos pueden reflejar ciclos térmicos, oscilaciones ambientales o ajustes automáticos del sistema óptico.\n")

# 6. NUEVO ANÁLISIS: RELACIÓN ENTRE PÉRDIDAS ÓPTICAS Y RENDIMIENTO (solo QTI)

#cat("\n=== ANÁLISIS FÍSICO DE LAS PÉRDIDAS (QTI) ===\n")

if("channel_loss" %in% colnames(qti)){
  correlacion_perdidas <- cor(qti$channel_loss, qti$secure_key_rate)
  #cat(paste("Correlación entre pérdidas ópticas y tasa de clave:", round(correlacion_perdidas, 4), "\n"))
  #cat("Una correlación negativa confirma que el aumento de pérdidas ópticas en el canal reduce la capacidad de generación de clave segura.\n")
}

# 7. ANÁLISIS DE DISTRIBUCIONES Y ESTADÍSTICAS ADICIONALES

#cat("\n=== DISTRIBUCIONES DE VARIABLES ===\n")

# Cálculo de asimetría y curtosis (forma de la distribución)
distribucion_estadisticas <- data_all %>%
  group_by(source) %>%
  summarise(
    asimetria_qber = mean((qber - mean(qber, na.rm = TRUE))^3, na.rm = TRUE) / (sd(qber, na.rm = TRUE)^3),
    curtosis_qber = mean((qber - mean(qber, na.rm = TRUE))^4, na.rm = TRUE) / (sd(qber, na.rm = TRUE)^4),
    asimetria_rate = mean((secure_key_rate - mean(secure_key_rate, na.rm = TRUE))^3, na.rm = TRUE) / (sd(secure_key_rate, na.rm = TRUE)^3),
    curtosis_rate = mean((secure_key_rate - mean(secure_key_rate, na.rm = TRUE))^4, na.rm = TRUE) / (sd(secure_key_rate, na.rm = TRUE)^4)
  )


#cat("\nInterpretación:\n")
#cat("La asimetría mide la desviación de la distribución respecto a la simetría normal.\n")
#cat("La curtosis mide la concentración de valores alrededor de la media. Estos indicadores son relevantes para detectar sesgos o comportamientos extremos.\n")

# 8. EXPORTACIÓN DE RESULTADOS

# Creación de carpetas de salida
dir.create(file.path(ruta, "processed"), showWarnings = FALSE)
dir.create(file.path(ruta, "processed/primer_analisis"), showWarnings = FALSE)

# Exportación de todos los conjuntos de resultados
write_csv(data_all, file.path(ruta, "processed/primer_analisis/QKD_ALL_CLEAN.csv"))
write_csv(summary_stats, file.path(ruta, "processed/primer_analisis/STATS_POR_FUENTE.csv"))
write_csv(outliers, file.path(ruta, "processed/primer_analisis/OUTLIERS_DETECTADOS.csv"))
write_csv(variacion_media, file.path(ruta, "processed/primer_analisis/VARIACION_TEMPORAL.csv"))
write_csv(distribucion_estadisticas, file.path(ruta, "processed/primer_analisis/DISTRIBUCION_METRICAS.csv"))

#cat("\nArchivos exportados en la carpeta 'processed/primer_analisis':\n")
#cat(" - QKD_ALL_CLEAN.csv .......... Datos unificados y limpios.\n")
#cat(" - STATS_POR_FUENTE.csv ....... Estadísticas descriptivas globales.\n")
#cat(" - OUTLIERS_DETECTADOS.csv .... Registros atípicos detectados.\n")
#cat(" - VARIACION_TEMPORAL.csv ..... Estabilidad media temporal.\n")
#cat(" - DISTRIBUCION_METRICAS.csv .. Asimetría y curtosis de las variables.\n")

})