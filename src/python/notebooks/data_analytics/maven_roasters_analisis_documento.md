# Análisis de Operaciones Retail: Maven Roasters

## Sección A: Contexto Empresarial y Descripción del Proceso

### Descripción de Maven Roasters

Maven Roasters es una cadena de cafeterías con presencia en la ciudad de Nueva York. La empresa opera tres locaciones estratégicamente distribuidas en la ciudad y cuenta con un historial operacional de seis meses completos de operaciones documentadas. Durante este período, la compañía ha consolidado su modelo de negocio retail orientado al consumo de café premium y productos complementarios.

Las tres ubicaciones de Maven Roasters funcionan como puntos de venta directa al consumidor. Cada local opera sistemas de punto de venta integrados que capturan información transaccional en tiempo real. Esta infraestructura tecnológica permite el seguimiento granular de cada operación comercial ejecutada.

### Proceso Operacional Documentado

El conjunto de datos refleja las operaciones diarias de punto de venta (POS) en las tres cafeterías de Maven Roasters. Cada registro representa una transacción individual completada durante el horario de operación. El proceso captura información crítica en el momento exacto de la venta, incluyendo detalles temporales, geográficos, de producto y monetarios.

El flujo operacional documentado incluye:

- Registro temporal preciso de cada transacción (fecha y hora exactas)
- Identificación de la ubicación específica donde se ejecuta la venta
- Captura de detalles de producto (categoría, tipo y nombre específico)
- Registro de cantidades vendidas por artículo
- Documentación del valor monetario unitario y total de cada transacción

Este proceso sistemático genera un registro completo de la actividad comercial. Los datos se acumulan de forma continua durante las horas de operación en las tres locaciones simultáneamente.

### Problema de Consultoría y Objetivos

Maven Roasters enfrenta el desafío típico de las operaciones retail multilocación: optimizar el rendimiento comercial en un entorno con múltiples variables operacionales. La gerencia requiere visibilidad analítica sobre los patrones subyacentes que determinan el éxito o fracaso de diferentes aspectos del negocio.

**Problema Central:**

Determinar los factores operacionales que generan valor para Maven Roasters y aquellos que representan oportunidades de mejora. La empresa necesita comprender si opera como un sistema unificado o como tres entidades con dinámicas diferenciadas.

**Objetivos Analíticos:**

- Identificar patrones temporales en la demanda que permitan optimización de recursos
- Determinar qué elementos del portafolio de productos generan mayor contribución al negocio
- Evaluar la homogeneidad o heterogeneidad operacional entre las tres ubicaciones
- Descubrir oportunidades de crecimiento y áreas de riesgo en la estructura actual
- Proporcionar recomendaciones accionables basadas en evidencia cuantitativa

### Características Específicas del Conjunto de Datos

El análisis se fundamenta en un conjunto de datos con las siguientes características verificadas:

- **Volumen transaccional:** 149,116 transacciones individuales registradas
- **Período temporal:** Enero a Junio de 2023 (6 meses calendario completos)
- **Cobertura geográfica:** 3 ubicaciones en la ciudad de Nueva York
- **Ingresos totales documentados:** $698,812.33 USD
- **Dimensiones de información:** 11 variables capturadas por transacción

Las 11 dimensiones del conjunto de datos incluyen:

1. Identificador único de transacción
2. Fecha de la transacción
3. Hora exacta de la transacción
4. Código de ubicación
5. Nombre descriptivo de la ubicación
6. Categoría de producto
7. Tipo de producto
8. Detalle específico del producto
9. Cantidad de unidades vendidas
10. Precio unitario del producto
11. Valor monetario total de la línea de transacción

Este nivel de granularidad permite análisis multidimensionales que cruzan variables temporales, geográficas y de producto.

---

## Sección B: Preguntas de Análisis

### Preguntas Guiadas

#### Pregunta 1: Análisis de Comportamiento Temporal de Ventas

**¿Cómo se comportan las ventas a lo largo del tiempo y en diferentes momentos del día?**

Esta pregunta busca identificar patrones de demanda tanto a nivel macro (tendencias mensuales, semanales) como micro (variaciones por hora del día). El análisis temporal permite determinar si Maven Roasters experimenta estacionalidad, ciclos predecibles o tendencias de crecimiento o declive. A nivel intradiario, permite identificar las horas pico de demanda y los períodos de baja actividad.

**Importancia estratégica:**

La comprensión de los patrones temporales es fundamental para la asignación eficiente de recursos humanos, gestión de inventario y planificación de capacidad operacional. Permite optimizar turnos de personal, minimizar tiempos de espera en horas críticas y reducir costos operacionales en períodos de baja demanda.

**Enfoque analítico:**

- Análisis de series temporales utilizando las columnas de fecha y hora
- Agregación de ingresos totales por diferentes períodos (día, semana, mes)
- Segmentación por franjas horarias para identificar picos de demanda
- Evaluación de volumen de transacciones versus valor promedio por período
- Identificación de días de la semana con mayor y menor actividad

#### Pregunta 2: Análisis de Rendimiento del Portafolio de Productos

**¿Qué productos y categorías impulsan el negocio y cuáles tienen bajo rendimiento?**

Esta pregunta evalúa la contribución diferencial de cada elemento del portafolio. Identifica los productos estrella que generan la mayor parte de los ingresos, los productos de volumen alto pero margen bajo, y aquellos con baja rotación que podrían estar consumiendo recursos sin generar valor proporcional.

**Importancia estratégica:**

El análisis de portafolio permite decisiones informadas sobre continuidad, descontinuación o promoción de productos específicos. Identificar los generadores de valor permite enfocar esfuerzos de marketing y operacionales. Detectar productos de bajo rendimiento libera recursos y simplifica operaciones.

**Enfoque analítico:**

- Clasificación de productos por contribución a ingresos totales utilizando columnas de categoría, tipo y detalle de producto
- Análisis de volumen de transacciones versus valor monetario generado
- Identificación de categorías con mayor penetración en la base de transacciones
- Evaluación de precios promedio por categoría y tipo de producto
- Análisis Pareto para determinar la concentración de ingresos en pocos productos

#### Pregunta 3: Análisis Comparativo de Dinámicas por Ubicación

**¿Las tres ubicaciones operan bajo la misma lógica o cada una responde a una dinámica diferente?**

Esta pregunta evalúa la homogeneidad operacional entre las tres cafeterías. Determina si las ubicaciones son intercambiables o si cada una tiene características únicas que requieren estrategias diferenciadas. Examina diferencias en mix de productos, patrones temporales, volumen de ventas y comportamiento del cliente.

**Importancia estratégica:**

Comprender las diferencias entre ubicaciones permite estrategias de gestión adaptadas a cada realidad local. Si las ubicaciones son homogéneas, se justifica estandarización operacional. Si son heterogéneas, requieren gestión descentralizada y autonomía táctica. Esta información es crítica para decisiones de expansión futura.

**Enfoque analítico:**

- Comparación de ingresos totales y volumen de transacciones por ubicación
- Análisis de mix de productos vendidos en cada cafetería
- Evaluación de patrones temporales específicos por ubicación
- Comparación de ticket promedio entre ubicaciones
- Identificación de productos con rendimiento diferencial por ubicación

### Preguntas Auto-Identificadas

#### Pregunta 4: Análisis de Composición y Valor del Ticket de Compra

**¿Cuál es la estructura típica del ticket de compra en Maven Roasters y cómo varía la composición de artículos por transacción?**

Esta pregunta explora el comportamiento de compra del cliente a nivel de transacción individual. Busca determinar cuántos artículos compra típicamente un cliente en una visita, si existen patrones de compra complementaria (por ejemplo, café más alimento), y cómo el número de artículos por transacción se relaciona con el valor monetario total.

**Relevancia para la gerencia de Maven Roasters:**

El análisis de composición de ticket es crítico para estrategias de upselling y cross-selling. Si la mayoría de transacciones contienen un solo artículo, existe oportunidad para incrementar el valor promedio mediante bundling o recomendaciones. Si existen combinaciones recurrentes de productos, pueden diseñarse ofertas específicas. Este conocimiento impacta directamente en el ingreso por cliente sin incrementar tráfico.

**Columnas y métricas a utilizar:**

- Agrupación por identificador de transacción para contar artículos por ticket
- Suma de cantidades vendidas por transacción individual
- Cálculo de valor monetario total por transacción
- Análisis de correlación entre número de artículos y valor del ticket
- Identificación de combinaciones frecuentes de categorías en la misma transacción
- Segmentación de transacciones por tamaño (mono-producto, multi-producto)

#### Pregunta 5: Análisis de Concentración de Ingresos y Riesgo de Dependencia

**¿Qué porcentaje de los ingresos totales depende de un número limitado de productos específicos y representa esto un riesgo de concentración para el negocio?**

Esta pregunta evalúa la distribución de ingresos a través del portafolio completo de productos. Busca determinar si Maven Roasters presenta alta concentración de ingresos en pocos SKUs, lo que representaría vulnerabilidad ante cambios en preferencias o disponibilidad. Examina la diversificación real del modelo de negocio más allá del número nominal de productos ofrecidos.

**Relevancia para la gerencia de Maven Roasters:**

La concentración excesiva de ingresos representa un riesgo estratégico significativo. Si un producto individual o un grupo pequeño de productos genera la mayoría de los ingresos, cualquier disrupción (cambio de proveedor, aumento de costos, cambio en tendencias de consumo) puede impactar desproporcionadamente el negocio completo. Por otro lado, una distribución demasiado fragmentada puede indicar falta de productos diferenciadores. Este análisis permite balancear diversificación con enfoque estratégico.

**Columnas y métricas a utilizar:**

- Agregación de ingresos totales por detalle específico de producto
- Cálculo de participación porcentual de cada producto en ingresos totales
- Aplicación de análisis de curva ABC o regla 80/20
- Identificación del número de productos que generan 50%, 80% y 95% de ingresos
- Evaluación de concentración a nivel de categoría y tipo de producto
- Cálculo de índice de Herfindahl-Hirschman para medir concentración de mercado interno
