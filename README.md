
<div style="text-align: center;">
  <img src="https://github.com/Hack-io-Data/Imagenes/blob/main/01-LogosHackio/logo_naranja@4x.png?raw=true" alt="logo" />
</div>

# Proyecto: Predicción de Retención de Empleados 🏢


## Introducción

Esta vez te toca trabajar en Recursos Humanos y te enfrentas a uno de los mayores dolores de cabeza de cualquier empresa: la rotación de empleados. ¿Por qué algunas personas deciden quedarse mientras otras se van? ¿Será el salario? ¿Las horas extra? ¿La relación con su jefe?

En este proyecto, usarás datos recopilados de una empresa ficticia (sí, no te preocupes, no es información confidencial) que incluye desde encuestas de satisfacción hasta métricas de desempeño y horarios laborales. Tu tarea será desentrañar patrones, analizar tendencias y construir un modelo que pueda predecir si un empleado permanecerá o decidirá decir adiós.

Pero esto no es solo sobre números y gráficos; se trata de entender cómo las decisiones empresariales impactan la vida de las personas y cómo, con un poco de análisis, podríamos ayudar a las empresas a ser mejores lugares para trabajar. Así que prepárate para explorar datos, ensuciarte las manos con algoritmos y, quién sabe, tal vez descubrir el secreto para mantener a los empleados felices y comprometidos.

Este proyecto no solo es un ejercicio técnico, también es un entrenamiento toma de decisiones basadas en datos. 

## Objetivo del Proyecto

El principal desafío de este proyecto es abordar una de las preguntas más importantes para cualquier departamento de Recursos Humanos: ¿qué empleados tienen más probabilidades de quedarse en la empresa y cuáles podrían decidir irse? Para lograr esto, tu trabajo será construir un modelo de machine learning capaz de predecir si un empleado permanecerá en la empresa o decidirá marcharse. 

El enfoque no solo será técnico. A través del análisis de los datos, deberás identificar cuáles son los factores más influyentes en la retención o rotación del personal. Por ejemplo (estos son solo algunos ejemplos, pero puedes hacerte todas las preguntas que quieras para entender bien los datos e identificar los factores más influyentes):

- ¿Es la satisfacción laboral un predictor clave?

- ¿Tienen más probabilidades de irse quienes trabajan largas horas o aquellos con relaciones tensas con sus jefes?

- ¿Qué papel juegan las promociones o los aumentos de salario?

Tu modelo debe ser capaz de responder estas preguntas y ofrecer predicciones precisas que puedan usarse para tomar decisiones informadas. Esto significa que, además de construir un modelo que funcione, **deberás interpretar sus resultados y proponer estrategias basadas en ellos**.

## Descripción de los Datos

- `employee_survey_data.csv`: Contiene datos de encuestas sobre la satisfacción y percepción de los empleados.

- `general_data.csv`: Información demográfica y general de los empleados

- `manager_survey_data.csv`: Encuestas realizadas por los supervisores sobre desempeño y satisfacción.

**NOTA** Dentro de la carpeta de datos, encontrarás un archivo `diccionario-datos` donde encontrarás la descripción detallada de todos los datos aportados. 

## Estructura del Proyecto

Este proyecto está dividido en varias etapas las cuales son: 

1. **Exploración de los datos**: Debes analizar las características disponibles y tomar las primeras decisiones clave:

    - **Limpieza de datos**: Busca y maneja valores nulos, duplicados o inconsistencias que puedan afectar la calidad de tu análisis.

    - **Entendimiento de las variables**: Identifica las variables relevantes, sus tipos y posibles relaciones entre ellas. Por ejemplo, ¿qué variables parecen estar relacionadas con la decisión de quedarse o irse?

    - **Correlaciones y tendencias**: Usa herramientas estadísticas básicas para identificar correlaciones significativas. Esto te ayudará a entender qué factores tienen mayor impacto.




2. **Análisis Exploratorio de Datos (EDA)**: Debes profundizar en los datos para descubrir patrones y tendencias importantes. Esto incluye:

    - **Visualización de datos**: Crea gráficos como histogramas, boxplots y gráficos de dispersión para entender mejor la distribución de las variables y sus relaciones.

    - **Impacto de las variables**: Examina cómo factores como la satisfacción laboral, las horas trabajadas o el nivel de desempeño están relacionados con la retención.

    - **Detección de sesgos**: Asegúrate de que no haya sesgos en los datos que puedan influir en los resultados del modelo.



3. **Preparación del Modelo**: Antes de construir el modelo, debes preparar los datos adecuadamente. Este paso incluye:

    - **División de datos**: Divide el conjunto de datos en conjuntos de entrenamiento (training) y prueba (test) para evaluar el modelo de manera justa.

    - **Preprocesamiento**:

        - Normaliza o estandariza las variables si es necesario (por ejemplo, escalando variables como salarios o horas trabajadas).

        - Convierte variables categóricas en un formato utilizable, como codificación one-hot.

        - **Balanceo de clases**: Si las categorías "se queda" y "se va" están desequilibradas, considera técnicas como sobremuestreo o     submuestreo.


4. **Construcción del Modelo**: Es hora de construir y evaluar diferentes algoritmos de machine learning. Aquí deberás:

    - **Probar múltiples algoritmos**: Implementa al menos dos algoritmos de clasificación, como árboles de decisión, regresión logística o bosques aleatorios.

    - **Evaluar el desempeño**: Usa métricas como:
        - **Precisión**: ¿Qué tan bien clasifica el modelo?

        - **F1-score**: ¿Qué tan equilibrado es entre precisión y sensibilidad?

        - **Matriz de confusión**: Analiza los verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.

        - etc. 

    - **Optimización del modelo**: Ajusta hiperparámetros para mejorar el desempeño, utilizando herramientas como GridSearchCV.


5. **Conclusiones**: La etapa final consiste en analizar los resultados obtenidos y proponer estrategias prácticas. Esto incluye:

    - **Resumen de hallazgos**: ¿Cuáles son los factores más relevantes que afectan la retención de empleados? ¿Qué variables tienen mayor peso en las predicciones del modelo?

    - **Recomendaciones**: Basándote en el análisis y el modelo, sugiere acciones que la empresa podría tomar para mejorar la retención.

    - Por ejemplo, ¿se deben ajustar las políticas de horarios flexibles? ¿O tal vez invertir más en programas de bienestar?

    - **Evaluación del proyecto**: Reflexiona sobre los puntos fuertes y débiles de tu modelo y cómo podría mejorarse.


## Como Entregar el Proyecto

La entrega del proyecto se realizará a través de una **issue en GitHub**, trabajando en un repositorio propio en tu cuenta personal. Los pasos que deberás seguir para hacer la entrega del proyecto son:


- **Crear un nuevo repositorio en tu cuenta de GitHub:**

   - Crea un nuevo repositorio llamado `Proyecto8-NombreProyecto`. Este nombre es obligatorio, no podremos llamarlo de otra forma. 

   - Configuralo como público. 


- **Desarrolla el proyecto:**

   - Implementa el código para la resolución del problema.

   - Recuerda hacer commits regulares mientras avanzas en el desarrollo:

     ```bash
     git add .
     git commit -m "Descripción del avance"
     git push
     ```


- **Crear una issue en el repositorio original del curso:**

   - Ve al repositorio original del curso y dirígete a la pestaña de **Issues**.

- **Abrir una nueva issue para tu entrega:**

   - Haz clic en **New Issue** y llena los siguientes campos:

     - **Título:** Usa el formato "Entrega Proyecto: ProyectoClasificacion - [Tu Nombre]".

     - **Descripción:** En la descripción, incluye:

       - Una breve explicación de tu proyecto.

       - Instrucciones para ejecutar tu código (si aplica).

       - Un enlace a tu repositorio personal donde está alojado el proyecto.


## 🚀 Entrega del Proyecto 🚀

**Fecha y hora límite:**

🗓️ **Lunes a las 9:00 AM.**


**Nota importante:**

⚠️ **Todos los proyectos que sean entregados o modificados después de la hora límite (lunes a las 9:00 AM) se considerarán como no entregados.** Por favor, asegúrate de completar y enviar tu trabajo a tiempo para evitar problemas.


# 🎤 Presentación de Proyectos 🎤

El lunes tendremos las **presentaciones de los proyectos**. La dinámica será la siguiente:

- De forma **aleatoria**, seleccionaremos entre **3 y 5 alumnos** para presentar su proyecto.

- Cada alumno tendrá **5 minutos** para explicar su proyecto y hacer una demo en vivo. Durante este tiempo podrán mostrar cómo funciona su juego y resaltar las características principales.

**Detalles importantes:**

- Es importante que lleguéis puntuales, ya que comenzaremos las presentaciones de inmediato.

- Asegúrate de que tu código esté listo y funcional para la demo.

- Todos debemos estar preparados para presentar, ya que la selección será completamente aleatoria.
