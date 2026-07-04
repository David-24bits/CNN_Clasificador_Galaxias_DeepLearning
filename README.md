# 🌌 Deep Galaxias - Clasificación Morfológica con CNN Especializada

**Universidad Tecnológica Metropolitana (UTEM)**  
Facultad de Ingeniería - Deep Learning - Proyecto Final  
**Desarrollado por:** David Sepúlveda - Vicente Escudero  
**Fecha:** Diciembre 2025

## 📋 Descripción

Proyecto de investigación que desarrolla un sistema automatizado de clasificación morfológica de galaxias utilizando arquitecturas CNN especializadas. Este trabajo surge de la necesidad de modernizar herramientas astronómicas obsoletas (como astroNN) que no aprovechan las capacidades del hardware y frameworks actuales.

**🎯 Logro Principal:** SimpleCNN especializada alcanza **68.16% accuracy**, superando EfficientNetV2 (transfer learning) en **+32.7%** - demostrando que la especialización de dominio supera enfoques genéricos en clasificación morfológica astronómica.

**📊 Dataset:** Galaxy10 DECals (17,736 imágenes → 21,894 post-augmentation) clasificadas en 10 tipos morfológicos del DESI Legacy Imaging Surveys.

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.11 o superior
- Git (opcional)

### 1. Clonar/Descargar el Proyecto
```bash
# Si usas Git
git clone <repository-url>
cd deep_galaxias

# O descargar y extraer el ZIP
```

### 2. Crear Entorno Virtual
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar Dependencias
```bash
# Instalación completa
pip install -r requirements.txt

# Verificar instalación PyTorch con GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"

# Para desarrollo (opcional)
pip install jupyter notebook ipykernel
```

## 🏃‍♂️ Ejecución

### Aplicación Web Streamlit
```bash
# Activar entorno virtual (si no está activado)
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/macOS

# Ejecutar aplicación
streamlit run app.py
```

La aplicación estará disponible en:
- **Local**: http://localhost:8501
- **Red**: http://[tu-ip]:8501

### Notebook de Entrenamiento
```bash
# Iniciar Jupyter
jupyter notebook entrenamiento_completo.ipynb
```

## 📁 Estructura del Proyecto

```
deep_galaxias/
├── app.py                           # Aplicación principal Streamlit
├── entrenamiento_completo.ipynb     # Notebook completo (106 celdas)
├── Galaxy10_DECals.h5               # Dataset Galaxy10 (original)
├── requirements.txt                 # Dependencias del proyecto
├── README.md                        # Documentación principal
├── pauta.txt                        # Criterios de evaluación
├── logo_utem.jpeg                   # Logo institucional
├── .venv/                           # Entorno virtual Python
├── cnn_2/                           # SimpleCNN (modelo ganador)
│   ├── cnn_galaxy10_20251027_225658_acc_0.6816.pth
│   ├── metrics/                     # Métricas detalladas
│   └── plots/                       # Visualizaciones de entrenamiento
└── fine/                            # EfficientNetV2 (baseline)
    ├── fine_efficientnet_v2_20251027_202622_acc_0.5133.pth
    ├── metrics/                     # Métricas comparativas
    └── plots/                       # Curvas de aprendizaje
```

## 🎯 Características del Proyecto

### **Técnicas y Metodología**
- **🤖 SimpleCNN Especializada**: Arquitectura de 5 bloques (32→512 filtros) optimizada para morfología galáctica
- **🔄 EfficientNetV2 Baseline**: Transfer learning como comparación estado del arte
- **⚖️ Balanceado Híbrido**: Data augmentation selectivo + class weights para desbalance
- **📊 Validación Rigurosa**: Split estratificado 70/15/15 con múltiples métricas
- **🔬 Reproducibilidad**: Seeds fijos (42), early stopping, protocolo documentado

### **Aplicación Web**
- **📱 Interfaz Streamlit**: Clasificación interactiva en tiempo real
- **📊 Visualizaciones**: Plotly para gráficos científicos profesionales
- **🎨 Diseño UTEM**: Colores corporativos e identidad institucional
- **⚡ Optimización**: Cache inteligente y carga eficiente de modelos

### **Innovación Científica**
- **🌟 Modernización**: Supera herramientas legacy (astroNN) con PyTorch 2.x
- **🎯 Especialización**: Demuestra ventaja de dominio específico vs genérico
- **📈 Escalabilidad**: Pipeline preparado para surveys masivos (LSST, Euclid)

## 📊 Clases de Galaxias

0. **Disturbed Galaxies** - Galaxias Perturbadas
1. **Merging Galaxies** - Galaxias en Fusión
2. **Round Smooth Galaxies** - Galaxias Suaves Redondas
3. **In-between Round Smooth** - Intermedias Suaves
4. **Cigar Shaped Smooth** - Galaxias Alargadas
5. **Barred Spiral Galaxies** - Espirales con Barra
6. **Unbarred Tight Spiral** - Espirales Cerradas
7. **Unbarred Loose Spiral** - Espirales Abiertas
8. **Edge-on without Bulge** - De Canto sin Bulbo
9. **Edge-on with Bulge** - De Canto con Bulbo

## 🛠️ Stack Tecnológico Moderno

### **Frameworks de Deep Learning**
- **PyTorch 2.9.1**: Framework principal (moderniza astroNN basado en TensorFlow 1.x)
- **torchvision 0.24.1**: Transformaciones y augmentation optimizado
- **CUDA 12.x**: Soporte GPU moderno para entrenamiento acelerado

### **Aplicación Web y Visualización**
- **Streamlit 1.51.0**: Interfaz web interactiva y responsive
- **Plotly 6.5.0**: Gráficos científicos interactivos
- **Matplotlib 3.10.7 + Seaborn 0.13.2**: Visualizaciones estáticas profesionales

### **Procesamiento de Datos Astronómicos**
- **NumPy 2.3.5 + Pandas 2.3.3**: Análisis numérico y manipulación de datos
- **H5PY 3.15.1**: Manejo eficiente de dataset Galaxy10 DECals
- **Pillow 12.0.0**: Procesamiento de imágenes astronómicas
- **scikit-learn**: Métricas de evaluación y división de datos

**🚀 Ventaja Tecnológica:** Stack completamente actualizado vs herramientas legacy, aprovechando hardware moderno (GPUs >24GB, mixed precision, DataLoader optimizado).

## 📈 Rendimiento del Modelo

| Métrica | SimpleCNN | EfficientNetV2 | Mejora Relativa |
|---------|-----------|----------------|----------------|
| **Test Accuracy** | **68.16%** | 51.33% | **+32.7%** |
| **F1-Weighted** | **0.67** | 0.49 | **+36.7%** |
| **ROC-AUC Macro** | **0.78** | 0.65 | **+20.0%** |
| **mAP** | **0.71** | 0.58 | **+22.4%** |
| **Paradigma** | Especialización | Transfer Learning | **Dominio específico** |

**🏆 Resultado:** La especialización arquitectónica supera significativamente el transfer learning para clasificación morfológica de galaxias.

## 🐛 Solución de Problemas

### Error: "Weights only load failed" (PyTorch 2.6+)
```python
# Solución aplicada en el código:
checkpoint = torch.load(model_path, weights_only=False)
# Causa: PyTorch 2.6+ cambió default a weights_only=True por seguridad
```

### Error: "streamlit command not found"
```bash
# Activar entorno virtual primero:
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/macOS

# Luego ejecutar:
streamlit run app.py
# O usar ruta completa:
python -m streamlit run app.py
```

### Problemas de GPU/CUDA
```python
# Verificar instalación CUDA:
import torch
print(f'CUDA disponible: {torch.cuda.is_available()}')
print(f'Dispositivo: {torch.cuda.get_device_name(0)}')

# El código detecta automáticamente:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Error: "Module not found" (Jupyter)
```bash
# Instalar kernel en entorno virtual:
python -m ipykernel install --user --name=.venv --display-name="Deep Galaxias"
# Luego seleccionar kernel en Jupyter
```

### Problemas de Memoria
- **GPU insuficiente**: Reducir batch_size en código (línea 64: `batch_size = 32`)
- **RAM limitada**: Usar `pin_memory=False` en DataLoader
- **Dataset grande**: El código carga eficientemente con `lazy loading`

## 🔬 Resultados Científicos

### **Hallazgos Principales**
1. **Especialización > Transfer Learning**: 32.7% mejora demuestra valor de arquitecturas domain-specific
2. **Coherencia Astronómica**: Patrones de error alineados con ambigüedad morfológica real
3. **Escalabilidad Moderna**: Pipeline preparado para procesamiento masivo (>10⁶ galaxias)
4. **Reproducibilidad**: Protocolo científico riguroso con validación estadística

### **Validación Estadística**
- **Significancia**: p < 0.001 (test McNemar)
- **IC 95%**: [14.2%, 19.5%] diferencia en accuracy
- **Consistencia**: Mejora replicada en todas las métricas (F1, ROC-AUC, mAP)

## 🚀 Trabajo Futuro

### **Extensiones Científicas**
- **Multi-Survey**: Validación en DES, HSC, KiDS
- **Vision Transformers**: Swin-T para dependencias espaciales largas
- **Multimodal**: Integración fotometría + morfología + redshift
- **Aplicación Real**: Clasificación automática para Euclid, LSST, Roman

### **Otros Dominios Astronómicos**
- **Clasificación Estelar**: Tipos espectrales, variables, binarias
- **Detección de Exoplanetas**: Tránsitos en curvas de luz
- **Objetos Transitorios**: Supernovas, kilonovas, TDEs

## 👥 Información Académica

**Desarrolladores:** David Sepúlveda - Vicente Escudero  
**Institución:** Universidad Tecnológica Metropolitana (UTEM)  
**Facultad:** Ingeniería - Asignatura Deep Learning  
**Fecha:** Diciembre 2025  

### **Declaración de Herramientas IA**
- **GitHub Copilot**: Autocompletado código (~15% líneas)
- **ChatGPT-4**: Consultas específicas métricas astronómicas
- **Claude Sonnet 4**: Estructuración documentación
- **Trabajo Original**: 85% desarrollo propio del equipo

## 📄 Licencia

Proyecto académico - UTEM 2025  
Código disponible para fines educativos y científicos

---

**🌟 ¡Modernizando la Astronomía con Deep Learning Especializado!**  
*"Donde la especialización de dominio supera la generalización: 68.16% vs 51.33%"*