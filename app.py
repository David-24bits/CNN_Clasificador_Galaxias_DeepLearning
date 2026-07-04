import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y DISEÑO UTEM
# ==============================================================================
st.set_page_config(
    page_title="Deep Galaxy - UTEM",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colores UTEM
UTEM_BLUE = "#002e5d"
UTEM_LIGHT_BLUE = "#00A499"
BG_COLOR = "#FFFFFF"

# CSS Personalizado Avanzado
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Roboto', sans-serif;
    }}

    /* Encabezados */
    h1, h2, h3 {{
        color: {UTEM_BLUE};
        font-weight: 700;
    }}

    /* Botones personalizados */
    .stButton>button {{
        width: 100%;
        background-color: {UTEM_BLUE};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {UTEM_LIGHT_BLUE};
        color: white;
        transform: scale(1.02);
    }}

    /* Métricas estilizadas */
    [data-testid="stMetricValue"] {{
        font-size: 1.5rem !important;
        color: {UTEM_BLUE} !important;
    }}
    
    /* Contenedores personalizados */
    .info-box {{
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {UTEM_BLUE};
        margin-bottom: 20px;
        height: 100%; /* Para igualar altura si es necesario */
    }}

    .success-box {{
        background-color: #e6fffa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid {UTEM_LIGHT_BLUE};
        text-align: center;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        color: {UTEM_BLUE};
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {UTEM_BLUE} !important;
        color: white !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. ARQUITECTURA DEL MODELO (SimpleCNN)
# ==============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.block1 = self._make_block(3, 32)
        self.block2 = self._make_block(32, 64)
        self.block3 = self._make_block(64, 128)
        self.block4 = self._make_block(128, 256)
        self.block5 = self._make_block(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512 * 1 * 1, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def _make_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# ==============================================================================
# 3. FUNCIONES UTILITARIAS Y CARGA
# ==============================================================================

@st.cache_resource
def load_model_logic(model_path, device):
    """Carga el modelo y sus pesos de manera segura."""
    try:
        model = SimpleCNN(num_classes=10)
        
        # Para PyTorch 2.6+: usar weights_only=False para archivos de confianza
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            # Mapeo de clases estándar de Galaxy10 DECals
            default_classes = [
                'Disturbed Galaxies', 'Merging Galaxies', 'Round Smooth Galaxies',
                'In-between Round Smooth Galaxies', 'Cigar Shaped Smooth Galaxies',
                'Barred Spiral Galaxies', 'Unbarred Tight Spiral Galaxies',
                'Unbarred Loose Spiral Galaxies', 'Edge-on Galaxies without Bulge',
                'Edge-on Galaxies with Bulge'
            ]
            class_names = checkpoint.get('class_names', default_classes)
        else:
            model.load_state_dict(checkpoint)
            class_names = [
                'Disturbed Galaxies', 'Merging Galaxies', 'Round Smooth Galaxies',
                'In-between Round Smooth Galaxies', 'Cigar Shaped Smooth Galaxies',
                'Barred Spiral Galaxies', 'Unbarred Tight Spiral Galaxies',
                'Unbarred Loose Spiral Galaxies', 'Edge-on Galaxies without Bulge',
                'Edge-on Galaxies with Bulge'
            ]
            
        model.to(device)
        model.eval()
        return model, class_names
    except Exception as e:
        return None, str(e)

def process_image(image):
    """Transformación de imagen estándar para el modelo."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def plot_probabilities(df_probs):
    """Genera un gráfico de barras horizontal elegante con colores UTEM."""
    fig = go.Figure(go.Bar(
        x=df_probs['Probabilidad'],
        y=df_probs['Clase'],
        orientation='h',
        text=df_probs['Probabilidad'].apply(lambda x: f"{x*100:.1f}%"),
        textposition='outside',
        marker=dict(
            color=df_probs['Probabilidad'],
            colorscale=[[0, '#a3cced'], [1, UTEM_BLUE]], # Gradiente azul
            cmin=0, cmax=1
        )
    ))
    
    fig.update_layout(
        title="<b>Distribución de Confianza</b>",
        title_font_color=UTEM_BLUE,
        xaxis=dict(range=[0, 1.15], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        height=400
    )
    return fig

# ==============================================================================
# 4. FUNCIÓN PARA MOSTRAR PRESENTACIÓN
# ==============================================================================

def show_presentation():
    """Mostrar presentación integrada en Streamlit"""
    
    st.title("🎯 Deep Galaxias - Presentación Final")
    st.markdown("---")
    
    # Selector de slides
    slide_options = [
        "1. 🎯 Portada y Pitch",
        "2. 🌍 Contexto y Objetivos", 
        "3. 📊 Datos y Gobernanza",
        "4. 🔧 Metodología",
        "5. 🏆 Resultados",
        "6. 🔍 Interpretabilidad",
        "7. ⚠️ Robustez",
        "8. 💡 Conclusiones",
        "9. 📚 Bibliografía",
        "10. ✍️ Autoría"
    ]
    
    selected_slide = st.selectbox("📊 Seleccionar Slide:", slide_options)
    
    # Mostrar contenido según slide seleccionado
    if "1. 🎯 Portada" in selected_slide:
        show_slide_portada()
    elif "2. 🌍 Contexto" in selected_slide:
        show_slide_contexto()
    elif "3. 📊 Datos" in selected_slide:
        show_slide_datos()
    elif "4. 🔧 Metodología" in selected_slide:
        show_slide_metodologia()
    elif "5. 🏆 Resultados" in selected_slide:
        show_slide_resultados()
    elif "6. 🔍 Interpretabilidad" in selected_slide:
        show_slide_interpretabilidad()
    elif "7. ⚠️ Robustez" in selected_slide:
        show_slide_robustez()
    elif "8. 💡 Conclusiones" in selected_slide:
        show_slide_conclusiones()
    elif "9. 📚 Bibliografía" in selected_slide:
        show_slide_bibliografia()
    elif "10. ✍️ Autoría" in selected_slide:
        show_slide_autoria()

def show_slide_portada():
    """Slide 1: Portada y Pitch"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 🎯 Deep Galaxias
        ### Clasificación Morfológica Automática con CNN Especializada
        
        **👥 Equipo:** David Sepúlveda - Vicente Escudero  
        **🏫 UTEM** - Facultad de Ingeniería - Deep Learning
        
        **📊 Dataset:** Galaxy10 DECals (17,736 galaxias)
        
        ---
        
        ### 💡 Frase de Valor
        > *Automatizamos la clasificación de 10 tipos morfológicos de galaxias 
        > con CNN especializada, alcanzando **68.16% accuracy** y superando 
        > transfer learning en **+32.7%**, revolucionando el análisis astronómico masivo.*
        """)

def show_slide_contexto():
    """Slide 2: Contexto y Objetivos"""
    st.markdown("""
    ## 🌍 Contexto, Objetivos e Hipótesis
    
    ### Marco del Dominio
    - **Problema Legacy:** Herramientas como astroNN obsoletas (TensorFlow 1.x, 2018-2019)
    - **Big Data Astronómico:** LSST >20TB/noche, DESI >1 billón píxeles
    - **Gap Tecnológico:** 5+ años entre capacidades hardware/software astronómico
    
    ### 🎯 Objetivos Específicos
    1. **Principal:** Desarrollar sistema CNN especializado para clasificación morfológica
    2. **Comparativo:** Especialización vs Transfer Learning (EfficientNetV2)
    3. **Modernización:** Pipeline PyTorch 2.x aprovechando hardware moderno
    
    ### 📊 KPIs Establecidos
    - **Primario:** Test Accuracy > 60% (vs ~40% métodos legacy)
    - **Secundarios:** F1-weighted, ROC-AUC macro, mAP > 0.60
    
    ### 🔬 Hipótesis Central
    > *"Arquitecturas CNN especializadas para dominio astronómico superarán 
    > modelos pre-entrenados genéricos en clasificación morfológica."*
    """)

def show_slide_datos():
    """Slide 3: Datos y Gobernanza"""
    st.markdown("""
    ## 📊 Datos y Gobernanza
    
    ### Dataset Galaxy10 DECals
    - **Fuente:** DESI Legacy Imaging Surveys (2024)
    - **Tamaño:** 17,736 → 21,894 imágenes post-augmentation
    - **Formato:** 256×256×3 píxeles (bandas g,r,z astronómicas)
    - **Clases:** 10 tipos morfológicos balanceados
    
    ### 🔄 Split Estratificado
    ```
    Train: 70% | Validation: 15% | Test: 15%
    Estratificado por clase + random_state=42
    ```
    
    ### ⚖️ Gestión de Desbalance
    - **Problema:** Clase 4 (Cigar) ~1,286 vs Clase 1 (Merging) ~1,871
    - **Solución:** Augmentation selectivo + pesos balanceados
    
    ### 🔒 Consideraciones Éticas
    - **Sesgo de Selección:** Limitado a surveys ópticos
    - **Transparencia:** Pipeline documentado y reproducible
    """)

def show_slide_metodologia():
    """Slide 4: Metodología"""
    st.markdown("""
    ## 🔧 Metodología y Diseño Experimental
    
    ### 🏗️ Arquitecturas Implementadas
    
    #### SimpleCNN (Especializada)
    - **Diseño:** 5 bloques Conv2D progresivos (32→64→128→256→512)
    - **Componentes:** BatchNorm + ReLU + MaxPool + Dropout(0.5)
    - **Parámetros:** ~2.8M entrenables
    
    #### EfficientNetV2-S (Transfer Learning)
    - **Base:** Pre-entrenado ImageNet1K (22M parámetros)
    - **Estrategia:** Features congelados + fine-tuning clasificador
    - **Parámetros:** ~580K entrenables
    
    ### ⚙️ Configuración de Entrenamiento
    - **Optimizador:** AdamW (lr=1e-4, weight_decay=1e-4)
    - **Loss:** CrossEntropyLoss con class weights
    - **Early Stopping:** 35 épocas patience
    """)

def show_slide_resultados():
    """Slide 5: Resultados"""
    st.markdown("""
    ## 🏆 Resultados y Comparación
    
    ### Resultados Principales
    """)
    
    # Tabla de resultados
    results_data = {
        "Modelo": ["SimpleCNN", "EfficientNetV2", "Mejora"],
        "Test Accuracy": ["68.16%", "51.33%", "+32.7%"],
        "F1-Weighted": ["0.67", "0.49", "+36.7%"],
        "ROC-AUC": ["0.78", "0.65", "+20.0%"],
        "mAP": ["0.71", "0.58", "+22.4%"]
    }
    
    df_results = pd.DataFrame(results_data)
    st.table(df_results)
    
    st.markdown("""
    ### 📊 Significancia Estadística
    - **Diferencia:** 16.83 puntos porcentuales
    - **p-value:** < 0.001 (altamente significativo)
    
    ### 🎯 Análisis por Grupos Morfológicos
    - **Espirales:** SimpleCNN +28% vs EfficientNet
    - **Lisas/Elípticas:** SimpleCNN +41% vs EfficientNet
    """)

def show_slide_interpretabilidad():
    """Slide 6: Interpretabilidad"""
    st.markdown("""
    ## 🔍 Interpretabilidad y Análisis de Errores
    
    ### Características Aprendidas
    
    #### SimpleCNN - Especialización
    ✅ **Filtros Astronómicos:** Gradientes radiales, asimetrías espirales  
    ✅ **Sin Conflicto:** No hay bias de objetos naturales (ImageNet)  
    ✅ **Morfología Específica:** Optimizado para estructuras galácticas
    
    #### EfficientNetV2 - Transfer Learning
    ❌ **Desajuste de Dominio:** Features naturales ≠ morfologías astronómicas  
    ❌ **Limitación Estructural:** Solo 2.6% parámetros adaptables
    
    ### 🎯 Análisis de Errores
    #### Predicciones Incorrectas Comunes
    1. **Espiral Barrada → No Barrada:** Barras sutiles no detectadas
    2. **Round → Cigar:** Confusión en excentricidad marginal  
    3. **Disturbed → Merging:** Ambigüedad en perturbación
    """)

def show_slide_robustez():
    """Slide 7: Robustez"""
    st.markdown("""
    ## ⚠️ Robustez y Riesgos
    
    ### 💪 Estabilidad del Modelo
    - **Reproducibilidad:** Seeds fijos (42) → resultados consistentes
    - **Hiperparámetros:** Robusto en rangos [1e-5, 1e-3] learning rate
    
    ### Limitaciones Explícitas
    1. **Dominio Específico:** Solo imágenes ópticas (g,r,z)
    2. **Resolución:** 224×224 puede perder detalles finos
    3. **Generalización:** Validación limitada a Galaxy10 DECals
    
    ### 🚨 Riesgos de Despliegue
    - **Falsos Positivos:** Impacto en estudios científicos downstream
    - **Deriva de Datos:** Cambios instrumentales afectan rendimiento
    
    ### 🛡️ Mitigaciones Propuestas
    - **Validación Multi-Survey:** DES, HSC, COSMOS
    - **Human-in-the-Loop:** Revisión experta casos <70% confianza
    """)

def show_slide_conclusiones():
    """Slide 8: Conclusiones"""
    st.markdown("""
    ## 💡 Conclusiones y Recomendaciones
    
    ### ✅ Lecciones Aprendidas
    1. **🎯 Especialización > Transfer Learning:** 68.16% vs 51.33%
    2. **⚡ Modernización Efectiva:** PyTorch 2.x supera herramientas legacy
    3. **🔄 Balanceado Inteligente:** Augmentation selectivo mitiga desbalance
    
    ### ⚖️ Trade-offs Identificados
    - **Simplicidad ↔ Precisión:** SimpleCNN menos compleja, más efectiva
    - **Entrenamiento ↔ Rendimiento:** Desde cero costoso pero superior
    
    ### 🚀 Recomendaciones Inmediatas
    - **Producción:** SimpleCNN como baseline nuevo estándar
    - **Threshold:** >80% confianza automático
    
    ### 🔮 Siguientes Pasos
    1. **Multi-Survey:** Validación DES Y6, HSC PDR3
    2. **Vision Transformers:** Swin-T para dependencias espaciales
    3. **Multimodal:** Fotometría + morfología + redshift
    """)

def show_slide_bibliografia():
    """Slide 9: Bibliografía"""
    st.markdown("""
    ## 📚 Bibliografía y Recursos
    
    ### Referencias Principales
    - **Dieleman et al. (2022):** Galaxy10 DECals Dataset
    - **Tan & Le (2021):** EfficientNetV2: Smaller Models and Faster Training
    - **Domínguez Sánchez et al. (2018):** Transfer learning for galaxy morphology
    
    ### 🔧 Stack Tecnológico
    - **Framework:** PyTorch 2.9.1, scikit-learn, matplotlib
    - **Hardware:** GPU CUDA 12.x, Mixed Precision Training
    
    ### 🌐 Recursos
    - **Dataset:** Galaxy10 DECals public access
    - **Documentación:** Jupyter notebook completo disponible
    """)

def show_slide_autoria():
    """Slide 10: Autoría"""
    st.markdown("""
    ## ✍️ Evidencias de Autoría
    
    ### 🤖 Declaración de Herramientas IA
    - **GitHub Copilot:** Autocompletado código Python/PyTorch (~15% líneas)
    - **ChatGPT-4:** Consultas específicas métricas astronómicas
    - **Claude Sonnet 4:** Estructuración documentación y presentación
    
    ### ✍️ Trabajo Original (85% propio)
    - **Diseño Experimental:** 100% original del equipo
    - **Arquitectura SimpleCNN:** Diseño personalizado astronómico
    - **Pipeline de Datos:** Augmentation selectivo innovador
    
    ### 📊 Evidencias Reproducibles
    - **Seeds Fijos:** random_state=42 en todas las operaciones
    - **Logs Completos:** Training curves, validation metrics guardados
    - **Código Documentado:** 2,500+ líneas con comentarios explicativos
    
    ### 👥 Equipo
    **David Sepúlveda - Vicente Escudero**  
    **UTEM - Facultad de Ingeniería - Deep Learning**
    """)

# ==============================================================================
# 5. INTERFAZ PRINCIPAL
# ==============================================================================

def main():
    # Rutas absolutas seguras
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "cnn_2", "cnn_galaxy10_20251027_225658_acc_0.6816.pth")
    LOGO_PATH = os.path.join(BASE_DIR, "logo_utem.jpeg")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- SIDEBAR ---
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown(f"<h1 style='color:{UTEM_BLUE}; text-align:center;'>UTEM</h1>", unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Panel de Control")
        st.caption(f"Dispositivo de inferencia: **{str(device).upper()}**")
        
        # Carga del modelo
        if os.path.exists(MODEL_PATH):
            model, class_names = load_model_logic(MODEL_PATH, device)
            if model:
                st.success("✅ Modelo Activo: SimpleCNN")
            else:
                st.error(f"Error cargando modelo: {class_names}")
                model = None
        else:
            st.warning(f"⚠️ Archivo no encontrado: {MODEL_PATH}")
            model = None

        st.markdown("---")
        st.markdown("### 📊 Presentación Final")
        
        # Verificar si existe el archivo de presentación
        PRESENTATION_PATH = os.path.join(BASE_DIR, "Deep Galaxias.pptx")
        
        if os.path.exists(PRESENTATION_PATH):
            with open(PRESENTATION_PATH, "rb") as pptx_file:
                pptx_data = pptx_file.read()
            
            st.download_button(
                label="📥 Descargar Presentación",
                data=pptx_data,
                file_name="Deep_Galaxias_Presentacion.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                help="Presentación completa del proyecto en PowerPoint"
            )
            
            # Botón para mostrar slides en la aplicación
            if st.button("🎯 Ver Slides", help="Mostrar presentación integrada"):
                st.session_state['show_presentation'] = True
                
            st.success("✅ Presentación disponible")
        else:
            st.info("📊 Subir 'Deep Galaxias.pptx' al directorio raíz")
        
        st.markdown("---")
        st.markdown("### ℹ️ Acerca de")
        st.info(
            "Desarrollado para el análisis morfológico de galaxias utilizando Deep Learning."
            "\n\n**Facultad de Ingeniería - UTEM**"
        )

    # --- ENCABEZADO PRINCIPAL ---
    st.markdown(f"<h1 style='text-align: center;'>🌌 Clasificador de Galaxias UTEM</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Inteligencia Artificial aplicada a la Astronomía</p>", unsafe_allow_html=True)
    st.write("") # Espaciador

    # --- TABS DE NAVEGACIÓN ---
    tab_info, tab_metrics, tab_demo, tab_presentation = st.tabs(["📖 Sobre el Proyecto", "📊 Métricas y Datos", "🚀 Clasificador en Vivo", "🎯 Presentación"])

    # Verificar si mostrar presentación
    if st.session_state.get('show_presentation', False):
        show_presentation()
        # Botón para volver
        if st.button("← Volver al Clasificador"):
            st.session_state['show_presentation'] = False
            st.rerun()
        return  # Salir para mostrar solo la presentación

    # ----------------------------------------------------------------------
    # TAB 1: INFORMACIÓN DEL PROYECTO (Lorem Ipsum)
    # ----------------------------------------------------------------------
    with tab_info:
        st.markdown("## 📖 Descripción del Proyecto")
        
        col_text, col_img_dec = st.columns([3, 1])
        with col_text:
            st.markdown("""
            **Deep Galaxias: Clasificación Morfológica Automática con CNN Especializada**
    
    Este proyecto desarrolla un sistema de **clasificación morfológica de galaxias** utilizando técnicas avanzadas de Deep Learning, específicamente diseñado para superar las limitaciones de herramientas astronómicas obsoletas como astroNN.
    
    **🎯 Objetivos del Proyecto:**
    - Desarrollar una arquitectura CNN especializada para el dominio astronómico
    - Superar significativamente el rendimiento de transfer learning tradicional
    - Establecer un nuevo estándar tecnológico aprovechando hardware moderno
    
    **🔬 Metodología:**
    Implementamos y comparamos dos enfoques: una **SimpleCNN especializada** diseñada específicamente para morfología galáctica, versus **EfficientNetV2** con transfer learning. Nuestros resultados demuestran que la especialización de dominio alcanza **68.16% de accuracy**, superando el transfer learning tradicional en **+32.7%**.
    
    **📊 Dataset:** 
    Utilizamos **Galaxy10 DECals** del DESI Legacy Imaging Surveys, conteniendo 17,736 imágenes de galaxias clasificadas en 10 tipos morfológicos (espirales, elípticas, irregulares, merging, etc.).
    
    **🚀 Impacto:**
    Esta investigación establece las bases para el procesamiento masivo de surveys astronómicos futuros como LSST, permitiendo la catalogación automática de millones de galaxias con coherencia científica validada.
    
    **👥 Desarrollado por:** David Sepúlveda - Vicente Escudero  
    **🏫 Universidad:** UTEM - Facultad de Ingeniería - Asignatura Deep Learning
    """)
            
            
        with col_img_dec:

            # Iconos decorativos para esta sección
            st.markdown(f"""
            <div style="background-color: {UTEM_BLUE}; padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
                <h3 style="color: white; margin-bottom: 0;">Deep Learning</h3>
                <div style="font-size: 40px;">🧠</div>
            </div>
            <div style="background-color: {UTEM_LIGHT_BLUE}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h3 style="color: white; margin-bottom: 0;">Galaxy10</h3>
                <div style="font-size: 40px;">🔭</div>
            </div>
            """, unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # TAB 2: MÉTRICAS Y DATOS (Con imagen de muestras restaurada)
    # ----------------------------------------------------------------------
    with tab_metrics:
        
        # 1. Descripción del Dataset e IMAGEN RESTAURADA
        st.markdown("## 1. Dataset: Galaxy10 DECals")
        
        # Restauramos las columnas para poner la imagen a la derecha
        col_desc, col_sample = st.columns([2, 1])
        
        with col_desc:
            st.markdown(f"""
            <div class="info-box">
                El dataset <b>Galaxy10 DECals</b> es una versión mejorada y más rigurosa del Galaxy10 original. 
                Las imágenes provienen del <i>DESI Legacy Imaging Surveys</i> y han sido categorizadas científicamente.
                <br><br>
                <ul>
                    <li><b>Cantidad:</b> ~17,736 imágenes en color (g, r, z bands).</li>
                    <li><b>Resolución:</b> 256x256 píxeles.</li>
                    <li><b>Clases:</b> 10 tipos morfológicos distintos (espirales, elípticas, irregulares, etc.).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col_sample:
            # Aquí está la imagen restaurada que solicitaste
            st.image("https://astronn.readthedocs.io/en/latest/_images/galaxy10_example.png", caption="Muestras por clase (Galaxy10)", use_container_width=True)

        st.markdown("---")

        # 2. Comparación de Modelos (DATOS REALES)
        st.markdown("## 2. Resultados Experimentales Comparativos")
        st.markdown("Comparación directa de métricas obtenidas en el conjunto de prueba (Test Set).")

        col_metric_a, col_metric_b = st.columns(2)

        with col_metric_a:
            st.markdown("### ❌ EfficientNetV2 (Fine-Tuning)")
            st.error("Rendimiento Base")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy Global", "51.33%")
            c2.metric("Weighted F1", "0.49")
            st.caption("Problemas significativos en clases minoritarias.")

        with col_metric_b:
            st.markdown(f"### 🏆 SimpleCNN (Arquitectura Propia)")
            st.success("Modelo Seleccionado")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy Global", "68.16%", delta="+16.83%")
            c2.metric("Weighted F1", "0.67", delta="+0.18")
            st.caption("Mejor generalización y consistencia.")

        # Tabla comparativa detallada POR CLASE
        st.markdown("### 📋 Desglose de F1-Score por Clase")
        with st.expander("Ver Tabla Comparativa Detallada", expanded=True):
            data_comparison = {
                "Clase": [
                    "0. Disturbed", "1. Merging", "2. Round Smooth", 
                    "3. In-between Round", "4. Cigar Shaped", "5. Barred Spiral", 
                    "6. Unbarred Tight", "7. Unbarred Loose", "8. Edge-on (No Bulge)", "9. Edge-on (Bulge)"
                ],
                "EfficientNet (F1)": [0.24, 0.40, 0.64, 0.50, 0.59, 0.45, 0.45, 0.37, 0.75, 0.66],
                "SimpleCNN (F1)":    [0.43, 0.65, 0.85, 0.79, 0.81, 0.58, 0.67, 0.47, 0.78, 0.78]
            }
            
            df_comp = pd.DataFrame(data_comparison)
            df_comp = df_comp.set_index("Clase")
            
            st.dataframe(
                df_comp.style.highlight_max(subset=["EfficientNet (F1)", "SimpleCNN (F1)"], axis=1, color=UTEM_LIGHT_BLUE)
                .format("{:.2f}"),
                use_container_width=True
            )

    # ----------------------------------------------------------------------
    # TAB 3: CLASIFICADOR (La herramienta)
    # ----------------------------------------------------------------------
    with tab_demo:
        if model is None:
            st.warning("⚠️ El modelo no está cargado. Verifique la ruta del archivo .pth en el código.")
        else:
            c_upload, c_result = st.columns([1, 1.5], gap="large")
            
            # --- COLUMNA IZQUIERDA: INPUT ---
            with c_upload:
                st.markdown("### 📡 Imagen de Entrada")
                
                uploaded_file = st.file_uploader(
                    "Cargar imagen de galaxia",
                    type=['jpg', 'png', 'jpeg'],
                    label_visibility="visible"
                )
                
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Imagen cargada", use_container_width=True)
                    
                    analyze_btn = st.button("🔍 ANALIZAR GALAXIA")
                else:
                    st.markdown(
                        f"""
                        <div style="border: 2px dashed #ccc; border-radius: 10px; height: 300px; display: flex; align-items: center; justify-content: center; color: #aaa;">
                            Esperando imagen...
                        </div>
                        """, unsafe_allow_html=True
                    )
                    analyze_btn = False

            # --- COLUMNA DERECHA: RESULTADOS ---
            with c_result:
                st.markdown("### 🧪 Resultados del Análisis")
                
                if uploaded_file and analyze_btn:
                    with st.spinner("Procesando tensores y extrayendo características..."):
                        # Inferencia
                        input_tensor = process_image(image).to(device)
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                        
                        # Procesar datos
                        probs_np = probs.cpu().numpy()[0]
                        top_prob, top_class_idx = torch.max(probs, 1)
                        top_class_name = class_names[top_class_idx.item()]
                        confidence = top_prob.item()
                        
                        # --- TARJETA DE RESULTADO PRINCIPAL ---
                        st.markdown(f"""
                        <div class="success-box">
                            <h4 style="margin:0; color: #555;">Clasificación Predicha</h4>
                            <h1 style="margin: 10px 0; color: {UTEM_BLUE}; font-size: 2.5em;">{top_class_name}</h1>
                            <div style="background-color: white; border-radius: 15px; width: 100%; height: 20px; margin-top: 10px; border: 1px solid #ddd; overflow: hidden;">
                                <div style="background-color: {UTEM_LIGHT_BLUE}; width: {confidence*100}%; height: 100%;"></div>
                            </div>
                            <p style="margin-top: 5px; font-weight: bold;">Confianza: {confidence*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # --- GRÁFICO PLOTLY ---
                        df_probs = pd.DataFrame({'Clase': class_names, 'Probabilidad': probs_np})
                        df_probs = df_probs.sort_values(by='Probabilidad', ascending=True)
                        
                        fig = plot_probabilities(df_probs)
                        st.plotly_chart(fig, use_container_width=True)
                        
                elif not uploaded_file:
                    st.info("👈 Sube una imagen en el panel izquierdo para comenzar el análisis.")
    
    # ----------------------------------------------------------------------
    # TAB 4: PRESENTACIÓN INTEGRADA
    # ----------------------------------------------------------------------
    with tab_presentation:
        st.markdown("## 🎯 Presentación del Proyecto")
        
        # Información sobre la presentación
        PRESENTATION_PATH = os.path.join(BASE_DIR, "Deep Galaxias.pptx")
        
        col_info, col_download = st.columns([2, 1])
        
        with col_info:
            st.markdown("""
            Esta sección contiene la **presentación académica completa** del proyecto Deep Galaxias,
            desarrollada siguiendo la estructura requerida para la evaluación final.
            
            **📊 Contenido:**
            - 10 slides principales según pauta académica
            - Resultados finales: SimpleCNN 68.16% vs EfficientNetV2 51.33%
            - Análisis comparativo, interpretabilidad y conclusiones
            - Declaración completa de herramientas IA utilizadas
            """)
        
        with col_download:
            if os.path.exists(PRESENTATION_PATH):
                with open(PRESENTATION_PATH, "rb") as pptx_file:
                    st.download_button(
                        label="📥 Descargar PowerPoint",
                        data=pptx_file.read(),
                        file_name="Deep_Galaxias_Final.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        help="Presentación completa en formato PPTX"
                    )
                st.success("✅ PowerPoint disponible")
            else:
                st.warning("⚠️ Archivo 'Deep Galaxias.pptx' no encontrado")
        
        st.markdown("---")
        
        # Mostrar presentación integrada
        show_presentation()

if __name__ == "__main__":
    main()