"""
Deep Galaxias - Deployment con ngrok
Ejecutar aplicación Streamlit con túnel público ngrok

Uso:
1. Instalar ngrok: https://ngrok.com/download
2. Autenticar: ngrok authtoken YOUR_TOKEN
3. Ejecutar: python run_with_ngrok.py
"""

import subprocess
import time
import threading
from pyngrok import ngrok
import sys
import os

# Configuración
STREAMLIT_PORT = 8501
NGROK_AUTH_TOKEN = None  # Opcional: colocar tu token aquí

def start_streamlit():
    """Iniciar aplicación Streamlit en background"""
    try:
        # Activar entorno virtual y ejecutar Streamlit
        if os.path.exists('.venv'):
            if sys.platform == "win32":
                python_path = os.path.join('.venv', 'Scripts', 'python.exe')
                streamlit_path = os.path.join('.venv', 'Scripts', 'streamlit.exe')
            else:
                python_path = os.path.join('.venv', 'bin', 'python')
                streamlit_path = os.path.join('.venv', 'bin', 'streamlit')
            
            cmd = [streamlit_path, "run", "app.py", "--server.port", str(STREAMLIT_PORT)]
        else:
            cmd = ["streamlit", "run", "app.py", "--server.port", str(STREAMLIT_PORT)]
        
        print(f"🚀 Iniciando Streamlit en puerto {STREAMLIT_PORT}...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process
    except Exception as e:
        print(f"❌ Error iniciando Streamlit: {e}")
        return None

def setup_ngrok():
    """Configurar túnel ngrok"""
    try:
        # Autenticar si hay token
        if NGROK_AUTH_TOKEN:
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        
        # Crear túnel
        print(f"🌐 Creando túnel ngrok para puerto {STREAMLIT_PORT}...")
        public_url = ngrok.connect(STREAMLIT_PORT)
        
        print("\n" + "="*60)
        print("🎉 DEEP GALAXIAS DESPLEGADO EXITOSAMENTE")
        print("="*60)
        print(f"📱 URL Pública: {public_url}")
        print(f"🏠 URL Local: http://localhost:{STREAMLIT_PORT}")
        print("="*60)
        print("\n💡 Compartir esta URL para acceso remoto")
        print("⚠️  Mantener esta ventana abierta para el túnel")
        print("🛑 Ctrl+C para detener\n")
        
        return public_url
    except Exception as e:
        print(f"❌ Error configurando ngrok: {e}")
        print("💡 Verificar:")
        print("   1. ngrok instalado: https://ngrok.com/download")
        print("   2. Token autenticado: ngrok authtoken YOUR_TOKEN")
        return None

def main():
    """Función principal de despliegue"""
    print("🌌 Deep Galaxias - Despliegue con ngrok")
    print("=" * 40)
    
    # Verificar directorio
    if not os.path.exists("app.py"):
        print("❌ Error: app.py no encontrado")
        print("💡 Ejecutar desde directorio deep_galaxias/")
        return
    
    # Iniciar Streamlit
    streamlit_process = start_streamlit()
    if not streamlit_process:
        return
    
    # Esperar que Streamlit inicie
    print("⏳ Esperando que Streamlit inicie...")
    time.sleep(5)
    
    # Configurar ngrok
    try:
        public_url = setup_ngrok()
        if not public_url:
            streamlit_process.terminate()
            return
        
        # Mantener proceso activo
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Deteniendo servicios...")
            ngrok.kill()
            streamlit_process.terminate()
            print("✅ Servicios detenidos correctamente")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        streamlit_process.terminate()

if __name__ == "__main__":
    main()