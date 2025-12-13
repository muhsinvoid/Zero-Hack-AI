import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    basedir = getattr(sys, '_MEIPASS', os.getcwd())
    return os.path.join(basedir, path)

if __name__ == "__main__":
    # Ana dosyamızın adını buraya yazıyoruz
    app_path = resolve_path("emlak_pro_v3.py")
    
    # Streamlit'i başlatma komutunu simüle ediyoruz
    sys.argv = ["streamlit", "run", app_path, "--global.developmentMode=false"]
    
    sys.exit(stcli.main())main())
