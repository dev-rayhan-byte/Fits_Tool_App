"""
Streamlit FITS Tool - app.py
A single-file Streamlit application implementing FITS upload, visualization,
metadata tools, conversion, basic processing, CLI batch conversion, and
an optional FastAPI REST endpoint for programmatic access.

Run locally for interactive UI:
    streamlit run streamlit_fits_tool_app.py

CLI examples (headless batch conversion):
    python streamlit_fits_tool_app.py --cli-convert --input files/*.fits --outdir out --format png --dpi 300

Run REST API server (separate from Streamlit):
    python streamlit_fits_tool_app.py --api --host 0.0.0.0 --port 8000

Dependencies (pip):
    streamlit astropy fitsio numpy pandas pillow matplotlib plotly
    scipy opencv-python openpyxl h5py fastapi uvicorn
"""

import io
import os
import sys
import glob
import argparse
from typing import List, Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import (PercentileInterval, AsinhStretch,
                                   LogStretch, SqrtStretch, LinearStretch,
                                   ImageNormalize)
import fitsio
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit and optional FastAPI
try:
    import streamlit as st
except Exception:
    st = None

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import StreamingResponse
    import uvicorn
except Exception:
    FastAPI = None
    StreamingResponse = None
# ------------------------- Utility functions -------------------------
def load_fits_data(path_or_buffer):
    """Load FITS file and return HDUList (astropy)."""
    try:
        hdul = fits.open(path_or_buffer, memmap=False)
    except Exception:
        # fallback to fitsio
        try:
            with fitsio.FITS(path_or_buffer) as f:
                hdul = fits.HDUList()
                for i in range(len(f)):
                    h = f[i]
                    hdr = fits.Header()
                    hdr.update(h.read_header())
                    data = h.read() if h.get_exttype() == 'IMAGE' else None
                    hdu = fits.PrimaryHDU(data=data, header=hdr)
                    hdul.append(hdu)
        except Exception as e:
            raise RuntimeError(f"Failed to open FITS: {e}")
    return hdul

def header_to_dataframe(header: fits.Header) -> pd.DataFrame:
    """Convert an astropy Header into a pandas DataFrame."""
    rows = []
    try:
        for card in header.cards:
            key = card.keyword or "(COMMENT/HISTORY)"
            rows.append({
                "keyword": key,
                "value": str(card.value),
                "comment": card.comment if card.comment else "",
            })
    except Exception:
        for k, v in header.items():
            try:
                comment = header.comments.get(k, "")
            except Exception:
                comment = ""
            rows.append({"keyword": str(k), "value": str(v), "comment": comment})
    return pd.DataFrame(rows)

def validate_fits_keywords(header: fits.Header, required_keywords: List[str] = None) -> List[str]:
    if required_keywords is None:
        required_keywords = ["SIMPLE", "BITPIX", "NAXIS", "TELESCOP", "DATE-OBS"]
    missing = [k for k in required_keywords if k not in header]
    return missing

def normalize_image(data: np.ndarray, stretch: str = "linear", percent: float = 99.5):
    arr = np.array(data, dtype=np.float32)
    interval = PercentileInterval(percent)
    vmin, vmax = interval.get_limits(arr)
    stretch_map = {
        "linear": LinearStretch(),
        "log": LogStretch(),
        "sqrt": SqrtStretch(),
        "asinh": AsinhStretch()
    }
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch_map.get(stretch, LinearStretch()))
    return norm

def array_to_pil(arr: np.ndarray, cmap: str = "gray", norm=None) -> Image.Image:
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.imshow(arr, cmap=cmap, norm=norm, origin='lower')
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
    buf = buf[:, :, [1,2,3,0]]  # ARGB -> RGBA
    plt.close(fig)
    return Image.fromarray(buf)

def save_image_pil(img: Image.Image, fmt: str = 'PNG', dpi: int = 300, out_path: Optional[str] = None):
    if out_path:
        img.save(out_path, format=fmt.upper(), dpi=(dpi, dpi))
        return out_path
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper(), dpi=(dpi, dpi))
    buf.seek(0)
    return buf

# ------------------------- Streamlit UI -------------------------

def build_streamlit_ui():
    import streamlit as st
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    # --- HEADER / LOGO ---
    st.set_page_config(page_title="Streamlit FITS Tool", layout="wide")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        try:
            logo = Image.open("Asset 4.png")  # local logo
            st.image(logo, width=100)
        except Exception:
            st.write("Logo not found")
    with col2:
        st.title("Streamlit FITS Tool [RAYsi]")
        st.markdown("**KARL, BU**")

    # ---------------- Sidebar Controls ----------------
    st.sidebar.title("Controls")
    uploaded = st.sidebar.file_uploader(
        "Upload FITS files",
        type=['fits','fit','fts'],
        accept_multiple_files=True
    )
    
    general_tab, processing_tab, export_tab = st.sidebar.tabs(["General", "Processing", "Export"])
    
    with general_tab:
        show_wcs = st.checkbox("Show WCS Grid / RA/Dec", True)
        default_cmap = st.selectbox(
            "Default colormap",
            ["gray","viridis","inferno","magma","plasma","cividis"],
            index=0
        )
        default_stretch = st.selectbox(
            "Default stretch",
            ["linear","log","sqrt","asinh"],
            index=0
        )
    
    with processing_tab:
        enable_bgsub = st.checkbox("Background subtraction (median)")
        enable_denoise = st.checkbox("Noise reduction (gaussian)")
    
    with export_tab:
        out_format = st.selectbox(
            "Export image format",
            ["PNG","TIFF","JPEG"],
            index=0
        )
        dpi = st.number_input("DPI for export", 72, 1200, 300)
    
    tabs = st.tabs(["Upload","Metadata","Visualization","Processing","Export"])
    
    if 'loaded' not in st.session_state:
        st.session_state['loaded'] = {}

    # ---------------- FOOTER / AUTHOR LIST ----------------
    st.markdown("---")
    st.markdown("**Authors & Contributors:**")
    st.markdown("""
    1. Rayhan Miah (App Developer)  
    2. Israt Jahan Powsi (App Developer)  
    3. Al Amin (QC Test)  
    4. Pranto Das (QC Test)  
    5. Abdul Hafiz Tamim (Image processing Dev)  
    6. Shahariar Emon (Domain Expert)  
    7. Dr. Md. Khorshed Alam (Supervisor)
    """)




    # UPLOAD
    with tabs[0]:
        st.subheader("Uploaded Files")
        if uploaded:
            for f in uploaded:
                st.write(f"**{f.name}** — {f.type or 'file'} — {f.size} bytes")
                if f.name not in st.session_state['loaded']:
                    try:
                        st.session_state['loaded'][f.name] = load_fits_data(f)
                        st.success(f"Loaded {f.name}")
                    except Exception as e:
                        st.error(f"Failed {f.name}: {e}")
        else:
            st.info("Upload FITS files in sidebar.")

    # METADATA
    with tabs[1]:
        st.subheader("FITS Header & Metadata")
        if st.session_state['loaded']:
            file = st.selectbox("Select file", options=list(st.session_state['loaded'].keys()))
            hdul = st.session_state['loaded'][file]
            hdu_idx = st.selectbox("Select HDU", list(range(len(hdul))))
            hdr = hdul[hdu_idx].header
            st.dataframe(header_to_dataframe(hdr), use_container_width=True)
            missing = validate_fits_keywords(hdr)
            if missing: st.warning(f"Missing keywords: {missing}")
            else: st.success("All required keywords present.")

    # VISUALIZATION
    with tabs[2]:
        st.subheader("Image Viewer")
        if st.session_state['loaded']:
            file = st.selectbox("Select file (viewer)", list(st.session_state['loaded'].keys()), key='viz')
            hdul = st.session_state['loaded'][file]
            img_hdus = [i for i,h in enumerate(hdul) if getattr(h,'data',None) is not None and h.data.ndim>=2]
            if img_hdus:
                hdu_idx = st.selectbox("Image HDU", img_hdus)
                arr = np.array(hdul[hdu_idx].data)
                if arr.ndim>2: arr=arr[0]
                stretch = st.selectbox("Stretch", ["linear","log","sqrt","asinh"], index=0)
                percentile = st.slider("Percentile", 90.0, 100.0, 99.5)
                cmap = st.selectbox("Colormap", ["gray","viridis","inferno","magma","plasma","cividis"], index=0)
                rotate = int(st.selectbox("Rotate", ["0","90","180","270"], index=0))
                if enable_bgsub: arr -= np.nanmedian(arr)
                if enable_denoise:
                    from scipy.ndimage import gaussian_filter
                    arr = gaussian_filter(arr, sigma=1)
                arr = np.rot90(arr, k=rotate//90)
                norm = normalize_image(arr, stretch=stretch, percent=percentile)
                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(arr, origin='lower', cmap=cmap, norm=norm)
                ax.set_title(f"{file} [HDU {hdu_idx}]")
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.warning("No image HDUs available for visualization.")

    # EXPORT
    with tabs[4]:
        st.subheader("Export FITS HDU as Image")
        if st.session_state['loaded']:
            file = st.selectbox("File to export", list(st.session_state['loaded'].keys()), key='export')
            hdul = st.session_state['loaded'][file]
            img_hdus = [i for i,h in enumerate(hdul) if getattr(h,'data',None) is not None and h.data.ndim>=2]
            if img_hdus:
                hdu_idx = st.selectbox("Select HDU", img_hdus, key='export_hdu')
                arr = np.array(hdul[hdu_idx].data)
                if arr.ndim>2: arr = arr[0]
                img = array_to_pil(arr, cmap=default_cmap, norm=normalize_image(arr, stretch=default_stretch))
                buf = save_image_pil(img, fmt=out_format, dpi=dpi)
                st.download_button("Download Image", buf, file_name=f"{file}_HDU{hdu_idx}.{out_format.lower()}")
            else:
                st.warning("No image HDUs to export.")

# ------------------------- Main entry -------------------------
def main():
    parser = argparse.ArgumentParser(description="Streamlit FITS Tool")
    parser.add_argument("--cli-convert", action="store_true", help="CLI batch convert FITS to images")
    parser.add_argument("--input", nargs="+", help="Input FITS files")
    parser.add_argument("--outdir", default=".", help="Output directory for images")
    parser.add_argument("--format", default="png", help="Image format PNG/TIFF/JPEG")
    parser.add_argument("--dpi", type=int, default=300, help="Image DPI")
    parser.add_argument("--api", action="store_true", help="Start FastAPI REST server")
    parser.add_argument("--host", default="127.0.0.1", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    args = parser.parse_args()

    if args.cli_convert:
        if not args.input: raise RuntimeError("No input files provided for CLI conversion")
        os.makedirs(args.outdir, exist_ok=True)
        for f in args.input:
            hdul = load_fits_data(f)
            for i,hdu in enumerate(hdul):
                if getattr(hdu,'data',None) is not None and hdu.data.ndim>=2:
                    arr = np.array(hdu.data)
                    if arr.ndim>2: arr=arr[0]
                    img = array_to_pil(arr)
                    out_path = os.path.join(args.outdir, f"{os.path.basename(f)}_HDU{i}.{args.format.lower()}")
                    save_image_pil(img, fmt=args.format, dpi=args.dpi, out_path=out_path)
                    print(f"Saved {out_path}")
        return

    if args.api:
        if FastAPI is None:
            raise RuntimeError("FastAPI not installed.")
        app = FastAPI()
        @app.post("/upload_fits/")
        async def upload_fits(file: UploadFile = File(...)):
            buf = io.BytesIO(await file.read())
            hdul = load_fits_data(buf)
            arrs = []
            for hdu in hdul:
                if getattr(hdu,'data',None) is not None and hdu.data.ndim>=2:
                    arr = np.array(hdu.data)
                    if arr.ndim>2: arr=arr[0]
                    img = array_to_pil(arr)
                    outbuf = save_image_pil(img, fmt='PNG')
                    arrs.append(outbuf)
            return StreamingResponse(arrs[0], media_type="image/png")
        uvicorn.run(app, host=args.host, port=args.port)
        return

    # Streamlit UI
    if st:
        build_streamlit_ui()
    else:
        print("Streamlit not installed. Use --cli-convert or --api mode.")

if __name__ == "__main__":
    main()
