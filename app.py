# ====================== FITS Converter & Visualizer ======================
# Run locally:
#    streamlit run fits_converter_visualizer.py
# Dependencies:
#    streamlit, astropy, fitsio, numpy, pandas, pillow, matplotlib, plotly,
#    scipy, photutils, requests, openpyxl, h5py

import io, os, sys, argparse
from typing import List, Optional
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import (PercentileInterval, AsinhStretch,
                                   LogStretch, SqrtStretch, LinearStretch,
                                   ImageNormalize)
import fitsio
import matplotlib.pyplot as plt
from PIL import Image

# Optional: Streamlit & FastAPI
try:
    import streamlit as st
except:
    st = None

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import StreamingResponse
    import uvicorn
except:
    FastAPI = None
    StreamingResponse = None

try:
    from photutils import CircularAperture, aperture_photometry
except:
    CircularAperture = None

import requests
from scipy.ndimage import gaussian_filter

# ------------------- Utility Functions -------------------

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
                    data = h.read() if h.get_exttype()=='IMAGE' else None
                    hdu = fits.PrimaryHDU(data=data, header=hdr)
                    hdul.append(hdu)
        except Exception as e:
            raise RuntimeError(f"Failed to open FITS: {e}")
    return hdul

def header_to_dataframe(header: fits.Header) -> pd.DataFrame:
    rows = []
    for card in header.cards:
        key = card.keyword or "(COMMENT/HISTORY)"
        rows.append({"keyword": key, "value": str(card.value), "comment": card.comment or ""})
    return pd.DataFrame(rows)

def validate_fits_keywords(header: fits.Header, required_keywords: List[str]=None) -> List[str]:
    if required_keywords is None:
        required_keywords = ["SIMPLE","BITPIX","NAXIS","TELESCOP","DATE-OBS"]
    return [k for k in required_keywords if k not in header]

def normalize_image(data: np.ndarray, stretch:str="linear", percent:float=99.5):
    arr = np.array(data, dtype=np.float32)
    vmin,vmax = PercentileInterval(percent).get_limits(arr)
    stretch_map = {"linear":LinearStretch(),"log":LogStretch(),"sqrt":SqrtStretch(),"asinh":AsinhStretch()}
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch_map.get(stretch, LinearStretch()))

def array_to_pil(arr: np.ndarray, cmap:str="gray", norm=None) -> Image.Image:
    fig = plt.figure(figsize=(4,4), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.imshow(arr, cmap=cmap, norm=norm, origin='lower')
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h,w,4))
    buf = buf[:,:, [1,2,3,0]]
    plt.close(fig)
    return Image.fromarray(buf)

def save_image_pil(img: Image.Image, fmt:str='PNG', dpi:int=300, out_path:Optional[str]=None):
    if out_path:
        img.save(out_path, format=fmt.upper(), dpi=(dpi,dpi))
        return out_path
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper(), dpi=(dpi,dpi))
    buf.seek(0)
    return buf

# ------------------- Streamlit UI -------------------

def build_streamlit_ui():
    st.set_page_config(page_title="FITS Converter & Visualizer", layout="wide")
    
    # Header
    col1, col2 = st.columns([1,6])
    with col1:
        try:
            logo = Image.open("Asset 4.png")
            st.image(logo, width=100)
        except:
            st.write("Logo not found")
    with col2:
        st.title("FITS Converter & Visualizer 🔭")
        st.markdown("**KARL, BU**")

    # Sidebar: Controls
    st.sidebar.title("Controls")
    
    # Upload
    uploaded = st.sidebar.file_uploader("Upload FITS files", type=['fits','fit','fts'], accept_multiple_files=True)

    # URL download
    st.sidebar.markdown("### Download FITS from URL")
    fits_url = st.sidebar.text_input("FITS file URL")
    if st.sidebar.button("Download FITS"):
        if fits_url:
            try:
                response = requests.get(fits_url)
                response.raise_for_status()
                filename = fits_url.split("/")[-1]
                st.session_state.setdefault('loaded', {})[filename] = load_fits_data(io.BytesIO(response.content))
                st.success(f"Downloaded and loaded {filename}")
            except Exception as e:
                st.error(f"Failed to download: {e}")

    # Visualization & processing options
    general_tab, processing_tab, export_tab = st.sidebar.tabs(["General","Processing","Export"])
    
    with general_tab:
        show_wcs = st.checkbox("Show WCS Grid / RA/Dec", True)
        default_cmap = st.selectbox("Colormap", ["gray","viridis","inferno","magma","plasma","cividis"], index=0)
        default_stretch = st.selectbox("Stretch", ["linear","log","sqrt","asinh"], index=0)
    
    with processing_tab:
        enable_bgsub = st.checkbox("Background subtraction")
        enable_denoise = st.checkbox("Noise reduction (Gaussian)")
    
    with export_tab:
        out_format = st.selectbox("Export format", ["PNG","TIFF","JPEG"], index=0)
        dpi = st.number_input("DPI", 72, 1200, 300)

    tabs = st.tabs(["Upload","Metadata","Visualization","Histogram","RGB Composite","Aperture Photometry","Export"])
    st.session_state.setdefault('loaded', {})

    # ------------------- UPLOAD TAB -------------------
    with tabs[0]:
        st.subheader("Uploaded FITS Files")
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
            st.info("Upload FITS files via drag-and-drop or URL.")

    # ------------------- METADATA TAB -------------------
    with tabs[1]:
        st.subheader("FITS Header & Metadata")
        if st.session_state['loaded']:
            file = st.selectbox("Select file", list(st.session_state['loaded'].keys()))
            hdul = st.session_state['loaded'][file]
            hdu_idx = st.selectbox("Select HDU", list(range(len(hdul))))
            hdr = hdul[hdu_idx].header
            st.dataframe(header_to_dataframe(hdr), use_container_width=True)
            missing = validate_fits_keywords(hdr)
            if missing: st.warning(f"Missing keywords: {missing}")
            else: st.success("All required keywords present.")

    # ------------------- VISUALIZATION TAB -------------------
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
                if enable_denoise: arr = gaussian_filter(arr, sigma=1)
                arr = np.rot90(arr, k=rotate//90)
                norm = normalize_image(arr, stretch=stretch, percent=percentile)
                
                fig, ax = plt.subplots(figsize=(6,6))
                if show_wcs:
                    try:
                        wcs = WCS(hdul[hdu_idx].header)
                        ax = plt.subplot(projection=wcs)
                        ax.coords.grid(True, color='white', ls='dotted')
                        ax.set_xlabel('RA')
                        ax.set_ylabel('Dec')
                    except:
                        pass
                ax.imshow(arr, origin='lower', cmap=cmap, norm=norm)
                ax.set_title(f"{file} [HDU {hdu_idx}]")
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.warning("No image HDUs available.")

    # ------------------- HISTOGRAM TAB -------------------
    with tabs[3]:
        st.subheader("Histogram & Interactive Stretch")
        file = st.selectbox("Select file", list(st.session_state['loaded'].keys()), key='hist')
        hdul = st.session_state['loaded'][file]
        img_hdus = [i for i,h in enumerate(hdul) if getattr(h,'data',None) is not None and h.data.ndim>=2]
        if img_hdus:
            hdu_idx = st.selectbox("Image HDU", img_hdus, key='hist_hdu')
            arr = np.array(hdul[hdu_idx].data)
            if arr.ndim>2: arr=arr[0]
            fig, ax = plt.subplots()
            ax.hist(arr.flatten(), bins=256, color='gray')
            ax.set_title(f"Histogram: {file} [HDU {hdu_idx}]")
            st.pyplot(fig)

    # ------------------- RGB COMPOSITE TAB -------------------
    with tabs[4]:
        st.subheader("RGB Composite from 3 FITS files")
        files = st.multiselect("Select 3 FITS files", list(st.session_state['loaded'].keys()))
        if len(files)==3:
            hdul_r, hdul_g, hdul_b = [st.session_state['loaded'][f] for f in files]
            arrs = []
            for hdul in [hdul_r, hdul_g, hdul_b]:
                img_hdus = [i for i,h in enumerate(hdul) if getattr(h,'data',None) is not None and h.data.ndim>=2]
                arr = np.array(hdul[img_hdus[0]].data)
                if arr.ndim>2: arr=arr[0]
                arrs.append((arr-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr)))
            rgb = np.dstack(arrs)
            st.image((rgb*255).astype(np.uint8), caption="RGB Composite", use_column_width=True)
        else:
            st.info("Select exactly 3 FITS files.")

    # ------------------- APERTURE PHOTOMETRY TAB -------------------
    with tabs[5]:
        st.subheader("Aperture Photometry")
        if CircularAperture is None:
            st.warning("Install photutils to use aperture photometry")
        else:
            file = st.selectbox("Select file", list(st.session_state['loaded'].keys()), key='phot')
            hdul = st.session_state['loaded'][file]
            img_hdus = [i for i,h in enumerate(hdul) if getattr(h,'data',None) is not None and h.data.ndim>=2]
            if img_hdus:
                hdu_idx = st.selectbox("HDU", img_hdus, key='phot_hdu')
                arr = np.array(hdul[hdu_idx].data)
                if arr.ndim>2: arr=arr[0]
                x = st.number_input("Aperture X", 0, arr.shape[1]-1, arr.shape[1]//2)
                y = st.number_input("Aperture Y", 0, arr.shape[0]-1, arr.shape[0]//2)
                r = st.number_input("Radius", 1, min(arr.shape)//2, 5)
                aperture = CircularAperture((x,y), r=r)
                phot_table = aperture_photometry(arr, aperture)
                st.write(phot_table)

    # ------------------- EXPORT TAB -------------------
    with tabs[6]:
        st.subheader("Export FITS HDU as Image")
        file = st.selectbox("Select file", list(st.session_state['loaded'].keys()), key='export')
        hdul = st.session_state['loaded'][file]
        img_hdus = [i for i,h in enumerate(hdul) if getattr(h,'data',None) is not None and h.data.ndim>=2]
        if img_hdus:
            hdu_idx = st.selectbox("HDU", img_hdus, key='export_hdu')
            arr = np.array(hdul[hdu_idx].data)
            if arr.ndim>2: arr=arr[0]
            img = array_to_pil(arr, cmap=default_cmap, norm=normalize_image(arr, stretch=default_stretch))
            buf = save_image_pil(img, fmt=out_format, dpi=dpi)
            st.download_button("Download Image", buf, file_name=f"{file}_HDU{hdu_idx}.{out_format.lower()}")

# ------------------- Main Entry -------------------

def main():
    parser = argparse.ArgumentParser(description="FITS Converter & Visualizer")
    parser.add_argument("--cli-convert", action="store_true", help="CLI batch convert FITS to images")
    parser.add_argument("--input", nargs="+", help="Input FITS files")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--format", default="png", help="Output format")
    parser.add_argument("--dpi", type=int, default=300, help="DPI")
    parser.add_argument("--api", action="store_true", help="Start REST API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # CLI conversion
    if args.cli_convert:
        if not args.input: raise RuntimeError("No input files for CLI")
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

    # REST API
    if args.api:
        if FastAPI is None: raise RuntimeError("FastAPI not installed")
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

    if st:
        build_streamlit_ui()
    else:
        print("Streamlit not installed. Use --cli-convert or --api mode.")

if __name__ == "__main__":
    main()
