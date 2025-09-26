# ====================== FITS Converter & Visualizer ======================
# Run locally:
#    streamlit run fits_converter_visualizer.py
# Dependencies:
#    streamlit, astropy, fitsio, numpy, pandas, pillow, matplotlib, plotly,
#    scipy, photutils, requests, openpyxl, h5py

import io, os, requests
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
from scipy.ndimage import gaussian_filter

# Optional: Streamlit & FastAPI
try:
    import streamlit as st
except:
    st = None

try:
    from photutils import CircularAperture, aperture_photometry
except:
    CircularAperture = None

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
    buf = buf[:,:, [1,2,3,0]]  # ARGB -> RGBA
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
    st.set_page_config(page_title="FITS Converter & Visualizer [RAYsi]", layout="wide")
    
    # Header
    col1, col2 = st.columns([1,6])
    with col1:
        try:
            logo = Image.open("Asset 4.png")
            st.image(logo, width=100)
        except:
            st.write("Logo not found")
    with col2:
        st.title("FITS Converter & Visualizer [RAYsi]")
        st.markdown("**KARL, BU**")

    # ------------------- SIDEBAR -------------------
st.sidebar.title("Controls")

uploaded = st.sidebar.file_uploader("Upload FITS files", type=['fits','fit','fts'], accept_multiple_files=True)

fits_url = st.sidebar.text_input("FITS file URL")
if st.sidebar.button("Download FITS") and fits_url:
    try:
        r = requests.get(fits_url); r.raise_for_status()
        filename = fits_url.split("/")[-1]
        st.session_state.setdefault('loaded', {})[filename] = load_fits_data(io.BytesIO(r.content))
        st.success(f"Downloaded and loaded {filename}")
    except Exception as e:
        st.error(f"Download failed: {e}")

# ------------------- Visualization & Processing Options -------------------
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

# ------------------- AUTHORS / CONTRIBUTORS -------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Authors & Contributors:**")
st.sidebar.markdown("""
Rayhan Miah (App Developer)  
Israt Jahan Powsi (App Developer)  
Al Amin (QC Test)  
Pranto Das (QC Test)  
Abdul Hafiz Tamim (Image processing Dev)  
Shahariar Emon (Domain Expert)  
Dr. Md. Khorshed Alam (Supervisor)
""")


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

    # ------------------- HISTOGRAM -------------------
    with tabs[3]:
        st.subheader("Histogram & Interactive Stretch")
        if st.session_state['loaded']:
            file = st.selectbox("Select file", list(st.session_state['loaded'].keys()), key='hist')
            hdul = st.session_state['loaded'].get(file)
            if hdul:
                img_hdus = [i for i,h in enumerate(hdul) if getattr(h,'data',None) is not None and h.data.ndim>=2]
                if img_hdus:
                    hdu_idx = st.selectbox("Image HDU", img_hdus, key='hist_hdu')
                    arr = np.array(hdul[hdu_idx].data)
                    if arr.ndim>2: arr=arr[0]
                    fig, ax = plt.subplots()
                    ax.hist(arr.flatten(), bins=256, color='gray')
                    ax.set_title(f"Histogram: {file} [HDU {hdu_idx}]")
                    ax.set_xlabel("Pixel Value")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)

    # ------------------- RGB COMPOSITE -------------------
    with tabs[4]:
        st.subheader("RGB Composite from 3 FITS files")
        files = st.multiselect("Select 3 FITS files", list(st.session_state['loaded'].keys()))
        if len(files)==3:
            hdul_r, hdul_g, hdul_b = [st.session_state['loaded'][f] for f in files]
            arrs = []
            for hdul in [hdul_r, hdul_g, hdul_b]:
                img_hdus = [i for i,h in enumerate(hdul) if getattr(h,'data',None) is not None and h.data.ndim>=2]
                arr = np.array(hdul[img_hdus[0]].data)
                if arr.ndim > 2: arr = arr[0]
                arrs.append(arr)
            
            # Normalize each channel
            norm_arrs = []
            for a in arrs:
                n = (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))
                norm_arrs.append(n)
            
            rgb = np.dstack(norm_arrs)
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(rgb, origin='lower')
            ax.set_title("RGB Composite")
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Select exactly 3 FITS files for RGB composite.")

    # ------------------- APERTURE PHOTOMETRY -------------------
    with tabs[5]:
        st.subheader("Aperture Photometry")
        if CircularAperture is None:
            st.warning("photutils not installed. Aperture photometry disabled.")
        elif st.session_state['loaded']:
            file = st.selectbox("Select file for photometry", list(st.session_state['loaded'].keys()), key='phot')
            hdul = st.session_state['loaded'][file]
            img_hdus = [i for i,h in enumerate(hdul) if getattr(h,'data',None) is not None and h.data.ndim>=2]
            if img_hdus:
                hdu_idx = img_hdus[0]
                arr = np.array(hdul[hdu_idx].data)
                if arr.ndim>2: arr=arr[0]
                x = st.number_input("X position", 0, arr.shape[1]-1, int(arr.shape[1]/2))
                y = st.number_input("Y position", 0, arr.shape[0]-1, int(arr.shape[0]/2))
                r = st.number_input("Aperture radius (pixels)", 1, min(arr.shape)//2, 5)
                aperture = CircularAperture((x,y), r)
                phot_table = aperture_photometry(arr, aperture)
                st.write(phot_table)

                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(arr, origin='lower', cmap='gray')
                aperture.plot(color='red', lw=1.5, ax=ax)
                ax.set_title("Aperture Photometry")
                st.pyplot(fig)

    # ------------------- EXPORT -------------------
    with tabs[6]:
        st.subheader("Export Images")
        if st.session_state['loaded']:
            file = st.selectbox("Select file to export", list(st.session_state['loaded'].keys()), key='export')
            hdul = st.session_state['loaded'][file]
            img_hdus = [i for i,h in enumerate(hdul) if getattr(h,'data',None) is not None and h.data.ndim>=2]
            if img_hdus:
                arr = np.array(hdul[img_hdus[0]].data)
                if arr.ndim>2: arr=arr[0]
                norm = normalize_image(arr, stretch=default_stretch)
                img = array_to_pil(arr, cmap=default_cmap, norm=norm)
                buf = save_image_pil(img, fmt=out_format, dpi=dpi)
                st.download_button(f"Download {out_format}", data=buf, file_name=f"{file}.{out_format.lower()}")



# ====================== MAIN ======================
if __name__ == "__main__" and st is not None:
    build_streamlit_ui()
