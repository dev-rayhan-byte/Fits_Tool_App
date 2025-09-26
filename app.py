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

This file is intentionally feature-rich but modular — remove parts you
don't need.
"""

import io
import os
import sys
import glob
import argparse
import json
import tempfile
from typing import List, Dict, Optional

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

# Only import FastAPI when required to avoid extra deps for UI users
try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import StreamingResponse
    import uvicorn
except Exception:
    FastAPI = None
    StreamingResponse = None

# ------------------------- Utility functions -------------------------

def load_fits_data(path_or_buffer):
    """Load FITS file and return HDUList (astropy) and list of HDU info."""
    try:
        hdul = fits.open(path_or_buffer, memmap=False)
    except Exception:
        # fallback to fitsio
        try:
            with fitsio.FITS(path_or_buffer) as f:
                # convert to astropy HDUList for compatibility
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
    """Convert an astropy Header into a pandas DataFrame.

    This is robust to different astropy versions where header.comments
    is not a plain dict. It also preserves COMMENT/HISTORY fields.
    """
    rows = []
    # Use header.cards to preserve order and capture COMMENT/HISTORY entries
    try:
        for card in header.cards:
            key = card.keyword
            # astropy uses '' for blank keywords in COMMENT/HISTORY cards
            if key is None or key == '':
                # represent COMMENT/HISTORY as special rows
                rows.append({
                    "keyword": card.keyword if card.keyword is not None else "(COMMENT/HISTORY)",
                    "value": str(card.value),
                    "comment": card.comment if card.comment is not None else "",
                })
                continue
            # Normal keyword
            comment = ""
            try:
                comment = header.comments[key]
            except Exception:
                # header.comments may not behave like a dict in some astropy versions
                try:
                    comment = card.comment or ""
                except Exception:
                    comment = ""
            rows.append({"keyword": str(key), "value": str(card.value), "comment": comment})
    except Exception:
        # Fallback: older astropy where header.cards might not exist
        for k, v in header.items():
            comment = ""
            try:
                comment = header.comments[k]
            except Exception:
                try:
                    # some Header implementations allow dict-like access
                    comment = header.get_comment(k)
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
    arr = np.array(data)
    arr = arr.astype(np.float32)
    interval = PercentileInterval(percent)
    vmin, vmax = interval.get_limits(arr)
    if stretch == "linear":
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
    elif stretch == "log":
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
    elif stretch == "sqrt":
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
    elif stretch == "asinh":
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    else:
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
    return norm


def array_to_pil(arr: np.ndarray, cmap: str = "gray", norm=None) -> Image.Image:
    # Use matplotlib to map to RGBA then convert to PIL
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis('off')
    if norm is None:
        im = ax.imshow(arr, cmap=cmap)
    else:
        im = ax.imshow(arr, cmap=cmap, norm=norm)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    # ARGB -> RGBA
    buf = buf[:, :, [1, 2, 3, 0]]
    plt.close(fig)
    image = Image.fromarray(buf)
    return image


def save_image_pil(img: Image.Image, fmt: str = 'PNG', dpi: int = 300, out_path: Optional[str] = None):
    if out_path is None:
        buf = io.BytesIO()
        img.save(buf, format=fmt, dpi=(dpi, dpi))
        buf.seek(0)
        return buf
    else:
        img.save(out_path, format=fmt, dpi=(dpi, dpi))
        return out_path

# ------------------------- Streamlit UI -------------------------

def build_streamlit_ui():
    st.set_page_config(page_title="Streamlit FITS Tool", layout="wide", initial_sidebar_state="expanded")
    st.markdown("# Streamlit FITS Tool 🔭")

    # Sidebar
    st.sidebar.title("Controls")
    uploaded = st.sidebar.file_uploader("Upload one or more FITS files", type=['fits', 'fit', 'fts'], accept_multiple_files=True)
    sidebar_tabs = st.sidebar.tabs(["General", "Processing", "Export"])

    with sidebar_tabs[0]:
        show_wcs = st.checkbox("Show WCS Grid / RA/Dec", value=True)
        default_cmap = st.selectbox("Default colormap", options=["gray", "viridis", "inferno", "magma", "plasma", "cividis"], index=0)
        default_stretch = st.selectbox("Default stretch", ["linear", "log", "sqrt", "asinh"], index=0)
    with sidebar_tabs[1]:
        enable_background_sub = st.checkbox("Background subtraction (median)")
        enable_denoise = st.checkbox("Noise reduction (gaussian)")
    with sidebar_tabs[2]:
        out_format = st.selectbox("Export image format", ["PNG", "TIFF", "JPEG"], index=0)
        dpi = st.number_input("DPI for export", min_value=72, max_value=1200, value=300)

    # Main tabs
    tabs = st.tabs(["Upload", "Metadata", "Visualization", "Processing", "Export"])

    # Keep a cache of loaded HDUs
    if 'loaded' not in st.session_state:
        st.session_state['loaded'] = {}

    # UPLOAD tab
    with tabs[0]:
        st.subheader("Uploaded Files")
        if uploaded:
            for f in uploaded:
                st.write(f"**{f.name}** — {f.type or 'file'} — {f.size} bytes")
                if f.name not in st.session_state['loaded']:
                    try:
                        hdul = load_fits_data(f)
                        st.session_state['loaded'][f.name] = hdul
                        st.success(f"Loaded {f.name} with {len(hdul)} HDU(s)")
                    except Exception as e:
                        st.error(f"Failed to read {f.name}: {e}")
        else:
            st.info("Drag and drop FITS files in the sidebar uploader to begin.")

    # METADATA tab
    with tabs[1]:
        st.subheader("FITS Header & Metadata")
        selected_file = st.selectbox("Select file", options=list(st.session_state['loaded'].keys()) if st.session_state['loaded'] else [])
        if selected_file:
            hdul = st.session_state['loaded'][selected_file]
            hdu_index = st.selectbox("Select HDU", options=list(range(len(hdul))))
            hdr = hdul[hdu_index].header
            st.markdown(f"**HDU {hdu_index}: {hdul[hdu_index].name}**")
            df = header_to_dataframe(hdr)
            st.dataframe(df, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export metadata as CSV"):
                    csv_buf = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", data=csv_buf, file_name=f"{selected_file}_hdu{hdu_index}_header.csv", mime='text/csv')
            with col2:
                if st.button("Export metadata as JSON"):
                    json_buf = df.to_json(orient='records').encode('utf-8')
                    st.download_button("Download JSON", data=json_buf, file_name=f"{selected_file}_hdu{hdu_index}_header.json", mime='application/json')

            # compliance check
            missing = validate_fits_keywords(hdr)
            if missing:
                st.warning(f"Missing required keywords: {', '.join(missing)}")
            else:
                st.success("Basic FITS keywords present.")

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

This file is intentionally feature-rich but modular — remove parts you
don't need.
"""

import io
import os
import sys
import glob
import argparse
import json
import tempfile
from typing import List, Dict, Optional

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

# Only import FastAPI when required to avoid extra deps for UI users
try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import StreamingResponse
    import uvicorn
except Exception:
    FastAPI = None
    StreamingResponse = None

# ------------------------- Utility functions -------------------------

def load_fits_data(path_or_buffer):
    """Load FITS file and return HDUList (astropy) and list of HDU info."""
    try:
        hdul = fits.open(path_or_buffer, memmap=False)
    except Exception:
        # fallback to fitsio
        try:
            with fitsio.FITS(path_or_buffer) as f:
                # convert to astropy HDUList for compatibility
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
    """Convert an astropy Header into a pandas DataFrame.

    This is robust to different astropy versions where header.comments
    is not a plain dict. It also preserves COMMENT/HISTORY fields.
    """
    rows = []
    # Use header.cards to preserve order and capture COMMENT/HISTORY entries
    try:
        for card in header.cards:
            key = card.keyword
            # astropy uses '' for blank keywords in COMMENT/HISTORY cards
            if key is None or key == '':
                # represent COMMENT/HISTORY as special rows
                rows.append({
                    "keyword": card.keyword if card.keyword is not None else "(COMMENT/HISTORY)",
                    "value": str(card.value),
                    "comment": card.comment if card.comment is not None else "",
                })
                continue
            # Normal keyword
            comment = ""
            try:
                comment = header.comments[key]
            except Exception:
                # header.comments may not behave like a dict in some astropy versions
                try:
                    comment = card.comment or ""
                except Exception:
                    comment = ""
            rows.append({"keyword": str(key), "value": str(card.value), "comment": comment})
    except Exception:
        # Fallback: older astropy where header.cards might not exist
        for k, v in header.items():
            comment = ""
            try:
                comment = header.comments[k]
            except Exception:
                try:
                    # some Header implementations allow dict-like access
                    comment = header.get_comment(k)
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
    arr = np.array(data)
    arr = arr.astype(np.float32)
    interval = PercentileInterval(percent)
    vmin, vmax = interval.get_limits(arr)
    if stretch == "linear":
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
    elif stretch == "log":
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
    elif stretch == "sqrt":
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
    elif stretch == "asinh":
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    else:
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
    return norm


def array_to_pil(arr: np.ndarray, cmap: str = "gray", norm=None) -> Image.Image:
    # Use matplotlib to map to RGBA then convert to PIL
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis('off')
    if norm is None:
        im = ax.imshow(arr, cmap=cmap)
    else:
        im = ax.imshow(arr, cmap=cmap, norm=norm)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    # ARGB -> RGBA
    buf = buf[:, :, [1, 2, 3, 0]]
    plt.close(fig)
    image = Image.fromarray(buf)
    return image


def save_image_pil(img: Image.Image, fmt: str = 'PNG', dpi: int = 300, out_path: Optional[str] = None):
    if out_path is None:
        buf = io.BytesIO()
        img.save(buf, format=fmt, dpi=(dpi, dpi))
        buf.seek(0)
        return buf
    else:
        img.save(out_path, format=fmt, dpi=(dpi, dpi))
        return out_path

# ------------------------- Streamlit UI -------------------------

def build_streamlit_ui():
    st.set_page_config(page_title="Streamlit FITS Tool", layout="wide", initial_sidebar_state="expanded")
    st.markdown("# Streamlit FITS Tool 🔭")

    # Sidebar
    st.sidebar.title("Controls")
    uploaded = st.sidebar.file_uploader("Upload one or more FITS files", type=['fits', 'fit', 'fts'], accept_multiple_files=True)
    sidebar_tabs = st.sidebar.tabs(["General", "Processing", "Export"])

    with sidebar_tabs[0]:
        show_wcs = st.checkbox("Show WCS Grid / RA/Dec", value=True)
        default_cmap = st.selectbox("Default colormap", options=["gray", "viridis", "inferno", "magma", "plasma", "cividis"], index=0)
        default_stretch = st.selectbox("Default stretch", ["linear", "log", "sqrt", "asinh"], index=0)
    with sidebar_tabs[1]:
        enable_background_sub = st.checkbox("Background subtraction (median)")
        enable_denoise = st.checkbox("Noise reduction (gaussian)")
    with sidebar_tabs[2]:
        out_format = st.selectbox("Export image format", ["PNG", "TIFF", "JPEG"], index=0)
        dpi = st.number_input("DPI for export", min_value=72, max_value=1200, value=300)

    # Main tabs
    tabs = st.tabs(["Upload", "Metadata", "Visualization", "Processing", "Export"])

    # Keep a cache of loaded HDUs
    if 'loaded' not in st.session_state:
        st.session_state['loaded'] = {}

    # UPLOAD tab
    with tabs[0]:
        st.subheader("Uploaded Files")
        if uploaded:
            for f in uploaded:
                st.write(f"**{f.name}** — {f.type or 'file'} — {f.size} bytes")
                if f.name not in st.session_state['loaded']:
                    try:
                        hdul = load_fits_data(f)
                        st.session_state['loaded'][f.name] = hdul
                        st.success(f"Loaded {f.name} with {len(hdul)} HDU(s)")
                    except Exception as e:
                        st.error(f"Failed to read {f.name}: {e}")
        else:
            st.info("Drag and drop FITS files in the sidebar uploader to begin.")

    # METADATA tab
    with tabs[1]:
        st.subheader("FITS Header & Metadata")
        selected_file = st.selectbox("Select file", options=list(st.session_state['loaded'].keys()) if st.session_state['loaded'] else [])
        if selected_file:
            hdul = st.session_state['loaded'][selected_file]
            hdu_index = st.selectbox("Select HDU", options=list(range(len(hdul))))
            hdr = hdul[hdu_index].header
            st.markdown(f"**HDU {hdu_index}: {hdul[hdu_index].name}**")
            df = header_to_dataframe(hdr)
            st.dataframe(df, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export metadata as CSV"):
                    csv_buf = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", data=csv_buf, file_name=f"{selected_file}_hdu{hdu_index}_header.csv", mime='text/csv')
            with col2:
                if st.button("Export metadata as JSON"):
                    json_buf = df.to_json(orient='records').encode('utf-8')
                    st.download_button("Download JSON", data=json_buf, file_name=f"{selected_file}_hdu{hdu_index}_header.json", mime='application/json')

            # compliance check
            missing = validate_fits_keywords(hdr)
            if missing:
                st.warning(f"Missing required keywords: {', '.join(missing)}")
            else:
                st.success("Basic FITS keywords present.")

    # VISUALIZATION tab
    with tabs[2]:
        st.subheader("Image Viewer")
        selected_file_viz = st.selectbox("Select file (viewer)", options=list(st.session_state['loaded'].keys()) if st.session_state['loaded'] else [], key='viz_file')
        if selected_file_viz:
            hdul = st.session_state['loaded'][selected_file_viz]
            # find image HDUs
            image_hdus = [i for i, h in enumerate(hdul) if (hasattr(h, 'data') and h.data is not None and h.data.ndim >= 2)]
            if not image_hdus:
                st.info("No 2D image HDU found in this file.")
            else:
                hdu_idx = st.selectbox("Image HDU", options=image_hdus)
                data = hdul[hdu_idx].data
                hdr = hdul[hdu_idx].header
                # basic processing controls
                stretch = st.selectbox("Stretch", ["linear", "log", "sqrt", "asinh"], index=0)
                # Ensure we have a 2D image to display
                arr = np.array(data)
                if arr.ndim > 2:
                    st.warning(f"HDU {hdu_idx} has {arr.ndim} dimensions — displaying the first slice along axis 0.")
                    arr = arr[0]
                if arr.ndim < 2:
                    st.error(f"HDU {hdu_idx} has invalid shape {arr.shape} for image display.")
                    continue
                percent = st.slider("Stretch percentile", 90.0, 100.0, 99.5)
                cmap = st.selectbox("Colormap", ["gray", "viridis", "inferno", "magma", "plasma", "cividis"], index=0)
                rotate = st.selectbox("Rotate", ["0", "90", "180", "270"], index=0)
                do_crop = st.checkbox("Enable crop tool")
                # build norm and display (using the 2D array 'arr')
                norm = normalize_image(arr, stretch=stretch, percent=percent)
                # Apply rotation
                if rotate != "0":
                    k = int(int(rotate) / 90)
                    arr = np.rot90(arr, k=k)
                if rotate != "0":
                    k = int(int(rotate) / 90)
                    arr = np.rot90(arr, k=k)
                if enable_background_sub:
                    med = np.nanmedian(arr)
                    arr = arr - med
                if enable_denoise:
                    from scipy.ndimage import gaussian_filter
                    arr = gaussian_filter(arr, sigma=1)
                # Show using matplotlib figure into Streamlit
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(arr, cmap=cmap, origin='lower', norm=norm)
                ax.set_title(f"{selected_file_viz} [HDU {hdu_idx}]")
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
                # Download rendered image
                pil_img = array_to_pil(arr, cmap=cmap, norm=norm)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                st.download_button("Download PNG", data=buf, file_name=f"{os.path.splitext(selected_file_viz)[0]}_hdu{hdu_idx}.png", mime='image/png')

                # WCS overlay preview
                if show_wcs:
                    try:
                        w = WCS(hdr)
                        ax = plt.subplot(projection=w)
                        plt.imshow(arr, origin='lower', cmap=cmap, norm=norm)
                        ax.coords.grid(True, color='white', linestyle='--')
                        st.pyplot(plt.gcf())
                        plt.close()
                    except Exception as e:
                        st.info(f"WCS not available or failed: {e}")

    # PROCESSING tab
    with tabs[3]:
        st.subheader("Processing Tools")
        st.write("Histogram and stretch editor")
        file_proc = st.selectbox("Select file (process)", options=list(st.session_state['loaded'].keys()) if st.session_state['loaded'] else [], key='proc_file')
        if file_proc:
            hdul = st.session_state['loaded'][file_proc]
            image_hdus = [i for i, h in enumerate(hdul) if (hasattr(h, 'data') and h.data is not None and h.data.ndim >= 2)]
            if image_hdus:
                hdu_idx = st.selectbox("Image HDU (proc)", options=image_hdus)
                arr = np.array(hdul[hdu_idx].data)
                if arr.ndim > 2:
                    st.warning(f"HDU {hdu_idx} has {arr.ndim} dimensions — using first slice for processing.")
                    arr = arr[0]
                if arr.ndim < 2:
                    st.error(f"HDU {hdu_idx} has invalid shape {arr.shape} for processing.")
                    continue
                # histogram
                fig, ax = plt.subplots()
                ax.hist(arr.flatten(), bins=256, log=True)
                ax.set_title('Pixel Value Distribution (log scale)')
                st.pyplot(fig)
                plt.close(fig)
                if st.button("Auto Enhance"):
                    # simple autoscale
                    norm = normalize_image(arr, stretch='asinh', percent=99.9)
                    pil_img = array_to_pil(arr, cmap=default_cmap, norm=norm)
                    buf = io.BytesIO(); pil_img.save(buf, format='PNG'); buf.seek(0)
                    st.image(buf)
                    st.download_button("Download enhanced PNG", data=buf, file_name=f"{os.path.splitext(file_proc)[0]}_enhanced.png", mime='image/png')

    # EXPORT tab
    with tabs[4]:
        st.subheader("Export / Conversion")
        files_export = st.multiselect("Select files to export", options=list(st.session_state['loaded'].keys()) if st.session_state['loaded'] else [])
        fmt = st.selectbox("Format", ["PNG", "TIFF", "JPEG"], index=0)
        dpi = st.number_input("DPI", min_value=72, max_value=1200, value=300)
        if st.button("Batch convert selected") and files_export:
            out_zip_buf = io.BytesIO()
            import zipfile
            with zipfile.ZipFile(out_zip_buf, 'w') as zf:
                for fname in files_export:
                    hdul = st.session_state['loaded'][fname]
                    # pick first image HDU
                    image_hdus = [i for i, h in enumerate(hdul) if (hasattr(h, 'data') and h.data is not None and h.data.ndim >= 2)]
                    if not image_hdus:
                        continue
                    arr = np.array(hdul[image_hdus[0]].data)
                    if arr.ndim > 2:
                        arr = arr[0]
                    if arr.ndim < 2:
                        # skip non-image HDU
                        continue
                    norm = normalize_image(arr, stretch=default_stretch)
                    pil_img = array_to_pil(arr, cmap=default_cmap, norm=norm)
                    buf = io.BytesIO(); pil_img.save(buf, format=fmt); buf.seek(0)
                    zf.writestr(f"{os.path.splitext(fname)[0]}.{fmt.lower()}", buf.read())
            out_zip_buf.seek(0)
            st.download_button("Download ZIP", data=out_zip_buf, file_name="converted_images.zip", mime='application/zip')

    st.sidebar.markdown("---")
    st.sidebar.caption("Made with ❤️ for astronomers — supports FITS metadata, visualization, conversion, and basic processing.")

# ------------------------- Headless CLI functions -------------------------

def batch_convert(input_patterns: List[str], outdir: str, out_format: str = 'png', dpi: int = 300):
    os.makedirs(outdir, exist_ok=True)
    files = []
    for pat in input_patterns:
        files.extend(glob.glob(pat))
    for fp in files:
        try:
            hdul = load_fits_data(fp)
            # choose first image HDU
            image_hdus = [i for i, h in enumerate(hdul) if (hasattr(h, 'data') and h.data is not None and h.data.ndim >= 2)]
            if not image_hdus:
                print(f"No image HDU in {fp}, skipping")
                continue
            arr = np.array(hdul[image_hdus[0]].data)
            if arr.ndim > 2:
                arr = arr[0]
            if arr.ndim < 2:
                print(f"No 2D image in {fp}, skipping")
                continue
            norm = normalize_image(arr, stretch='asinh')
            pil_img = array_to_pil(arr, cmap='gray', norm=norm)
            out_file = os.path.join(outdir, f"{os.path.splitext(os.path.basename(fp))[0]}.{out_format}")
            save_image_pil(pil_img, fmt=out_format.upper(), dpi=dpi, out_path=out_file)
            print(f"Saved {out_file}")
        except Exception as e:
            print(f"Failed {fp}: {e}")

# ------------------------- Simple FastAPI wrapper -------------------------

def build_api():
    if FastAPI is None:
        raise RuntimeError("FastAPI not installed. Install fastapi and uvicorn to enable REST API.")
    app = FastAPI(title="Streamlit FITS Tool API")

    @app.post('/convert')
    async def convert_file(file: UploadFile = File(...), fmt: str = 'png'):
        contents = await file.read()
        hdul = load_fits_data(io.BytesIO(contents))
        image_hdus = [i for i, h in enumerate(hdul) if (hasattr(h, 'data') and h.data is not None and h.data.ndim >= 2)]
        if not image_hdus:
            return {"error": "no image HDU"}
        arr = np.array(hdul[image_hdus[0]].data)
        if arr.ndim > 2:
            arr = arr[0]
        if arr.ndim < 2:
            return {"error": "no 2D image HDU"}
        norm = normalize_image(arr, stretch='asinh')
        pil_img = array_to_pil(arr, cmap='gray', norm=norm)
        buf = io.BytesIO(); pil_img.save(buf, format=fmt.upper()); buf.seek(0)
        return StreamingResponse(buf, media_type=f'image/{fmt}')

    return app

# ------------------------- Entrypoint -------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streamlit FITS Tool - CLI/Server utilities')
    parser.add_argument('--cli-convert', action='store_true', help='Run batch conversion in CLI mode and exit')
    parser.add_argument('--input', nargs='+', help='Input file patterns for CLI conversion (e.g. files/*.fits)')
    parser.add_argument('--outdir', default='out', help='Output directory for CLI conversion')
    parser.add_argument('--format', default='png', help='Output image format for CLI conversion')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for export')
    parser.add_argument('--api', action='store_true', help='Run REST API server instead of Streamlit UI')
    parser.add_argument('--host', default='127.0.0.1', help='Host for API server')
    parser.add_argument('--port', type=int, default=8000, help='Port for API server')
    args = parser.parse_args()

    if args.cli_convert:
        if not args.input:
            print('Provide --input patterns to convert')
            sys.exit(1)
        batch_convert(args.input, args.outdir, out_format=args.format, dpi=args.dpi)
        sys.exit(0)

    if args.api:
        if FastAPI is None:
            print('FastAPI not installed. Install fastapi and uvicorn to use API mode.')
            sys.exit(1)
        app = build_api()
        uvicorn.run(app, host=args.host, port=args.port)
        sys.exit(0)

    # If no CLI flags provided, try to start Streamlit UI
    if st is None:
        print('Streamlit not installed or running in non-Streamlit mode. To use the UI, run with: streamlit run streamlit_fits_tool_app.py')
        sys.exit(1)
    else:
        build_streamlit_ui()
        
    # PROCESSING tab
with tabs[3]:
    st.subheader("Processing Tools")
    st.write("Histogram and stretch editor")
    file_proc = st.selectbox(
        "Select file (process)",
        options=list(st.session_state['loaded'].keys()) if st.session_state['loaded'] else [],
        key='proc_file'
    )
    if file_proc:
        hdul = st.session_state['loaded'][file_proc]
        image_hdus = [i for i, h in enumerate(hdul) if (hasattr(h, 'data') and h.data is not None and h.data.ndim >= 2)]
        if image_hdus:
            hdu_idx = st.selectbox("Image HDU (proc)", options=image_hdus)
            arr = np.array(hdul[hdu_idx].data)
            if arr.ndim > 2:
                st.warning(f"HDU {hdu_idx} has {arr.ndim} dimensions — using first slice for processing.")
                arr = arr[0]
            if arr.ndim < 2:
                st.error(f"HDU {hdu_idx} has invalid shape {arr.shape} for processing.")
            else:
                fig, ax = plt.subplots()
                ax.hist(arr.flatten(), bins=256, log=True)
                ax.set_title('Pixel Value Distribution (log scale)')
                st.pyplot(fig)
                plt.close(fig)

                if st.button("Auto Enhance"):
                    norm = normalize_image(arr, stretch='asinh', percent=99.9)
                    pil_img = array_to_pil(arr, cmap=default_cmap, norm=norm)
                    buf = io.BytesIO(); pil_img.save(buf, format='PNG'); buf.seek(0)
                    st.image(buf)
                    st.download_button(
                        "Download enhanced PNG", data=buf,
                        file_name=f"{os.path.splitext(file_proc)[0]}_enhanced.png", mime='image/png'
                    )

    # EXPORT tab
    with tabs[4]:
        st.subheader("Export / Conversion")
        files_export = st.multiselect("Select files to export", options=list(st.session_state['loaded'].keys()) if st.session_state['loaded'] else [])
        fmt = st.selectbox("Format", ["PNG", "TIFF", "JPEG"], index=0)
        dpi = st.number_input("DPI", min_value=72, max_value=1200, value=300)
        if st.button("Batch convert selected") and files_export:
            out_zip_buf = io.BytesIO()
            import zipfile
            with zipfile.ZipFile(out_zip_buf, 'w') as zf:
                for fname in files_export:
                    hdul = st.session_state['loaded'][fname]
                    # pick first image HDU
                    image_hdus = [i for i, h in enumerate(hdul) if (hasattr(h, 'data') and h.data is not None and h.data.ndim >= 2)]
                    if not image_hdus:
                        continue
                    arr = np.array(hdul[image_hdus[0]].data)
                    if arr.ndim > 2:
                        arr = arr[0]
                    if arr.ndim < 2:
                        # skip non-image HDU
                        continue
                    norm = normalize_image(arr, stretch=default_stretch)
                    pil_img = array_to_pil(arr, cmap=default_cmap, norm=norm)
                    buf = io.BytesIO(); pil_img.save(buf, format=fmt); buf.seek(0)
                    zf.writestr(f"{os.path.splitext(fname)[0]}.{fmt.lower()}", buf.read())
            out_zip_buf.seek(0)
            st.download_button("Download ZIP", data=out_zip_buf, file_name="converted_images.zip", mime='application/zip')

    st.sidebar.markdown("---")
    st.sidebar.caption("Made with ❤️ for astronomers — supports FITS metadata, visualization, conversion, and basic processing.")

# ------------------------- Headless CLI functions -------------------------

def batch_convert(input_patterns: List[str], outdir: str, out_format: str = 'png', dpi: int = 300):
    os.makedirs(outdir, exist_ok=True)
    files = []
    for pat in input_patterns:
        files.extend(glob.glob(pat))
    for fp in files:
        try:
            hdul = load_fits_data(fp)
            # choose first image HDU
            image_hdus = [i for i, h in enumerate(hdul) if (hasattr(h, 'data') and h.data is not None and h.data.ndim >= 2)]
            if not image_hdus:
                print(f"No image HDU in {fp}, skipping")
                continue
            arr = np.array(hdul[image_hdus[0]].data)
            if arr.ndim > 2:
                arr = arr[0]
            if arr.ndim < 2:
                print(f"No 2D image in {fp}, skipping")
                continue
            norm = normalize_image(arr, stretch='asinh')
            pil_img = array_to_pil(arr, cmap='gray', norm=norm)
            out_file = os.path.join(outdir, f"{os.path.splitext(os.path.basename(fp))[0]}.{out_format}")
            save_image_pil(pil_img, fmt=out_format.upper(), dpi=dpi, out_path=out_file)
            print(f"Saved {out_file}")
        except Exception as e:
            print(f"Failed {fp}: {e}")

# ------------------------- Simple FastAPI wrapper -------------------------

def build_api():
    if FastAPI is None:
        raise RuntimeError("FastAPI not installed. Install fastapi and uvicorn to enable REST API.")
    app = FastAPI(title="Streamlit FITS Tool API")

    @app.post('/convert')
    async def convert_file(file: UploadFile = File(...), fmt: str = 'png'):
        contents = await file.read()
        hdul = load_fits_data(io.BytesIO(contents))
        image_hdus = [i for i, h in enumerate(hdul) if (hasattr(h, 'data') and h.data is not None and h.data.ndim >= 2)]
        if not image_hdus:
            return {"error": "no image HDU"}
        arr = np.array(hdul[image_hdus[0]].data)
        if arr.ndim > 2:
            arr = arr[0]
        if arr.ndim < 2:
            return {"error": "no 2D image HDU"}
        norm = normalize_image(arr, stretch='asinh')
        pil_img = array_to_pil(arr, cmap='gray', norm=norm)
        buf = io.BytesIO(); pil_img.save(buf, format=fmt.upper()); buf.seek(0)
        return StreamingResponse(buf, media_type=f'image/{fmt}')

    return app

# ------------------------- Entrypoint -------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Streamlit FITS Tool - CLI/Server utilities')
    parser.add_argument('--cli-convert', action='store_true', help='Run batch conversion in CLI mode and exit')
    parser.add_argument('--input', nargs='+', help='Input file patterns for CLI conversion (e.g. files/*.fits)')
    parser.add_argument('--outdir', default='out', help='Output directory for CLI conversion')
    parser.add_argument('--format', default='png', help='Output image format for CLI conversion')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for export')
    parser.add_argument('--api', action='store_true', help='Run REST API server instead of Streamlit UI')
    parser.add_argument('--host', default='127.0.0.1', help='Host for API server')
    parser.add_argument('--port', type=int, default=8000, help='Port for API server')
    args = parser.parse_args()

    if args.cli_convert:
        if not args.input:
            print('Provide --input patterns to convert')
            sys.exit(1)
        batch_convert(args.input, args.outdir, out_format=args.format, dpi=args.dpi)
        sys.exit(0)

    if args.api:
        if FastAPI is None:
            print('FastAPI not installed. Install fastapi and uvicorn to use API mode.')
            sys.exit(1)
        app = build_api()
        uvicorn.run(app, host=args.host, port=args.port)
        sys.exit(0)

    # If no CLI flags provided, try to start Streamlit UI
    if st is None:
        print('Streamlit not installed or running in non-Streamlit mode. To use the UI, run with: streamlit run streamlit_fits_tool_app.py')
        sys.exit(1)
    else:
        build_streamlit_ui()
