import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from dentscan import mesh as M, pipeline as P, validation as V  # noqa: E402

st.set_page_config(page_title="dentscan", layout="wide")
st.title("dentscan: research scanner pipeline")
st.warning("Research tool. Not validated or cleared for clinical use or patient-specific milling.")

if "work" not in st.session_state:
    st.session_state.work = Path(tempfile.mkdtemp(prefix="dentscan_"))
    st.session_state.mesh = None
work = st.session_state.work


def show(mesh, max_tri=40_000):
    m = mesh
    if len(m.triangles) > max_tri:
        m = m.simplify_quadric_decimation(max_tri)
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    fig = go.Figure(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                              i=t[:, 0], j=t[:, 1], k=t[:, 2], color="ivory", flatshading=True))
    fig.update_layout(scene=dict(aspectmode="data"), height=600, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)


tab1, tab2, tab3, tab4 = st.tabs(["1 Capture", "2 Clean + Scale", "3 Validate", "4 Export"])

with tab1:
    vid = st.file_uploader("Upload phone video (orbit around a stone model)", type=["mp4", "mov", "mkv"])
    c1, c2, c3 = st.columns(3)
    every_n = c1.slider("Use every Nth frame", 1, 15, 5)
    blur = c2.slider("Sharpness threshold", 20, 300, 100)
    quality = c3.selectbox("COLMAP quality", ["low", "medium", "high"], index=1)
    if vid and st.button("Reconstruct"):
        vp = work / "input.mp4"
        vp.write_bytes(vid.read())
        with st.spinner("Extracting frames..."):
            n = P.extract_frames(vp, work / "frames", every_n, blur)
        st.write(f"{n} sharp frames kept")
        if n < 20:
            st.error("Too few usable frames. Re-record slower with better light.")
        else:
            try:
                with st.spinner("Running COLMAP (this can take a while)..."):
                    path = P.run_colmap(work / "frames", work / "colmap", quality)
                st.session_state.mesh = M.load_mesh(path)
                st.success("Mesh reconstructed.")
            except Exception as e:
                st.error(str(e))
    ply = st.file_uploader("...or load an existing mesh (PLY/STL/OBJ)", type=["ply", "stl", "obj"])
    if ply and st.button("Load mesh"):
        p = work / ply.name
        p.write_bytes(ply.read())
        st.session_state.mesh = M.load_mesh(p)

with tab2:
    mesh = st.session_state.mesh
    if mesh is None:
        st.info("Reconstruct or load a mesh first.")
    else:
        tri = st.number_input("Target triangles", 10_000, 1_000_000, 200_000, 10_000)
        smooth = st.slider("Taubin smoothing iterations", 0, 10, 3)
        meas = st.number_input("Reference distance in model units", value=1.0, format="%.6f")
        real = st.number_input("Same distance in real mm", value=10.0)
        if st.button("Clean and scale"):
            m = M.clean_mesh(mesh, int(tri), smooth)
            m, f = M.scale_to_real(m, meas, real)
            st.session_state.mesh = m
            st.success(f"Scale factor {f:.5f}")
        show(st.session_state.mesh)

with tab3:
    mesh = st.session_state.mesh
    if mesh is None:
        st.info("Load a scan of a precision sphere first.")
    else:
        nominal = st.number_input("Nominal sphere diameter (mm)", value=25.4)
        if st.button("Compute accuracy"):
            st.json(V.sphere_report(np.asarray(mesh.vertices), nominal))
        st.caption("Isolate the sphere in the mesh before validating.")

with tab4:
    mesh = st.session_state.mesh
    if mesh is None:
        st.info("Nothing to export yet.")
    else:
        out = work / "scan.stl"
        M.export_stl(mesh, out)
        st.download_button("Download STL", out.read_bytes(), "scan.stl", "model/stl")
