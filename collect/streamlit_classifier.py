import streamlit as st
import os
import yaml
import json
from pathlib import Path

# 1. Load config
GLOBAL_CONFIG = "/tmp/is_the_mountain_out_config.json"
if os.path.exists(GLOBAL_CONFIG):
    with open(GLOBAL_CONFIG, "r") as f:
        config = json.load(f)
    data_root = Path(config["data_root"])
else:
    data_root = Path("data")

st.set_page_config(page_title="Mountain Classifier", layout="wide")

# Custom CSS for iOS-style selection
st.markdown(
    """
    <style>
    .stCheckbox {
        position: absolute;
        top: 5px;
        left: 5px;
        z-index: 10;
    }
    .img-selected {
        border: 4px solid #007aff !important;
        border-radius: 8px;
    }
    .img-container {
        position: relative;
        border: 4px solid transparent;
        transition: all 0.2s;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🏔️ Batch Mountain Classifier")

# 2. Efficient Label Loading
labels_p = data_root / "labels.yaml"
if "labels" not in st.session_state:
    if labels_p.exists():
        with open(labels_p, "r") as f:
            st.session_state.labels = yaml.safe_load(f) or {}
    else:
        st.session_state.labels = {}


# 3. Incremental Scanning (Efficiency)
def get_unlabeled_batch(root, limit=50):
    """Finds the first N unlabeled images without scanning the entire tree."""
    batch = []
    # Use os.walk for faster sequential access
    for r, dirs, files in os.walk(root):
        for f in sorted(files):
            if f.endswith(".jpg"):
                full_path = Path(r) / f
                rel_path = str(full_path.relative_to(root))
                if rel_path not in st.session_state.labels:
                    batch.append(full_path)
                    if len(batch) >= limit:
                        return batch
    return batch


# 4. Batch Actions
def apply_batch_label(image_paths, label):
    for p in image_paths:
        rel = str(p.relative_to(data_root))
        st.session_state.labels[rel] = label

    with open(labels_p, "w") as f:
        yaml.safe_dump(st.session_state.labels, f)
    st.toast(
        f"Tagged {len(image_paths)} images as {'OUT' if label == 1 else 'NOT OUT'}"
    )
    st.rerun()


# 5. UI Layout
current_batch = get_unlabeled_batch(data_root)

if not current_batch:
    st.success("🎉 All clear! No more unlabeled images found.")
    st.stop()

# Header Stats
out_count = list(st.session_state.labels.values()).count(1)
not_count = list(st.session_state.labels.values()).count(0)
st.sidebar.metric("Labeled", len(st.session_state.labels))
st.sidebar.metric("Mountain Out", out_count)
st.sidebar.metric("Not Out", not_count)

# Floating Action Bar for batch operations
if "selected" not in st.session_state:
    st.session_state.selected = set()

# Clean up selected set if paths are no longer in current batch
st.session_state.selected = {p for p in st.session_state.selected if p in current_batch}

cols = st.columns(5)
for i, img_path in enumerate(current_batch):
    with cols[i % 5]:
        is_selected = img_path in st.session_state.selected

        # Approximate iOS selection: Checkbox overlay + border highlight
        selected = st.checkbox(
            "Select", key=f"cb_{i}", label_visibility="collapsed", value=is_selected
        )
        if selected:
            st.session_state.selected.add(img_path)
        else:
            st.session_state.selected.discard(img_path)

        css_class = "img-selected" if selected else ""
        st.markdown(f'<div class="img-container {css_class}">', unsafe_allow_html=True)
        st.image(str(img_path))
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption(f"{img_path.parent.name}")

st.divider()

# Footer Controls
sticky_cols = st.columns([1, 1, 1, 2])
with sticky_cols[0]:
    if st.button(
        "🏔️ ALL OUT (1)",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.selected,
    ):
        apply_batch_label(list(st.session_state.selected), 1)
        st.session_state.selected = set()

with sticky_cols[1]:
    if st.button(
        "☁️ ALL NOT OUT (0)",
        use_container_width=True,
        disabled=not st.session_state.selected,
    ):
        apply_batch_label(list(st.session_state.selected), 0)
        st.session_state.selected = set()

with sticky_cols[2]:
    if st.button("Clear Selection", use_container_width=True):
        st.session_state.selected = set()
        st.rerun()

with sticky_cols[3]:
    st.info(
        f"💡 Selected {len(st.session_state.selected)} images. Select images in the grid above to bulk-label them."
    )
