import sys
from pathlib import Path
import streamlit as st
import json

# Fix module path so `from src...` imports work when running this page directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT_MODEL = "data/models/player_embedding_model.pkl"

st.set_page_config(page_title="Cluster Admin", layout="wide")
st.title("Cluster Labels Admin")

@st.cache_resource
def load_model(path: str):
    from src.modeling.similarity import PlayerEmbeddingModel
    # Let exceptions propagate so the calling code can show tracebacks
    return PlayerEmbeddingModel.load(path)


p = Path(ROOT_MODEL)
model = None
if not p.exists():
    st.error(
        "Model file not found at '" + str(p.resolve()) + "'.\nRun the training script to create it:\n`python3 -m src.modeling.train_player_embedding`"
    )
else:
    try:
        model = load_model(ROOT_MODEL)
    except Exception as e:
        import traceback

        st.error(f"Failed to load model from '{p.resolve()}'. See traceback below.")
        st.code(traceback.format_exc())

if model is None:
    st.stop()
else:
    df = model.df
    clusters = sorted(list(set(model.role_labels)))

    st.markdown(f"**Model contains {len(df)} players across {len(clusters)} clusters.**")

    st.info("Edit labels below and click **Apply & Save Mapping** to persist labels into the model file.")

    # Build editable form
    with st.form("cluster_labels_form"):
        new_map = {}
        for cid in clusters:
            # current label prefers model mapping, then global ROLE_LABELS
            label_current = None
            try:
                label_current = getattr(model, 'cluster_label_map', {}) or {}
                label_current = label_current.get(int(cid)) if label_current else None
            except Exception:
                label_current = None

            if not label_current:
                try:
                    from src.modeling.similarity import ROLE_LABELS

                    entry = ROLE_LABELS.get(int(cid))
                    if isinstance(entry, dict):
                        label_current = entry.get('label')
                    elif isinstance(entry, str):
                        label_current = entry
                except Exception:
                    label_current = None

            if not label_current:
                label_current = f"Cluster {cid}"

            members = df[df['role_cluster'] == int(cid)]['Player'].dropna().unique().tolist()[:8]

            st.markdown("---")
            st.markdown(f"**Cluster {cid}** — Current label: **{label_current}**")
            st.write("Sample members:")
            st.write(members)

            new_label = st.text_input(f"Label for cluster {cid}", value=label_current, key=f"lbl_{cid}")
            new_map[int(cid)] = new_label

        apply_btn = st.form_submit_button("Apply & Save Mapping")

    if apply_btn:
        try:
            model.set_cluster_label_map(new_map)
            model.save(ROOT_MODEL)
            st.success(f"Saved mapping to {ROOT_MODEL}")
            # Some Streamlit versions may not expose experimental_rerun; show reload hint instead
            if hasattr(st, 'experimental_rerun'):
                try:
                    st.experimental_rerun()
                except Exception:
                    st.info("Reload the page to see the updated labels.")
            else:
                st.info("Reload the page to see the updated labels.")
        except Exception as e:
            st.error(f"Could not save mapping: {e}")

    # Download current mapping
    try:
        current_map = getattr(model, 'cluster_label_map', {}) or {}
        mapping_json = json.dumps({int(k): v for k, v in current_map.items()}, ensure_ascii=False, indent=2)
    except Exception:
        mapping_json = json.dumps({}, indent=2)

    st.download_button("Download mapping JSON", data=mapping_json, file_name="cluster_mapping.json", mime="application/json")

    # Option to export template based on current ROLE_LABELS
    if st.button("Export ROLE_LABELS template"):
        try:
            from src.modeling.similarity import ROLE_LABELS

            template = {int(k): (v['label'] if isinstance(v, dict) else v) for k, v in ROLE_LABELS.items()}
            st.download_button("Download ROLE_LABELS template", data=json.dumps(template, ensure_ascii=False, indent=2), file_name="role_labels_template.json", mime="application/json")
        except Exception as e:
            st.error(f"Could not export template: {e}")
