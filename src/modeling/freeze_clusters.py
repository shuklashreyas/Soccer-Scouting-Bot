import json
import argparse
from pathlib import Path
from src.modeling.similarity import PlayerEmbeddingModel


def list_clusters(model_path: str, max_examples: int = 8):
    model = PlayerEmbeddingModel.load(model_path)
    df = model.df
    clusters = sorted(list(set(model.role_labels)))
    out = {}
    for cid in clusters:
        members = df[df['role_cluster'] == int(cid)]['Player'].dropna().unique().tolist()
        label = model._cluster_name(int(cid))
        out[int(cid)] = {
            'label': label,
            'size': len(members),
            'examples': members[:max_examples]
        }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def write_template(model_path: str, out_path: str):
    model = PlayerEmbeddingModel.load(model_path)
    clusters = sorted(list(set(model.role_labels)))
    template = {int(cid): model._cluster_name(int(cid)) for cid in clusters}
    Path(out_path).write_text(json.dumps(template, indent=2, ensure_ascii=False))
    print(f"Wrote template mapping to {out_path}")


def apply_mapping(model_path: str, mapping_path: str, out_model_path: str = None):
    model = PlayerEmbeddingModel.load(model_path)
    mapping = json.loads(Path(mapping_path).read_text())
    model.set_cluster_label_map(mapping)
    # If out_model_path provided, save there, otherwise overwrite original
    save_path = out_model_path if out_model_path else model_path
    model.save(save_path)
    print(f"Applied mapping and saved model to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect and freeze cluster labels in a persisted PlayerEmbeddingModel')
    sub = parser.add_subparsers(dest='cmd')

    p_list = sub.add_parser('list', help='List clusters with sample members')
    p_list.add_argument('model', help='Path to pickled PlayerEmbeddingModel')

    p_template = sub.add_parser('template', help='Write a JSON template mapping cluster->label')
    p_template.add_argument('model', help='Path to pickled PlayerEmbeddingModel')
    p_template.add_argument('out', help='Output JSON path')

    p_apply = sub.add_parser('apply', help='Apply a mapping JSON to the persisted model')
    p_apply.add_argument('model', help='Path to pickled PlayerEmbeddingModel')
    p_apply.add_argument('mapping', help='Mapping JSON file path')
    p_apply.add_argument('--out', help='Optional output model path (defaults to overwrite)')

    args = parser.parse_args()

    if args.cmd == 'list':
        list_clusters(args.model)
    elif args.cmd == 'template':
        write_template(args.model, args.out)
    elif args.cmd == 'apply':
        apply_mapping(args.model, args.mapping, out_model_path=args.out)
    else:
        parser.print_help()
