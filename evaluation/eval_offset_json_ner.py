import argparse
import json
import os
from collections import Counter, defaultdict

PRESETS = {
    "genia": ("cell_line", "cell_type", "DNA", "RNA", "protein"),
    "ace2005": ("PER", "ORG", "GPE", "LOC", "FAC", "VEH", "WEA"),
    "conll2003": ("PER", "ORG", "LOC", "MISC"),
    "ontonotes5": ("CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"),
}

def parse_entities(json_str):
    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            return []
        entities = []
        for item in data:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            label = item.get("label")
            start = item.get("start")
            end = item.get("end")
            
            if text is not None and label is not None and start is not None and end is not None:
                entities.append((str(text), str(label), int(start), int(end)))
        return entities
    except json.JSONDecodeError:
        return []
    except Exception:
        return []

def compute_f1(preds, golds):
    tp = 0
    fp = 0
    fn = 0
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for p_list, g_list in zip(preds, golds):
        p_counter = Counter(p_list)
        g_counter = Counter(g_list)

        for entity, count in p_counter.items():
            tp_count = min(count, g_counter[entity])
            tp += tp_count
            per_type[entity[1]]["tp"] += tp_count
            
        for entity, count in p_counter.items():
            fp_count = max(0, count - g_counter[entity])
            fp += fp_count
            per_type[entity[1]]["fp"] += fp_count
            
        for entity, count in g_counter.items():
            fn_count = max(0, count - p_counter[entity])
            fn += fn_count
            per_type[entity[1]]["fn"] += fn_count

    def safe_div(n, d):
        return n / d if d > 0 else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    per_type_results = {}
    for label, counts in per_type.items():
        p = safe_div(counts["tp"], counts["tp"] + counts["fp"])
        r = safe_div(counts["tp"], counts["tp"] + counts["fn"])
        f = safe_div(2 * p * r, p + r)
        per_type_results[label] = {
            "precision": round(p, 6)*100,
            "recall": round(r, 6)*100,
            "f1": round(f, 6)*100,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"]
        }

    return {
        "micro": {
            "precision": round(precision, 6)*100,
            "recall": round(recall, 6)*100,
            "f1": round(f1, 6)*100,
            "tp": tp,
            "fp": fp,
            "fn": fn
        },
        "per_type": per_type_results
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate offset-based NER predictions in JSONL format.")
    parser.add_argument(
        "--file", 
        type=str,
        help="Path to the generated_predictions.jsonl file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Directory where results will be saved. Defaults to file directory."
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Explicit label list for per-label reporting. If unset, uses --preset."
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        required=True,
        help="Dataset label preset for per-label reporting."
    )
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(os.path.abspath(file_path))

    preds = []
    golds = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                predict_str = data.get("predict", "[]")
                label_str = data.get("label", "[]")
                
                if isinstance(predict_str, list):
                    predict_str = json.dumps(predict_str)
                if isinstance(label_str, list):
                    label_str = json.dumps(label_str)
                
                preds.append(parse_entities(predict_str))
                golds.append(parse_entities(label_str))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line[:50]}...")
                continue

    results = compute_f1(preds, golds)
    results["samples"] = len(preds)
    labels = tuple(PRESETS[args.preset]) if args.labels is None else tuple(args.labels)
    per_type = results.get("per_type", {})
    ordered_per_label = {}
    for lab in labels:
        m = per_type.get(lab, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0})
        ordered_per_label[lab] = m
    results["per_label"] = ordered_per_label

    print(json.dumps(results, ensure_ascii=False, indent=2))

    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, "results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to {output_file}")

if __name__ == "__main__":
    main()