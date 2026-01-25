import argparse
import json
import os
import re
from collections import Counter, defaultdict

PRESETS = {
    "genia": ("cell_line", "cell_type", "DNA", "RNA", "protein"),
    "ace2005": ("PER", "ORG", "GPE", "LOC", "FAC", "VEH", "WEA"),
    "conll2003": ("PER", "ORG", "LOC", "MISC"),
    "conll03_so": ("A", "B", "C", "D"),
    "conll03_se": ("A", "B", "C", "D"),
    "ontonotes5": ("CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"),
}

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())

def extract_spans(br_text):
    if not isinstance(br_text, str):
        br_text = str(br_text)
    br_text = br_text.strip()
    event = re.compile(r"(\[)|(\|\s*([^\]]+)\])")
    parts = []
    clen = 0
    spans = []
    stack = []
    cursor = 0
    for m in event.finditer(br_text):
        seg = br_text[cursor : m.start()]
        if seg:
            if m.group(1):
                pass
            else:
                seg = seg.rstrip()
            parts.append(seg)
            clen += len(seg)
        if m.group(1):
            stack.append(clen)
        else:
            lab = norm(m.group(3))
            if stack:
                st = stack.pop()
                if clen > st and lab:
                    spans.append((lab, st, clen))
        cursor = m.end()
    tail = br_text[cursor:]
    if tail:
        parts.append(tail)
        clen += len(tail)
    clean = "".join(parts)
    return clean, spans

def spans_to_texts(spans, clean_text):
    res = []
    for lab, s, e in spans:
        if 0 <= s <= e <= len(clean_text):
            res.append((lab, clean_text[s:e]))
    return res

def has_repeated_entity(spans, clean_text):
    for _, s, e in spans:
        if 0 <= s <= e <= len(clean_text):
            ent = clean_text[s:e].strip()
            if not ent:
                continue
            pat = re.compile(rf"(?<!\w){re.escape(ent)}(?!\w)")
            if len(pat.findall(clean_text)) > 1:
                return True
    return False

def get_ambiguous_entities(spans, clean_text):
    res = []
    for lab, s, e in spans:
        if 0 <= s <= e <= len(clean_text):
            ent = clean_text[s:e].strip()
            if not ent:
                continue
            pat = re.compile(rf"(?<!\w){re.escape(ent)}(?!\w)")
            if len(pat.findall(clean_text)) > 1:
                res.append({"label": lab, "text": ent})
    return res

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
            per_type[entity[0]]["tp"] += tp_count
            
        for entity, count in p_counter.items():
            fp_count = max(0, count - g_counter[entity])
            fp += fp_count
            per_type[entity[0]]["fp"] += fp_count
            
        for entity, count in g_counter.items():
            fn_count = max(0, count - p_counter[entity])
            fn += fn_count
            per_type[entity[0]]["fn"] += fn_count

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
    parser = argparse.ArgumentParser(description="Evaluate bracketed NER matching only.")
    parser.add_argument(
        "--file", 
        type=str,
        required=True,
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
        mismatch_entries = []
        strict_count = 0
        matching_count = 0
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                predict_str = data.get("predict", "")
                label_str = data.get("label", "")
                
                p_clean, pred_spans = extract_spans(predict_str)
                g_clean, gold_spans = extract_spans(label_str)
                if p_clean == g_clean:
                    preds.append(pred_spans)
                    golds.append(gold_spans)
                    strict_count += 1
                else:
                    preds.append(spans_to_texts(pred_spans, p_clean))
                    golds.append(spans_to_texts(gold_spans, g_clean))
                    matching_count += 1
                    ambiguous = get_ambiguous_entities(pred_spans, p_clean)
                    if ambiguous:
                        mismatch_entries.append({
                            "index": idx,
                            "predict": predict_str,
                            "gold": label_str,
                            "clean_pred": p_clean,
                            "clean_gold": g_clean,
                            "ambiguous_entities": ambiguous
                        })
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line[:50]}...")
                continue

    results = compute_f1(preds, golds)
    results["samples"] = len(preds)
    results["mode_counts"] = {"strict": strict_count, "matching": matching_count}
    labels = tuple(PRESETS[args.preset]) if args.labels is None else tuple(args.labels)
    per_type = results.get("per_type", {})
    ordered_per_label = {}
    for lab in labels:
        m = per_type.get(lab, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0})
        ordered_per_label[lab] = m
    results["per_label"] = ordered_per_label
    console_results = {k: v for k, v in results.items() if k != "mode_counts"}
    print(json.dumps(console_results, ensure_ascii=False, indent=2))

    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, "results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to {output_file}")
    if mismatch_entries:
        mismatch_file = os.path.join(out_dir, "check_mismatch_samples.json")
        with open(mismatch_file, "w", encoding="utf-8") as mf:
            json.dump(mismatch_entries, mf, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
