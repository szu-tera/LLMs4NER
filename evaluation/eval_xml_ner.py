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

def parse_xml_spans(xml_text, allowed_tags):
    if not xml_text:
        return []
    
    tag_group = "|".join(sorted(allowed_tags, key=len, reverse=True))
    event_pattern = re.compile(rf"<(?:/)?({tag_group})>")

    clean_len = 0
    entities = []
    stack = [] 

    cursor = 0
    for m in event_pattern.finditer(xml_text):
        segment = xml_text[cursor: m.start()]
        if segment:
            clean_len += len(segment)

        tag = m.group(1)
        is_close = xml_text[m.start(): m.start()+2] == "</"
        
        if not is_close:
            stack.append((tag, clean_len))
        else:
            idx = None
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == tag:
                    idx = i
                    break
            if idx is not None:
                open_tag, start_pos = stack[idx]
                del stack[idx:]
                end_pos = clean_len
                if end_pos > start_pos:
                    entities.append((open_tag, int(start_pos), int(end_pos)))

        cursor = m.end()

    tail = xml_text[cursor:]
    if tail:
        clean_len += len(tail)

    return entities

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

def strip_xml_tags(xml_text, allowed_tags):
    if not xml_text:
        return ""
    tag_group = "|".join(sorted(allowed_tags, key=len, reverse=True))
    event_pattern = re.compile(rf"<(?:/)?({tag_group})>")
    out = []
    cursor = 0
    for m in event_pattern.finditer(xml_text):
        out.append(xml_text[cursor:m.start()])
        cursor = m.end()
    out.append(xml_text[cursor:])
    return "".join(out)

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

def main():
    parser = argparse.ArgumentParser(description="Evaluate XML NER matching only.")
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
        help="Dataset label preset for parsing and reporting."
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

    labels_list = tuple(PRESETS[args.preset]) if args.labels is None else tuple(args.labels)
    allowed_tags = set(labels_list)

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
                predict_str = str(data.get("predict", ""))
                label_str = str(data.get("label", ""))
                
                p_spans = parse_xml_spans(predict_str, allowed_tags)
                g_spans = parse_xml_spans(label_str, allowed_tags)
                p_clean = strip_xml_tags(predict_str, allowed_tags)
                g_clean = strip_xml_tags(label_str, allowed_tags)
                if p_clean == g_clean:
                    preds.append(p_spans)
                    golds.append(g_spans)
                    strict_count += 1
                else:
                    preds.append(spans_to_texts(p_spans, p_clean))
                    golds.append(spans_to_texts(g_spans, g_clean))
                    matching_count += 1
                    ambiguous = get_ambiguous_entities(p_spans, p_clean)
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
    
    per_type = results.get("per_type", {})
    ordered_per_label = {}
    for lab in labels_list:
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
