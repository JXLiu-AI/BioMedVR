"""Generate _confuse.json (LLM confusion-aware attributes) for one dataset.
Usage: python gen_confuse.py <dataset> [--num 5]
"""
import os, sys, json, time, argparse, re
from openai import OpenAI

DOWNSTREAM_PATH = ""

def load_classes(dataset):
    """Read class list from existing _des.json."""
    p = f'attributes/gpt3/gpt3/{dataset}_des.json'
    return list(json.load(open(p)).keys())

def gen_for_class(client, model, dataset, classname, num):
    sys_msg = "You are an expert in fine-grained visual recognition."
    user_msg = (
        f'Generate the {num} most visually confusing negative descriptions for the category '
        f'"{classname}" in the {dataset} dataset. Each description should portray a visually similar but '
        f'semantically incorrect appearance that could be mistaken for "{classname}". '
        f'Reply as a numbered list, one per line, each at least 20 words.'
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
        temperature=0.7,
        max_tokens=400,
    )
    txt = resp.choices[0].message.content
    # parse numbered list
    items = []
    for line in txt.splitlines():
        m = re.match(r'^\s*\d+[\.\)\:\-]\s*(.+)$', line)
        if m:
            s = m.group(1).strip()
            if len(s) >= 20:
                items.append(s)
    if len(items) < num:
        # also catch lines without numbering
        for line in txt.splitlines():
            line = line.strip()
            if len(line) >= 20 and not re.match(r'^\s*\d+', line) and not line.startswith('#'):
                items.append(line)
    return items[:num]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--num', type=int, default=5)
    ap.add_argument('--model', default='gpt-4.1')
    ap.add_argument('--out_dir', default='attributes/gpt3/gpt3')
    ap.add_argument('--base_url', default=None)
    args = ap.parse_args()

    api_key = os.environ['OPENAI_API_KEY']
    if args.base_url:
        client = OpenAI(api_key=api_key, base_url=args.base_url)
    else:
        client = OpenAI(api_key=api_key)

    classes = load_classes(args.dataset)
    print(f'[{args.dataset}] {len(classes)} classes; model={args.model}')
    out_path = f'{args.out_dir}/{args.dataset}_confuse.json'
    out = {}
    if os.path.exists(out_path):
        out = json.load(open(out_path))
        print(f'  found existing {out_path}, will resume.')
    for i, cls in enumerate(classes):
        if cls in out and len(out[cls]) >= args.num:
            continue
        for attempt in range(3):
            try:
                items = gen_for_class(client, args.model, args.dataset, cls, args.num)
                if len(items) >= 1:
                    out[cls] = items
                    print(f'  [{i+1}/{len(classes)}] {cls}: {len(items)} attrs')
                    break
            except Exception as e:
                print(f'  [{i+1}/{len(classes)}] {cls}: {type(e).__name__}: {e}; retry {attempt+1}')
                time.sleep(2 + attempt * 3)
        else:
            print(f'  [{i+1}/{len(classes)}] {cls}: FAILED after retries')
            out[cls] = []
        # save progress every 10
        if (i+1) % 10 == 0:
            json.dump(out, open(out_path, 'w'), indent=2)
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f'wrote {out_path}: {sum(len(v) for v in out.values())} total attrs across {len(out)} classes')

if __name__ == '__main__':
    main()
