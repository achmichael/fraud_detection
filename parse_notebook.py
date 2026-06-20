import json
import sys

def parse_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb['cells']):
        print(f"=== Cell {i+1} ({cell['cell_type']}) ===")
        print("--- Source ---")
        print("".join(cell.get('source', [])))
        print("--- Output ---")
        
        if cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            for out in outputs:
                if out['output_type'] == 'stream':
                    print("".join(out.get('text', [])))
                elif out['output_type'] in ('execute_result', 'display_data'):
                    data = out.get('data', {})
                    if 'text/plain' in data:
                        print("".join(data['text/plain']))
                    elif 'text/html' in data:
                        print("[HTML Output present]")
                elif out['output_type'] == 'error':
                    print(f"Error: {out.get('ename')}: {out.get('evalue')}")
        print("\n")

if __name__ == '__main__':
    parse_notebook(sys.argv[1])
