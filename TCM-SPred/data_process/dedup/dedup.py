#!/usr/bin/env python3
# dedup_txt.py
import sys
from pathlib import Path

def dedup_txt(in_file: Path, out_file: Path):
 
    seen = set()
    with in_file.open(encoding='utf-8') as fin, \
         out_file.open('w', encoding='utf-8') as fout:
        for line in fin:
            line = line.rstrip('\n')   
            if line not in seen:      
                seen.add(line)
                fout.write(line + '\n')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python dedup_txt.py input.txt [output_dedup.txt]')
        sys.exit(1)

    in_path  = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.with_suffix('.dedup.txt')
    dedup_txt(in_path, out_path)
    print(f'Done! Deduplicated {in_path} -> {out_path}')