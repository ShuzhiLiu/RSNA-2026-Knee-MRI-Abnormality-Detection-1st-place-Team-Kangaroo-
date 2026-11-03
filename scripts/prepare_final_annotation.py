#!/usr/bin/env python3
"""Build final Astra input artifacts or expose its original MRI MCP viewer.

This tool performs no annotation model calls. Exact released numeric labels can
be consumed directly with final_solution.py prepare.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('astra-v2-input',help='build IND24 initial images and clinical prompt from one prepared study spec')
    p.add_argument('--study-spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p = sub.add_parser('astra-viewer',help='serve the original MRI viewer over stdin/stdout')
    p.add_argument('--version',choices=('v1','v2'),required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='astra-v2-input':
        from rsna_knee.annotation.astra_v2.build import build_case
        if args.output.exists():
            raise FileExistsError(args.output)
        source = ROOT/'src/rsna_knee/annotation/astra_v2'
        args.output.mkdir(parents=True)
        build_case(json.loads(args.study_spec.read_text()),args.output,(source/'acquire_prompt.txt').read_text())
    else:
        sys.argv = ['astra-viewer','--case',str(args.case),'--log',str(args.log)]
        runpy.run_module(f'rsna_knee.annotation.astra_{args.version}.viewer',run_name='__main__')


if __name__=='__main__':main()
