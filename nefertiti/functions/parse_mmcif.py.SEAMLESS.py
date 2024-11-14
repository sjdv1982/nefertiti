import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("mmcif_file")
parser.add_argument("outfile")
parser.add_argument("--no-auth-chains", action="store_true")
parser.add_argument("--no-auth-residues", action="store_true")
args = parser.parse_args()

input_files = [
    {"name": "infile.cif", "mapping": args.mmcif_file},
    {"name": "parse_mmcif.py", "mapping": sys.argv[0][: -len(".SEAMLESS.py")]},
]
result_files = {"outfile.npy": args.outfile}

#############################################################

order = ["python", "parse_mmcif.py", "infile.cif", "outfile.npy"]
if args.no_auth_chains:
    order.append("--no-auth-chains")
if args.no_auth_residues:
    order.append("--no-auth-residues")

interface = {
    "@order": order,
    "files": input_files,
    "results": result_files,
    "values": ["python", "outfile.npy"],
}

import json

print(json.dumps(interface))
