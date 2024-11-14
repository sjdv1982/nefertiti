import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("pdb_file")
parser.add_argument("outfile")
args = parser.parse_args()

input_files = [
    {"name": "infile.pdb", "mapping": args.pdb_file},
    {"name": "parse_pdb.py", "mapping": sys.argv[0][: -len(".SEAMLESS.py")]},
]
result_files = {"outfile.npy": args.outfile}

#############################################################

order = ["python", "parse_pdb.py", "infile.pdb", "outfile.npy"]
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
