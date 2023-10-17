import math, sys, os, gc, re, time
import argparse

parser = argparse.ArgumentParser(
    add_help=False,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Unified Domain Cutter for structure-based domain parsing",
    epilog="""An example:\npython Run_UniDoc_from_scratch_structure.py -i seq.pdb -c Chain \n """,
)
arghelp = parser.add_argument_group("help information")
arghelp.add_argument("-h", "--help", action="help", help="show this message and exit")

argrequired = parser.add_argument_group("mandatory arguments")
argrequired.add_argument(
    "-i", dest="PDB", required=True, type=str, help="native structure in PDB format"
)
argrequired.add_argument(
    "-c", dest="CHAIN", required=True, type=str, help="the chain of parsed protein"
)

argoptional = parser.add_argument_group("optional arguments")
argoptional.add_argument("-o", type=str, dest="OUT", default=0, help="the ouput file")

args = parser.parse_args()

pdbfile = args.PDB
chain = args.CHAIN
out = args.OUT

bindir = "./bin"


def logo():
    print(
        """\
``````_```_```````_`____`````````````````
`````|`|`|`|_`__`(_)``_`\``___```___`````
`````|`|`|`|`'_`\|`|`|`|`|/`_`\`/`__|````
`````|`|_|`|`|`|`|`|`|_|`|`(_)`|`(__`````
``````\___/|_|`|_|_|____/`\___/`\___|````
`````````````````````````````````````````"""
    )
    print(
        """\
*****************************************
***** UniDoc: Unified Domain Cutter *****
***(for structure-based domain parsing)**
*****************************************"""
    )


def caculate_ss(pdbfile, chain, bindir):
    binpath = os.path.join(bindir, "stride")
    assert os.path.exists(pdbfile)
    print(f"Running command: {'%s %s -r%s>pdb_ss'%(binpath,pdbfile,chain)}")
    return os.system("%s %s -r%s>pdb_ss" % (binpath, pdbfile, chain))


def parse_domain(pdbfile, chain, bindir, out):
    binpath = os.path.join(bindir, "UniDoc_structure")
    if out:
        print(f"Running command: {'%s %s %s pdb_ss > %s'%(binpath,pdbfile,chain,out)}")
        return os.system("%s %s %s pdb_ss > %s" % (binpath, pdbfile, chain, out))
    else:
        print(f"Running command: {'%s %s %s pdb_ss'%(binpath,pdbfile,chain)}")
        return os.system("%s %s %s pdb_ss" % (binpath, pdbfile, chain))


def main():
    logo()
    run_code = 0
    print("reading input files...")
    ## step 1: caculate the secondary structure with STRIDE
    print("step 1: caculate the secondary structure with STRIDE")
    if run_code == 0:
        run_code += caculate_ss(pdbfile, chain, bindir)

    ## step 2: parse domain
    print("step 2: parse domain")
    if run_code == 0:
        run_code += parse_domain(pdbfile, chain, bindir, out)


if __name__ == "__main__":
    main()
