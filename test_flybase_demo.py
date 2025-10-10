from flybase_tool import FlyBaseTool
import random

print("="*60)
print("FLYBASE TOOL DEMONSTRATION - RANDOM GENE TESTING")
print("="*60)

tool = FlyBaseTool()

# Verified Drosophila genes that exist in FlyBase
# Using proper FlyBase nomenclature
verified_genes = [
    "Stat92E",  # JAK/STAT pathway
    "hop",      # JAK kinase
    "dome",     # cytokine receptor
    "upd",      # cytokine
    "Toll",     # immune receptor
    "spz",      # Toll ligand
    "Myd88",    # adaptor protein
    "pelle",    # kinase
    "cact",     # IκB homolog
    "dl",       # NFκB homolog
    "Dif",      # NFκB homolog
    "imd",      # immune deficiency
    "Tak1",     # kinase
    "w",        # white eye color
    "N",        # Notch
    "Dl",       # Delta
    "dpp",      # decapentaplegic
    "hh",       # hedgehog
    "wg",       # wingless
    "arm",      # armadillo
    "en",       # engraved
    "eve",      # even-skipped
    "ftz",      # fushi tarazu
    "bsk",      # basket (JNK)
    "hep",      # hemipterous
    "kay",      # kayak
    "Jra",      # Jun-related antigen
]

# Randomly select genes
num_tests = 5
random_genes = random.sample(verified_genes, num_tests)

print(f"\nRandomly testing {num_tests} verified FlyBase genes:")
print(", ".join(random_genes))
print()

for gene in random_genes:
    print(f"\n{'='*60}")
    print(f"Random gene: {gene}")
    print(f"{'='*60}")
    result = tool(f"gene_summary:{gene}")
    print(result)

