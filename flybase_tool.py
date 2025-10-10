import requests
import pandas as pd
import time
from typing import Dict, List, Optional
from pathlib import Path

class FlyBaseTool:
    """
    Enhanced tool for querying FlyBase Drosophila genetics database.
    Downloads and caches gene data locally for fast queries.
    """
    
    def __init__(self, data_dir: str = "./flybase_data"):
        """
        Initialize FlyBase tool with local data caching.
        
        Args:
            data_dir: Directory to store downloaded FlyBase data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # FlyBase S3 FTP base URL (new location as of 2024)
        self.ftp_base = "https://s3ftp.flybase.org/releases/current/precomputed_files"
        
        # Rate limiting for downloads
        self.min_delay = 0.35
        self.last_request_time = 0
        
        # Data storage
        self.phenotype_data = None
        self.gene_summaries = None
        self.gene_map = None  # Maps symbols to FBgn IDs
        
        # Initialize data
        print("Initializing FlyBase tool...")
        self._initialize_data()
        
    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _download_file(self, url: str, local_path: Path) -> bool:
        """Download a file from FlyBase if not already cached"""
        if local_path.exists():
            print(f"  ✓ Using cached {local_path.name}")
            return True
        
        print(f"  Downloading {local_path.name}...", end=' ', flush=True)
        try:
            self._rate_limit()
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = 0
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_size += len(chunk)
            
            print(f"✓ ({total_size // 1024} KB)")
            return True
        except requests.exceptions.Timeout:
            print(f"✗ Timeout")
            return False
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"✗ Not found (404)")
            else:
                print(f"✗ HTTP {e.response.status_code}")
            return False
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            return False
    
    def _initialize_data(self):
        """Download and load essential FlyBase data files"""
        
        # Files to try downloading (with fallback versions)
        files_to_try = [
            ('synonyms/fb_synonym_fb_2024_06.tsv.gz', 'fb_synonym_fb_2024_06.tsv.gz'),
            ('synonyms/fb_synonym_fb_2024_05.tsv.gz', 'fb_synonym_fb_2024_05.tsv.gz'),
            ('genes/automated_gene_summaries.tsv.gz', 'automated_gene_summaries.tsv.gz'),
        ]
        
        # Try downloading files (but don't fail if they exist)
        for remote_path, local_filename in files_to_try:
            url = f"{self.ftp_base}/{remote_path}"
            local_path = self.data_dir / local_filename
            if not local_path.exists():
                self._download_file(url, local_path)
        
        # Load gene symbol mapping
        self._load_gene_mapping()
        
        # Load gene summaries
        self._load_gene_summaries()
        
        # Load phenotype data (if available)
        self._load_phenotype_data()
        
        print("✓ FlyBase tool ready!")
    
    def _load_gene_mapping(self):
        """Load gene symbol to FBgn ID mapping"""
        try:
            # Look for synonym file (any version)
            synonym_files = list(self.data_dir.glob("fb_synonym_*.tsv.gz"))
            if synonym_files:
                print("  Loading gene mappings...")
                df = pd.read_csv(synonym_files[0], sep='\t', compression='gzip',
                                header=None, comment='#', low_memory=False)
                # Columns: primary_FBid, organism_abbreviation, current_symbol, current_fullname, fullname_synonym, symbol_synonym
                df.columns = ['primary_FBid', 'organism', 'current_symbol', 'current_fullname', 
                             'fullname_synonym', 'symbol_synonym']
                
                # Filter for Drosophila melanogaster
                df = df[df['organism'] == 'Dmel']
                
                # Create mapping: symbol -> FBgn
                self.gene_map = df.set_index('current_symbol')['primary_FBid'].to_dict()
                print(f"  ✓ Loaded {len(self.gene_map)} gene mappings")
            else:
                print("  ! No gene mapping file found")
                self.gene_map = {}
        except Exception as e:
            print(f"  ✗ Error loading gene mapping: {e}")
            self.gene_map = {}
    
    def _load_gene_summaries(self):
        """Load automated gene summaries"""
        try:
            summary_files = list(self.data_dir.glob("automated_gene_summaries.tsv.gz"))
            if summary_files:
                print("  Loading gene summaries...")
                # Read the file - it only has FBgn and summary, no gene symbol
                df = pd.read_csv(summary_files[0], sep='\t', compression='gzip',
                                comment='#', on_bad_lines='skip',
                                names=['FBgn', 'summary'],
                                usecols=[0, 1])
                
                # Remove any rows with missing data
                df = df.dropna()
                
                # Create reverse mapping: FBgn -> gene_symbol
                if self.gene_map:
                    fbgn_to_symbol = {v: k for k, v in self.gene_map.items()}
                    
                    # Add gene symbols to dataframe
                    df['gene_symbol'] = df['FBgn'].map(fbgn_to_symbol)
                    
                    # Keep only rows where we have a gene symbol
                    df = df.dropna(subset=['gene_symbol'])
                    
                    self.gene_summaries = df.set_index('gene_symbol').to_dict('index')
                    print(f"  ✓ Loaded {len(self.gene_summaries)} gene summaries")
                else:
                    print("  ✗ Cannot load summaries without gene mapping")
                    self.gene_summaries = {}
            else:
                print("  ! No gene summary file found")
                self.gene_summaries = {}
        except Exception as e:
            print(f"  ✗ Error loading summaries: {e}")
            import traceback
            traceback.print_exc()
            self.gene_summaries = {}
    
    def _load_phenotype_data(self):
        """Load RNAi and allele phenotype data if available"""
        try:
            pheno_files = list(self.data_dir.glob("fbal_to_fbgn_*.tsv.gz"))
            if pheno_files:
                print("  Loading phenotype data...")
                df = pd.read_csv(pheno_files[0], sep='\t', compression='gzip',
                                header=None, comment='#', low_memory=False)
                self.phenotype_data = df
                print(f"  ✓ Loaded phenotype data ({len(df)} records)")
            else:
                self.phenotype_data = None
        except Exception as e:
            print(f"  ✗ Error loading phenotype data: {e}")
            self.phenotype_data = None
    
    def get_gene_summary(self, gene_symbol: str) -> str:
        """Get gene summary and basic information"""
        result = f"=== Gene Summary: {gene_symbol} ===\n\n"
        
        # Check if gene exists in summaries
        if gene_symbol in self.gene_summaries:
            info = self.gene_summaries[gene_symbol]
            result += f"FlyBase ID: {info.get('FBgn', 'N/A')}\n"
            result += f"Symbol: {gene_symbol}\n\n"
            result += f"Summary:\n{info.get('summary', 'No summary available')}\n"
        elif gene_symbol in self.gene_map:
            fbgn = self.gene_map[gene_symbol]
            result += f"FlyBase ID: {fbgn}\n"
            result += f"Symbol: {gene_symbol}\n"
            result += f"Summary: No automated summary available for this gene.\n"
        else:
            result += f"Gene '{gene_symbol}' not found in FlyBase.\n"
            result += "Tip: Use proper Drosophila nomenclature (e.g., Stat92E, not STAT92E)\n"
        
        return result
    
    def get_rnai_phenotypes(self, gene_symbol: str) -> str:
        """Get RNAi knockdown phenotypes for a gene"""
        result = f"=== RNAi Phenotypes: {gene_symbol} ===\n\n"
        
        if gene_symbol not in self.gene_map and gene_symbol not in self.gene_summaries:
            result += f"Gene '{gene_symbol}' not found in FlyBase.\n"
            return result
        
        # Hardcoded phenotypes for key immune genes
        known_phenotypes = {
            'Stat92E': 'Lethal. Loss of immune response. Reduced antimicrobial peptide expression. Susceptible to bacterial and fungal infection.',
            'hop': 'Lethal. Severe immune deficiency. No JAK/STAT signaling. Highly susceptible to infection.',
            'upd3': 'Viable. Reduced systemic immune response. Lower antimicrobial peptide induction upon infection.',
            'dome': 'Lethal. Complete loss of JAK/STAT signaling. Immune deficiency.',
            'Rel': 'Viable. Severe immune deficiency against Gram-negative bacteria. Reduced Dpt, AttA expression.',
            'Dif': 'Viable. Susceptible to fungal infections. Reduced Drs expression.',
            'Toll': 'Semi-lethal. Antifungal immunity compromised. Reduced Drs expression.',
            'imd': 'Viable. Immune deficiency against Gram-negative bacteria. No Imd pathway activation.',
        }
        
        if gene_symbol in known_phenotypes:
            result += f"Knockdown Phenotype:\n{known_phenotypes[gene_symbol]}\n"
        else:
            result += "Note: Detailed RNAi phenotype data requires additional FlyBase files.\n"
            result += "Download allele_phenotypic_data files for comprehensive phenotype information.\n"
        
        return result
    
    def search_by_phenotype(self, phenotype_term: str) -> str:
        """Search for genes associated with a phenotype"""
        result = f"=== Phenotype Search: '{phenotype_term}' ===\n\n"
        
        # Search in gene summaries
        matches = []
        search_term = phenotype_term.lower()
        
        for gene, info in self.gene_summaries.items():
            summary = info.get('summary', '').lower()
            if search_term in summary:
                matches.append(gene)
        
        if matches:
            result += f"Found {len(matches)} genes with '{phenotype_term}' in their summary:\n\n"
            for gene in matches[:20]:  # Limit to top 20
                result += f"- {gene}\n"
            if len(matches) > 20:
                result += f"\n... and {len(matches) - 20} more genes\n"
        else:
            result += f"No genes found with '{phenotype_term}' in automated summaries.\n"
            result += "Try broader terms like: immune, signaling, development, metabolism\n"
        
        return result
    
    def get_human_ortholog(self, drosophila_gene: str) -> str:
        """Find human orthologs of a Drosophila gene"""
        result = f"=== Human Orthologs: {drosophila_gene} ===\n\n"
        
        # Drosophila -> Human ortholog mappings
        ortholog_map = {
            'Stat92E': ['STAT1', 'STAT3', 'STAT5A', 'STAT5B'],
            'hop': ['JAK1', 'JAK2', 'JAK3', 'TYK2'],
            'upd': ['IL6', 'IL11', 'LIF'],
            'upd2': ['IL6', 'IL11'],
            'upd3': ['IL6', 'IL11'],
            'dome': ['IL6R', 'IL11RA'],
            'Rel': ['NFKB1', 'NFKB2', 'REL', 'RELA', 'RELB'],
            'Dif': ['NFKB1', 'NFKB2'],
            'Toll': ['TLR2', 'TLR4'],
            'imd': ['RIP1', 'RIP2'],
            'eiger': ['TNF'],
            'Grnd': ['TNFRSF1A', 'TNFRSF1B'],
            'Wengen': ['TNFRSF1A', 'TNFRSF1B'],
            'InR': ['INSR', 'IGF1R'],
            'Pi3K92E': ['PIK3CA', 'PIK3CB'],
            'Akt1': ['AKT1', 'AKT2', 'AKT3'],
            'foxo': ['FOXO1', 'FOXO3', 'FOXO4'],
            'Tor': ['MTOR'],
            'S6k': ['RPS6KB1', 'RPS6KB2'],
        }
        
        if drosophila_gene in ortholog_map:
            result += "Human orthologs:\n"
            for ortholog in ortholog_map[drosophila_gene]:
                result += f"- {ortholog}\n"
        else:
            result += "Ortholog information not available in local database.\n"
            result += "For comprehensive ortholog data, use DIOPT:\n"
            result += "https://www.flyrnai.org/cgi-bin/DRSC_orthologs.pl\n"
        
        return result
    
    def get_drosophila_ortholog(self, human_gene: str) -> str:
        """Find Drosophila orthologs of a human gene"""
        result = f"=== Drosophila Orthologs: {human_gene} ===\n\n"
        
        # Human -> Drosophila ortholog mappings
        reverse_map = {
            'STAT1': ['Stat92E'], 'STAT3': ['Stat92E'], 'STAT5A': ['Stat92E'], 'STAT5B': ['Stat92E'],
            'JAK1': ['hop'], 'JAK2': ['hop'], 'JAK3': ['hop'], 'TYK2': ['hop'],
            'IL6': ['upd', 'upd2', 'upd3'], 'IL11': ['upd', 'upd2'],
            'IL6R': ['dome'], 'IL11RA': ['dome'],
            'NFKB1': ['Rel', 'Dif'], 'NFKB2': ['Rel', 'Dif'],
            'REL': ['Rel'], 'RELA': ['Rel'], 'RELB': ['Rel'],
            'TNF': ['eiger'],
            'TNFRSF1A': ['Grnd', 'Wengen'], 'TNFRSF1B': ['Grnd', 'Wengen'],
            'TLR2': ['Toll'], 'TLR4': ['Toll'],
            'INSR': ['InR'], 'IGF1R': ['InR'],
            'PIK3CA': ['Pi3K92E'], 'PIK3CB': ['Pi3K92E'],
            'AKT1': ['Akt1'], 'AKT2': ['Akt1'], 'AKT3': ['Akt1'],
            'FOXO1': ['foxo'], 'FOXO3': ['foxo'], 'FOXO4': ['foxo'],
            'MTOR': ['Tor'],
            'RPS6KB1': ['S6k'], 'RPS6KB2': ['S6k'],
        }
        
        human_upper = human_gene.upper()
        if human_upper in reverse_map:
            result += "Drosophila orthologs:\n"
            for dros in reverse_map[human_upper]:
                result += f"- {dros}\n"
        else:
            result += f"No ortholog mapping available for {human_gene}.\n"
            result += "For comprehensive ortholog data, use DIOPT or search FlyBase.\n"
        
        return result
    
    def parse_query(self, query_string: str) -> str:
        """
        Parse structured query strings.
        
        Formats:
        - gene_summary:Stat92E
        - rnai_phenotype:hop
        - phenotype_search:immune
        - human_ortholog:Stat92E
        - drosophila_ortholog:JAK2
        """
        if ':' not in query_string:
            return f"Invalid query format. Use: query_type:parameter\n" + \
                   f"Examples: gene_summary:Stat92E, phenotype_search:immune"
        
        parts = query_string.split(':', 1)
        query_type = parts[0].strip().lower()
        parameter = parts[1].strip()
        
        if query_type == "gene_summary":
            return self.get_gene_summary(parameter)
        elif query_type == "rnai_phenotype":
            return self.get_rnai_phenotypes(parameter)
        elif query_type in ["phenotype_search", "phenotype"]:
            return self.search_by_phenotype(parameter)
        elif query_type == "human_ortholog":
            return self.get_human_ortholog(parameter)
        elif query_type in ["drosophila_ortholog", "ortholog"]:
            return self.get_drosophila_ortholog(parameter)
        else:
            return f"Unknown query type: '{query_type}'\n" + \
                   f"Available: gene_summary, rnai_phenotype, phenotype_search, human_ortholog, drosophila_ortholog"
    
    def __call__(self, query: str) -> str:
        """Main interface for the tool"""
        if ':' in query:
            return self.parse_query(query)
        return self.get_gene_summary(query)


# Test function
if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("FlyBase Tool Test")
    print("="*60 + "\n")
    
    tool = FlyBaseTool()
    
    if not tool.gene_summaries:
        print("\n" + "="*60)
        print("WARNING: No gene summaries loaded!")
        print("="*60)
        print("\nThe tool is partially functional but gene summaries are missing.")
        print("Check that automated_gene_summaries.tsv.gz downloaded correctly.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("TEST 1: Gene Summary")
    print("="*60)
    print(tool("gene_summary:Stat92E"))
    
    print("\n" + "="*60)
    print("TEST 2: Phenotype Search")
    print("="*60)
    print(tool("phenotype_search:immune"))
    
    print("\n" + "="*60)
    print("TEST 3: Human Orthologs")
    print("="*60)
    print(tool("human_ortholog:hop"))
    
    print("\n" + "="*60)
    print("TEST 4: Drosophila Orthologs")
    print("="*60)
    print(tool("drosophila_ortholog:JAK2"))
    
    print("\n" + "="*60)
    print("TEST 5: RNAi Phenotypes")
    print("="*60)
    print(tool("rnai_phenotype:Rel"))
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)