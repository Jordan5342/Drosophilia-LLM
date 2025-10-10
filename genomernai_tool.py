import requests
import pandas as pd
import os
import gzip
import shutil
from typing import Optional

class GenomeRNAiTool:
    """
    Tool for querying GenomeRNAi Drosophila screen data.
    Downloads bulk data once and caches locally for fast queries.
    """
    
    def __init__(self, cache_dir="./data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # GenomeRNAi provides downloadable datasets
        # Check http://www.genomernai.org/v4/download/ for current files
        self.data_urls = {
            # Main phenotype scores file (check actual URL on site)
            "screens": "http://www.genomernai.org/dump/all_screens.txt.gz",
            # Alternative: individual screen files
        }
        
        self.data_file = os.path.join(cache_dir, "genomernai_screens.csv")
        self.data = None
        
    def download_and_extract(self, url: str, output_file: str) -> bool:
        """Download gzipped file and extract"""
        try:
            print(f"Downloading from {url}...")
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Download to temp gzip file
            gz_file = output_file + ".gz"
            with open(gz_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract gzip
            print("Extracting data...")
            with gzip.open(gz_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Clean up gz file
            os.remove(gz_file)
            
            print(f"✓ Data downloaded and extracted to {output_file}")
            return True
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def load_data(self, force_download: bool = False) -> bool:
        """Load data from cache or download if needed"""
        
        # Check if cached data exists
        if os.path.exists(self.data_file) and not force_download:
            print(f"Loading cached data from {self.data_file}...")
            try:
                # Try reading as CSV first
                self.data = pd.read_csv(self.data_file, low_memory=False)
                print(f"✓ Loaded {len(self.data)} records")
                print(f"Columns: {list(self.data.columns)[:5]}...")  # Show first 5 columns
                return True
            except Exception as e:
                print(f"Error loading as CSV: {e}")
                # Try as tab-separated
                try:
                    self.data = pd.read_csv(self.data_file, sep='\t', low_memory=False)
                    print(f"✓ Loaded {len(self.data)} records")
                    return True
                except Exception as e2:
                    print(f"✗ Error loading cached data: {e2}")
                    print("Attempting to re-download...")
        
        # Download data
        if not self.download_csv_directly():
            return False
        
        # Load newly downloaded data
        try:
            self.data = pd.read_csv(self.data_file, low_memory=False)
            print(f"✓ Loaded {len(self.data)} records from downloaded file")
            print(f"Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"✗ Error loading downloaded data: {e}")
            return False

    def download_csv_directly(self) -> bool:
        """Alternative: download CSV format directly"""
        csv_url = "http://www.genomernai.org/exportcsv/allscreens"
        
        try:
            print(f"Downloading CSV from {csv_url}...")
            response = requests.get(csv_url, timeout=60)
            response.raise_for_status()
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"✓ CSV downloaded to {self.data_file}")
            
            # Immediately load it
            try:
                self.data = pd.read_csv(self.data_file, low_memory=False)
                print(f"✓ Loaded {len(self.data)} records")
                print(f"Available columns: {list(self.data.columns)}")
                return True
            except Exception as e:
                print(f"✗ Error parsing CSV: {e}")
                # Show first few lines to debug
                with open(self.data_file, 'r') as f:
                    print("First 200 chars of file:")
                    print(f.read(200))
                return False
            
        except Exception as e:
            print(f"✗ CSV download failed: {e}")
            return False

    def get_gene_scores(self, gene_symbol: str) -> str:
        """Get all screen scores for a specific gene"""
        if self.data is None:
            if not self.load_data():
                return "Error: Could not load GenomeRNAi data. Check internet connection or download manually."
        
        # Try different possible column names for gene symbols
        gene_cols = ['gene_symbol', 'Gene Symbol', 'gene', 'Gene']
        gene_col = None
        for col in gene_cols:
            if col in self.data.columns:
                gene_col = col
                break
        
        if gene_col is None:
            return f"Error: Could not find gene symbol column. Available columns: {list(self.data.columns)}"
        
        # Filter for the gene
        gene_data = self.data[self.data[gene_col].str.upper() == gene_symbol.upper()]
        
        if len(gene_data) == 0:
            return f"Gene '{gene_symbol}' not found in GenomeRNAi database"
        
        # Format results
        result = f"GenomeRNAi knockdown data for {gene_symbol}:\n"
        result += f"Found in {len(gene_data)} screens\n\n"
        
        for idx, row in gene_data.head(10).iterrows():
            result += f"Screen: {row.get('screen_name', row.get('Screen', 'Unknown'))}\n"
            
            # Try different score column names
            score_val = row.get('score', row.get('Score', row.get('phenotype_score', 'N/A')))
            zscore_val = row.get('z_score', row.get('Z-score', row.get('zscore', 'N/A')))
            
            result += f"  Score: {score_val}\n"
            result += f"  Z-score: {zscore_val}\n"
            result += f"  Phenotype: {row.get('phenotype', row.get('Phenotype', 'N/A'))}\n\n"
        
        if len(gene_data) > 10:
            result += f"... and {len(gene_data) - 10} more screens\n"
        
        return result
    
    def search_top_hits(self, phenotype: str = "", top_n: int = 20, 
                        min_zscore: float = 2.0) -> str:
        """Find top gene hits from screens"""
        if self.data is None:
            if not self.load_data():
                return "Error: Could not load data"
        
        # Filter by phenotype if specified
        filtered_data = self.data
        if phenotype:
            pheno_col = 'phenotype' if 'phenotype' in self.data.columns else 'Phenotype'
            filtered_data = filtered_data[
                filtered_data[pheno_col].str.contains(phenotype, case=False, na=False)
            ]
        
        # Find z-score column
        zscore_col = None
        for col in ['z_score', 'Z-score', 'zscore', 'Zscore']:
            if col in filtered_data.columns:
                zscore_col = col
                break
        
        if zscore_col is None:
            return "Error: Could not find z-score column in data"
        
        # Filter by significance
        filtered_data = filtered_data[filtered_data[zscore_col].abs() >= min_zscore]
        
        # Get top hits
        top_hits = filtered_data.nlargest(top_n, zscore_col, keep='all')
        
        result = f"Top {top_n} hits"
        if phenotype:
            result += f" for phenotype '{phenotype}'"
        result += f" (|z-score| >= {min_zscore}):\n\n"
        
        gene_col = 'gene_symbol' if 'gene_symbol' in top_hits.columns else 'Gene Symbol'
        
        for idx, row in top_hits.iterrows():
            result += f"{row[gene_col]}: z-score = {row[zscore_col]:.2f}\n"
        
        return result
    
    def __call__(self, query: str) -> str:
        """
        Main interface. Accepts queries like:
        - "gene_scores:Stat92E"
        - "top_hits:viability:20"
        - "Stat92E" (defaults to gene_scores)
        """
        if ':' in query:
            parts = query.split(':')
            query_type = parts[0].strip()
            
            if query_type == "gene_scores" and len(parts) >= 2:
                return self.get_gene_scores(parts[1].strip())
            
            elif query_type == "top_hits":
                phenotype = parts[1].strip() if len(parts) > 1 else ""
                top_n = int(parts[2]) if len(parts) > 2 else 20
                return self.search_top_hits(phenotype, top_n)
        
        # Default: treat as gene name
        return self.get_gene_scores(query)


# Test
if __name__ == "__main__":
    tool = GenomeRNAiTool()
    
    # Force load first
    if tool.load_data():
        print("Testing GenomeRNAi tool:")
        print("=" * 60)
        print(tool("gene_scores:Stat92E"))
        print("=" * 60)
    else:
        print("Failed to load data")