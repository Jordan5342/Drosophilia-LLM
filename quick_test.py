import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CRISPRDatabaseAnalyzer:
    """
    Analyze CRISPR/RNAi screening database to generate gene importance scores
    """
    
    def __init__(self, database_file):
        self.database_file = database_file
        self.df = None
        
    def load_data(self):
        """Load the CRISPR database"""
        try:
            # Try different separators
            for sep in ['\t', ',', '|']:
                try:
                    self.df = pd.read_csv(self.database_file, sep=sep)
                    if len(self.df.columns) > 10:  # Should have many columns
                        break
                except:
                    continue
            
            if self.df is None:
                raise ValueError("Could not parse the database file")
                
            print(f"Loaded {len(self.df)} constructs from database")
            print(f"Columns: {list(self.df.columns)}")
            return True
            
        except Exception as e:
            print(f"Error loading database: {e}")
            return False
    
    def analyze_database_structure(self):
        """Analyze what information is available in the database"""
        
        analysis = {
            "construct_info": {
                "IDON": "Construct identifier",
                "target_gene": "Target gene symbol/ID", 
                "left_primer": "PCR primer sequences",
                "right_primer": "PCR primer sequences",
                "template": "Template used for construct generation"
            },
            
            "targeting_quality": {
                "IR_sequence": "Inverted repeat sequence for RNAi",
                "IR_length": "Length of targeting sequence (longer = more specific)",
                "orientation": "Construct orientation in vector",
                "ON_target_19mers": "Number of perfect matches to target",
                "OFF_target_19mers": "Number of potential off-target sites",
                "Number_OFF_target_genes": "Genes with potential off-target effects"
            },
            
            "experimental_data": {
                "verification": "Whether construct was experimentally verified",
                "CAN_repeats": "Repeat sequences that might affect efficiency", 
                "Number_ON_target_genes": "Number of genes this construct targets",
                "Number_of_transgenic_lines": "How many fly lines were generated"
            }
        }
        
        return analysis
    
    def calculate_targeting_specificity_score(self):
        """
        Calculate specificity score based on on-target vs off-target ratio
        Higher score = more specific targeting
        """
        scores = []
        
        for _, row in self.df.iterrows():
            gene = row.get('target_gene', 'Unknown')
            
            # Get targeting metrics
            on_targets = row.get('ON_target_19mers', 1)
            off_targets = row.get('OFF_target_19mers', 0) 
            off_target_genes = row.get('Number_OFF_target_genes', 0)
            ir_length = row.get('IR_length', 200)
            
            # Calculate specificity score
            # Higher on-target, lower off-target = better
            if off_targets > 0:
                specificity = on_targets / (off_targets + 1)
            else:
                specificity = on_targets
                
            # Bonus for longer IR sequences (more specific)
            length_bonus = min(ir_length / 500, 1.0)
            
            # Penalty for off-target genes
            off_target_penalty = max(0, 1 - (off_target_genes * 0.1))
            
            final_score = specificity * length_bonus * off_target_penalty
            
            scores.append({
                'Gene': gene,
                'Score': round(min(final_score, 1.0), 6)
            })
            
        return pd.DataFrame(scores)
    
    def calculate_experimental_confidence_score(self):
        """
        Calculate confidence score based on experimental validation
        """
        scores = []
        
        for _, row in self.df.iterrows():
            gene = row.get('target_gene', 'Unknown')
            
            score = 0.1  # Base score
            
            # Verified constructs get higher scores
            if row.get('verification', '').lower() in ['yes', 'true', '1', 'verified']:
                score += 0.5
            
            # Multiple transgenic lines = more reliable
            transgenic_lines = row.get('Number_of_transgenic_lines', 0)
            if transgenic_lines > 0:
                score += min(transgenic_lines * 0.1, 0.4)
            
            # Fewer CAN repeats = better construct quality
            can_repeats = row.get('CAN_repeats', 0)
            if can_repeats == 0:
                score += 0.2
            elif can_repeats < 3:
                score += 0.1
            
            scores.append({
                'Gene': gene,
                'Score': round(min(score, 1.0), 6)
            })
            
        return pd.DataFrame(scores)
    
    def calculate_targeting_efficiency_score(self):
        """
        Calculate efficiency score based on construct design quality
        """
        scores = []
        
        for _, row in self.df.iterrows():
            gene = row.get('target_gene', 'Unknown')
            
            score = 0.2  # Base score
            
            # IR length affects efficiency
            ir_length = row.get('IR_length', 200)
            if 300 <= ir_length <= 700:  # Optimal range
                score += 0.4
            elif 200 <= ir_length < 300 or 700 < ir_length <= 1000:
                score += 0.2
            
            # Single target gene is better than multiple
            on_target_genes = row.get('Number_ON_target_genes', 1)
            if on_target_genes == 1:
                score += 0.3
            elif on_target_genes <= 3:
                score += 0.1
            
            # Proper orientation
            orientation = row.get('orientation', '').lower()
            if 'forward' in orientation or 'correct' in orientation:
                score += 0.1
            
            scores.append({
                'Gene': gene,
                'Score': round(min(score, 1.0), 6)
            })
            
        return pd.DataFrame(scores)
    
    def generate_composite_score(self):
        """
        Generate composite score combining multiple metrics
        """
        specificity_df = self.calculate_targeting_specificity_score()
        confidence_df = self.calculate_experimental_confidence_score()
        efficiency_df = self.calculate_targeting_efficiency_score()
        
        # Merge all scores
        composite = specificity_df.copy()
        composite = composite.merge(confidence_df, on='Gene', suffixes=('_specificity', '_confidence'))
        composite = composite.merge(efficiency_df, on='Gene')
        composite.columns = ['Gene', 'Specificity', 'Confidence', 'Efficiency']
        
        # Calculate weighted composite score
        composite['Score'] = (
            composite['Specificity'] * 0.4 +
            composite['Confidence'] * 0.4 + 
            composite['Efficiency'] * 0.2
        ).round(6)
        
        return composite[['Gene', 'Score']]
    
    def analyze_gene_coverage(self):
        """Analyze which genes have the best experimental coverage"""
        
        gene_stats = self.df.groupby('target_gene').agg({
            'IDON': 'count',  # Number of constructs per gene
            'Number_of_transgenic_lines': 'sum',  # Total lines per gene
            'verification': lambda x: sum(1 for v in x if str(v).lower() in ['yes', 'true', '1', 'verified']),
            'OFF_target_19mers': 'mean'  # Average off-targets
        }).reset_index()
        
        gene_stats.columns = ['Gene', 'NumConstructs', 'TotalLines', 'VerifiedConstructs', 'AvgOffTargets']
        
        # Calculate coverage score
        gene_stats['CoverageScore'] = (
            np.log1p(gene_stats['NumConstructs']) * 0.3 +
            np.log1p(gene_stats['TotalLines']) * 0.4 +
            (gene_stats['VerifiedConstructs'] / gene_stats['NumConstructs']) * 0.3
        )
        
        gene_stats['CoverageScore'] = (gene_stats['CoverageScore'] / gene_stats['CoverageScore'].max()).round(6)
        
        return gene_stats[['Gene', 'CoverageScore']].rename(columns={'CoverageScore': 'Score'})
    
    def create_quality_filters(self):
        """Create filtered datasets based on quality criteria"""
        
        filters = {
            'high_specificity': (
                (self.df['OFF_target_19mers'] <= 5) & 
                (self.df['Number_OFF_target_genes'] <= 2)
            ),
            'experimentally_verified': (
                self.df['verification'].str.lower().isin(['yes', 'true', '1', 'verified'])
            ),
            'multiple_lines': (
                self.df['Number_of_transgenic_lines'] >= 2
            ),
            'optimal_length': (
                (self.df['IR_length'] >= 300) & (self.df['IR_length'] <= 700)
            )
        }
        
        filtered_datasets = {}
        for name, condition in filters.items():
            filtered_datasets[name] = self.df[condition]['target_gene'].unique()
            
        return filtered_datasets

def generate_sample_crispr_data():
    """Generate sample CRISPR database for demonstration"""
    
    genes = ['white', 'yellow', 'Notch', 'Delta', 'wingless', 'hedgehog', 'engrailed', 
             'Act5C', 'tub', 'Rpl32', 'even-skipped', 'fushi-tarazu', 'giant', 'hunchback']
    
    data = []
    for i, gene in enumerate(genes * 3):  # Multiple constructs per gene
        construct_id = f"IDON_{i+1:04d}"
        
        data.append({
            'IDON': construct_id,
            'target_gene': gene,
            'left_primer': f"ATGC{''.join(np.random.choice(['A','T','G','C'], 20))}",
            'right_primer': f"CGTA{''.join(np.random.choice(['A','T','G','C'], 20))}",
            'IR_sequence': ''.join(np.random.choice(['A','T','G','C'], 400)),
            'IR_length': np.random.randint(200, 800),
            'orientation': np.random.choice(['forward', 'reverse']),
            'verification': np.random.choice(['yes', 'no'], p=[0.7, 0.3]),
            'CAN_repeats': np.random.randint(0, 5),
            'Number_ON_target_genes': np.random.choice([1, 1, 1, 2, 3], p=[0.6, 0.2, 0.1, 0.07, 0.03]),
            'ON_target_19mers': np.random.randint(1, 20),
            'OFF_target_19mers': np.random.randint(0, 50),
            'Number_OFF_target_genes': np.random.randint(0, 10),
            'Number_of_transgenic_lines': np.random.randint(0, 8)
        })
    
    return pd.DataFrame(data)

def main():
    print("CRISPR/RNAi Database Gene Score Generator")
    print("=" * 50)
    
    # For demonstration, use sample data
    print("Generating sample CRISPR database...")
    sample_df = generate_sample_crispr_data()
    sample_df.to_csv("sample_crispr_database.csv", index=False)
    
    # Initialize analyzer
    analyzer = CRISPRDatabaseAnalyzer("/Users/jordansztejman/Downloads/Sup table 2.csv")
    analyzer.df = sample_df
    
    # Generate different types of scores
    print("\nGenerating gene scores based on different criteria:")
    
    # 1. Targeting specificity
    specificity_scores = analyzer.calculate_targeting_specificity_score()
    specificity_scores.to_csv("targeting_specificity_scores.csv", index=False)
    print(f"1. Targeting specificity scores: {len(specificity_scores)} genes")
    
    # 2. Experimental confidence  
    confidence_scores = analyzer.calculate_experimental_confidence_score()
    confidence_scores.to_csv("experimental_confidence_scores.csv", index=False)
    print(f"2. Experimental confidence scores: {len(confidence_scores)} genes")
    
    # 3. Targeting efficiency
    efficiency_scores = analyzer.calculate_targeting_efficiency_score()
    efficiency_scores.to_csv("targeting_efficiency_scores.csv", index=False)
    print(f"3. Targeting efficiency scores: {len(efficiency_scores)} genes")
    
    # 4. Gene coverage analysis
    coverage_scores = analyzer.analyze_gene_coverage()
    coverage_scores.to_csv("gene_coverage_scores.csv", index=False)
    print(f"4. Gene coverage scores: {len(coverage_scores)} genes")
    
    # 5. Composite score
    composite_scores = analyzer.generate_composite_score()
    composite_scores.to_csv("composite_crispr_scores.csv", index=False)
    print(f"5. Composite scores: {len(composite_scores)} genes")
    
    # Show sample results
    print(f"\nSample composite scores:")
    print(composite_scores.head(10).to_string(index=False))
    
    print(f"\nScore statistics:")
    print(f"Range: {composite_scores['Score'].min():.3f} to {composite_scores['Score'].max():.3f}")
    print(f"Mean: {composite_scores['Score'].mean():.3f}")

if __name__ == "__main__":
    main()
def main_screen_data():
    # Path where your composite score CSV is saved
    composite_csv_path = Path('composite_crispr_scores.csv')

    # Ensure the CSV exists
    if not composite_csv_path.exists():
        print(f"ERROR: {composite_csv_path} not found.")
        return

    # Initialize ScreenData with your composite scores CSV
    sd = ScreenData(
        file_path=str(composite_csv_path),
        id_col='Gene',
        val_col='Score',
        bio_taskname='CRISPR_COMPOSITE',
        base_dir=Path('.'),
        save=True
    )

    print(f"Loaded data with {len(sd.data_df)} genes.")

    # Identify top 5% hits
    sd.identify_hits(type_='gaussian', top_ratio_threshold=0.05, save=True)
    print(f"Identified {len(sd.topmovers)} top movers (hits).")

    # Write task prompt json file
    task_description = "Identify key genes based on composite CRISPR targeting scores."
    measurement = "Composite score combining specificity, confidence, and efficiency."

    sd.set_task_prompt(task_description, measurement)
    print(f"Saved task prompt for {sd.bio_taskname}.")

if __name__ == "__main__":
    main_screen_data()
