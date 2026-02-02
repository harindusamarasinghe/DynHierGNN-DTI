# scripts/build_hierarchical_proteins.py

"""
Hierarchical Protein Graph Builder - Multi-Scale Structure-Aware

Inspired by:
- HIGH-PPI: Hierarchical Graph Learning for Protein-Protein Interaction (Nature Comms 2023)
- ProNet: Hierarchical Protein Representations (NeurIPS 2022)
- HoloProt: Multi-Scale Representation Learning (NeurIPS 2021)
- MGPPI: Multiscale Graph Neural Networks (Frontiers 2024)

Three levels:
1. Residue-level: Amino acids with ESM-2 embeddings + spatial edges
2. Secondary Structure-level: Alpha-helices, beta-sheets, loops (SSE elements)
3. Domain-level: Functional domains (N-lobe, C-lobe for kinases)

Key innovations:
- Use DSSP for accurate secondary structure annotation
- Spatial clustering (DBSCAN) for domain detection
- Multi-scale edge construction (sequential + spatial)
- ESM-2 embeddings for rich residue features
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, DSSP
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
# At the very top of the file
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MultiScaleProteinBuilder:
    """
    Build 3-level protein hierarchy with multi-scale structure awareness
    Following HIGH-PPI, ProNet, and MGPPI principles
    """
    
    def __init__(self, esm2_embeddings_dir=None):
        self.parser = PDBParser(QUIET=True)
        if esm2_embeddings_dir is None:
            self.esm2_embeddings_dir = PROJECT_ROOT / 'embeddings' / 'all_proteins'
        else:
            self.esm2_embeddings_dir = Path(esm2_embeddings_dir)        
        # Amino acid one-hot encoding
        self.aa_types = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
        
        # Secondary structure types (DSSP simplified)
        self.ss_types = ['H', 'E', 'L']  # Helix, Sheet, Loop
    
    def load_esm2_embeddings(self, protein_id):
        """Load pre-computed ESM-2 embeddings"""
        emb_path = self.esm2_embeddings_dir / f"{protein_id}_esm2.npy"
        
        if emb_path.exists():
            embeddings = np.load(emb_path)
            return torch.tensor(embeddings, dtype=torch.float)
        else:
            print(f"      ⚠️  ESM-2 embeddings not found for {protein_id}, using zeros")
            return None
    
    def get_residue_features(self, residue, aa_type, esm2_emb=None):
        """
        Build residue node features
        Following HIGH-PPI and ProNet design
        """
        features = []
        
        # 1. Amino acid type (one-hot, 21-dim)
        aa_onehot = [1 if aa_type == aa else 0 for aa in self.aa_types]
        features.extend(aa_onehot)
        
        # 2. ESM-2 embeddings (1280-dim) - rich semantic features
        if esm2_emb is not None:
            features.extend(esm2_emb.tolist())
        else:
            features.extend([0.0] * 1280)
        
        # 3. Structural properties (if available)
        try:
            # B-factor (flexibility indicator)
            ca_atom = residue['CA']
            b_factor = ca_atom.get_bfactor()
            features.append(b_factor / 100.0)  # Normalize
        except:
            features.append(0.0)
        
        return features
    
    def build_level1_residue_graph(self, pdb_file, protein_id, esm2_embeddings=None):
        """
        Level 1: Residue graph
        Multi-scale edges: sequential + spatial (following HIGH-PPI, ProNet)
        """
        structure = self.parser.get_structure(protein_id, pdb_file)
        model = structure[0]
        chain = list(model.get_chains())[0]
        residues = [res for res in chain.get_residues() if res.id[0] == ' ']
        
        num_residues = len(residues)
        
        if num_residues == 0:
            return None
        
        # Node features: AA type + ESM-2 + structural properties
        node_features = []
        ca_coords = []
        
        for i, residue in enumerate(residues):
            aa_type = residue.get_resname()
            # Convert 3-letter to 1-letter
            aa_1letter = self._three_to_one(aa_type)
            
            # Get ESM-2 embedding for this residue
            esm2_emb = esm2_embeddings[i] if esm2_embeddings is not None else None
            
            # Build features
            feat = self.get_residue_features(residue, aa_1letter, esm2_emb)
            node_features.append(feat)
            
            # Get CA coordinates
            try:
                ca = residue['CA'].get_coord()
                ca_coords.append(ca)
            except:
                ca_coords.append(np.array([0.0, 0.0, 0.0]))
        
        x = torch.tensor(node_features, dtype=torch.float)
        ca_coords = np.array(ca_coords)
        
        # Build multi-scale edges (HIGH-PPI approach)
        edge_indices = []
        edge_types = []  # 0: sequential, 1: spatial
        
        # Sequential edges (backbone connectivity)
        for i in range(num_residues - 1):
            edge_indices.append([i, i+1])
            edge_indices.append([i+1, i])
            edge_types.extend([0, 0])
        
        # Spatial edges (following ProNet: 8Å cutoff for residue graphs)
        dist_matrix = cdist(ca_coords, ca_coords)
        spatial_threshold = 8.0  # Angstroms
        
        for i in range(num_residues):
            for j in range(i+1, num_residues):
                if dist_matrix[i, j] <= spatial_threshold:
                    # Avoid duplicate sequential edges
                    if abs(i - j) > 1:
                        edge_indices.append([i, j])
                        edge_indices.append([j, i])
                        edge_types.extend([1, 1])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long) if edge_types else torch.empty(0, dtype=torch.long)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_residues,
            ca_coords=torch.tensor(ca_coords, dtype=torch.float)
        )

    def annotate_secondary_structure_dssp(self, pdb_file):
        """
        Use DSSP to annotate secondary structure
        """
        try:
            structure = self.parser.get_structure('protein', pdb_file)
            model = structure[0]

            # Run DSSP (mkdssp must be installed)
            dssp = DSSP(model, str(pdb_file), dssp='mkdssp')

            ss_annotations = {}
            residue_idx = 0

            # dssp is an ordered dict: key = (chain_id, res_id)
            # where res_id is a tuple (hetflag, seq_id, icode)
            for key in dssp.keys():
                # Do NOT try to unpack three values, just:
                chain_id, res_id = key      # res_id is a tuple, that's fine
                ss_code = dssp[key][2]

                if ss_code in ['H', 'G', 'I']:
                    ss_simple = 'H'
                elif ss_code in ['E', 'B']:
                    ss_simple = 'E'
                else:
                    ss_simple = 'L'

                ss_annotations[residue_idx] = ss_simple
                residue_idx += 1

            return ss_annotations

        except Exception as e:
            print(f"        DSSP annotation failed: {e}")
            # Fallback: all loops
            structure = self.parser.get_structure('protein', pdb_file)
            model = structure[0]
            chain = list(model.get_chains())[0]
            num_res = len([r for r in chain.get_residues() if r.id[0] == ' '])
            return {i: 'L' for i in range(num_res)}

    def build_level2_secondary_structure_graph(self, residue_graph, ss_annotations):
        """
        Level 2: Secondary structure graph
        Group consecutive residues with same SS, create SSE (Secondary Structure Element) nodes
        Following HIGH-PPI bottom-view design
        """
        num_residues = residue_graph.num_nodes
        
        # Group consecutive residues with same SS type
        sse_groups = []  # List of (ss_type, [residue_indices])
        current_group = []
        current_ss = None
        
        for res_idx in range(num_residues):
            ss = ss_annotations.get(res_idx, 'L')
            
            if ss == current_ss or current_ss is None:
                current_group.append(res_idx)
                current_ss = ss
            else:
                # Save current group, start new
                if current_group:
                    sse_groups.append((current_ss, current_group))
                current_group = [res_idx]
                current_ss = ss
        
        # Don't forget last group
        if current_group:
            sse_groups.append((current_ss, current_group))
        
        num_sses = len(sse_groups)
        
        # Pool residue features to SSE features (mean pooling + SS type)
        sse_features = []
        
        for ss_type, residue_indices in sse_groups:
            # Mean pooling of residue features
            res_feats = residue_graph.x[residue_indices]
            pooled_feat = res_feats.mean(dim=0)
            
            # Append SS type one-hot (3-dim: H, E, L)
            ss_onehot = [1 if ss_type == t else 0 for t in self.ss_types]
            
            # Append SSE length (normalized)
            sse_length = len(residue_indices) / num_residues
            
            # Concatenate
            sse_feat = torch.cat([
                pooled_feat,
                torch.tensor(ss_onehot, dtype=torch.float),
                torch.tensor([sse_length], dtype=torch.float)
            ])
            
            sse_features.append(sse_feat)
        
        sse_x = torch.stack(sse_features)
        
        # Build SSE graph edges
        # Strategy: connect adjacent SSEs + spatially close SSEs
        sse_edges = []
        ca_coords = residue_graph.ca_coords
        
        # Get centroid of each SSE
        sse_centroids = []
        for ss_type, res_indices in sse_groups:
            centroid = ca_coords[res_indices].mean(dim=0)
            sse_centroids.append(centroid)
        
        sse_centroids = torch.stack(sse_centroids).numpy()
        
        # Sequential edges (adjacent SSEs)
        for i in range(num_sses - 1):
            sse_edges.append([i, i+1])
            sse_edges.append([i+1, i])
        
        # Spatial edges (SSE centroids within 15Å)
        dist_matrix = cdist(sse_centroids, sse_centroids)
        spatial_threshold = 15.0
        
        for i in range(num_sses):
            for j in range(i+1, num_sses):
                if dist_matrix[i, j] <= spatial_threshold:
                    if abs(i - j) > 1:  # Not adjacent
                        sse_edges.append([i, j])
                        sse_edges.append([j, i])
        
        sse_edge_index = torch.tensor(sse_edges, dtype=torch.long).t().contiguous() if sse_edges else torch.empty((2, 0), dtype=torch.long)
        
        return Data(
            x=sse_x,
            edge_index=sse_edge_index,
            num_nodes=num_sses,
            sse_types=[ss for ss, _ in sse_groups],
            sse_groups=sse_groups
        )
    
    def detect_domains_spatial_clustering(self, sse_graph, ca_coords):
        """
        Level 3: Domain detection using spatial clustering
        Following HoloProt multi-scale approach
        
        For kinases: typically 2-3 domains (N-lobe, C-lobe, linker)
        """
        num_sses = sse_graph.num_nodes
        
        if num_sses < 2:
            # Single domain
            domain_labels = np.array([0])
        else:
            # Get SSE centroids
            sse_centroids = []
            for ss_type, res_indices in sse_graph.sse_groups:
                centroid = ca_coords[res_indices].mean(dim=0).numpy()
                sse_centroids.append(centroid)
            
            sse_centroids = np.array(sse_centroids)
            
            # DBSCAN clustering (eps=15Å, min_samples=1)
            clustering = DBSCAN(eps=15.0, min_samples=1).fit(sse_centroids)
            domain_labels = clustering.labels_
            
            # Ensure at least 1 domain
            if len(set(domain_labels)) == 0:
                domain_labels = np.array([0] * num_sses)
        
        num_domains = len(set(domain_labels))
        
        # Pool SSE features to domain features
        domain_features = []
        for domain_id in range(num_domains):
            domain_sse_indices = np.where(domain_labels == domain_id)[0]
            sse_feats = sse_graph.x[domain_sse_indices]
            domain_feat = sse_feats.mean(dim=0)
            domain_features.append(domain_feat)
        
        domain_x = torch.stack(domain_features)
        
        # Domain graph edges: fully connected (all domains interact)
        domain_edges = []
        for i in range(num_domains):
            for j in range(i+1, num_domains):
                domain_edges.append([i, j])
                domain_edges.append([j, i])
        
        domain_edge_index = torch.tensor(domain_edges, dtype=torch.long).t().contiguous() if domain_edges else torch.empty((2, 0), dtype=torch.long)
        
        return Data(
            x=domain_x,
            edge_index=domain_edge_index,
            num_nodes=num_domains
        )
    
    def build_hierarchical_protein_conformation(self, protein_id, conformation_idx, pdb_file):
        """
        Build 3-level hierarchy for one protein conformation
        """
        print(f"      Conf {conformation_idx}...", end=' ')
        
        # Load ESM-2 embeddings
        esm2_embeddings = self.load_esm2_embeddings(protein_id)
        
        # Level 1: Residue graph
        residue_graph = self.build_level1_residue_graph(pdb_file, protein_id, esm2_embeddings)
        
        if residue_graph is None:
            print("✗ Failed")
            return None
        
        # Annotate secondary structure
        ss_annotations = self.annotate_secondary_structure_dssp(pdb_file)
        
        # Level 2: Secondary structure graph
        sse_graph = self.build_level2_secondary_structure_graph(residue_graph, ss_annotations)
        
        # Level 3: Domain graph
        domain_graph = self.detect_domains_spatial_clustering(sse_graph, residue_graph.ca_coords)
        
        print(f"✓ R:{residue_graph.num_nodes} SSE:{sse_graph.num_nodes} D:{domain_graph.num_nodes}")
        
        return {
            'protein_id': protein_id,
            'conformation_idx': conformation_idx,
            'level1_residues': residue_graph,
            'level2_secondary_structures': sse_graph,
            'level3_domains': domain_graph,
            'ss_annotations': ss_annotations,
        }
    
    def _three_to_one(self, three_letter):
        """Convert 3-letter amino acid code to 1-letter"""
        aa_dict = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        return aa_dict.get(three_letter, 'X')

def main():
    print("="*70)
    print("HIERARCHICAL PROTEIN BUILDER - Multi-Scale Structure-Aware")
    print("Based on HIGH-PPI, ProNet, HoloProt, MGPPI architectures")
    print("="*70)
    
    # Load existing protein graphs
    protein_graphs_path = Path('data/processed/all_protein_graphs_esm2.pkl')
    with open(protein_graphs_path, 'rb') as f:
        protein_graphs = pickle.load(f)
    
    print(f"\n✓ Loaded {len(protein_graphs)} proteins")
    
    # Protein conformations directories (poc and phrase 1)
    conf_dirs = [
        PROJECT_ROOT / 'data' / 'conformations' / 'poc_consolidated',
        PROJECT_ROOT / 'data' / 'conformations' / 'full'
    ]

    conf_dirs = [d for d in conf_dirs if d.exists()]

    if not conf_dirs:
        print("\n✗ ERROR: No valid conformation directories found")
        return
        
    builder = MultiScaleProteinBuilder()
    hierarchical_proteins = {}
    
    print(f"\nBuilding hierarchical representations...")
    print("-"*70)
    
    stats = {
        'total_residues': 0,
        'total_sses': 0,
        'total_domains': 0,
        'ss_distribution': defaultdict(int),
    }
        
    for protein_id, conformation_graphs in protein_graphs.items():
        print(f"  Protein {protein_id}:")

        pdb_files = []
        source_dirs = []

        # Collect PDBs from ALL valid conformation directories
        for conf_dir in conf_dirs:
            protein_dir = conf_dir / f"POC_PROT_{protein_id}"
            if protein_dir.exists():
                files = sorted(protein_dir.glob("conf_*.pdb"))
                if files:
                    pdb_files.extend(files)
                    source_dirs.append(protein_dir)

        # No PDBs found anywhere
        if len(pdb_files) == 0:
            print(f"    ✗ No PDB files found for protein {protein_id}")
            continue

        # Ensure deterministic ordering across folders
        pdb_files = sorted(pdb_files, key=lambda p: (p.parent.name, p.name))

        hierarchical_confs = []

        for conf_idx, pdb_file in enumerate(pdb_files):
            hier_conf = builder.build_hierarchical_protein_conformation(
                protein_id, conf_idx, pdb_file
            )

            if hier_conf:
                hierarchical_confs.append(hier_conf)

                # Update stats
                stats['total_residues'] += hier_conf['level1_residues'].num_nodes
                stats['total_sses'] += hier_conf['level2_secondary_structures'].num_nodes
                stats['total_domains'] += hier_conf['level3_domains'].num_nodes

                for ss_type in hier_conf['level2_secondary_structures'].sse_types:
                    stats['ss_distribution'][ss_type] += 1

        if hierarchical_confs:
            hierarchical_proteins[protein_id] = hierarchical_confs

    # Save
    output_path = Path('data/processed/hierarchical_proteins.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(hierarchical_proteins, f)
    
    print("-"*70)
    print(f"\n✅ SUCCESS: Built hierarchical proteins")
    print(f"   Proteins: {len(hierarchical_proteins)}")
    
    total_confs = sum(len(confs) for confs in hierarchical_proteins.values())
    print(f"   Total conformations: {total_confs}")
    print(f"✅ Saved to: {output_path}")
    
    # Statistics
    if total_confs > 0:
        print(f"\n📊 Statistics:")
        print(f"   Total residues: {stats['total_residues']}")
        print(f"   Avg residues per conformation: {stats['total_residues']/total_confs:.1f}")
        print(f"   Total SSEs: {stats['total_sses']}")
        print(f"   Avg SSEs per conformation: {stats['total_sses']/total_confs:.1f}")
        print(f"   Total domains: {stats['total_domains']}")
        print(f"   Avg domains per conformation: {stats['total_domains']/total_confs:.1f}")
        
        print(f"\n🧬 Secondary Structure Distribution:")
        total_ss = sum(stats['ss_distribution'].values())
        for ss_type, count in sorted(stats['ss_distribution'].items()):
            pct = (count / total_ss * 100) if total_ss > 0 else 0
            ss_name = {'H': 'Helix', 'E': 'Sheet', 'L': 'Loop'}.get(ss_type, ss_type)
            print(f"   {ss_name}: {count} ({pct:.1f}%)")
    
    print("="*70)

if __name__ == "__main__":
    main()
