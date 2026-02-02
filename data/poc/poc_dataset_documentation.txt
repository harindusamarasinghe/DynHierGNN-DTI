
# POC Dataset Documentation

**Created:** 2025-11-21 11:00:03  
**Project:** DynHierGNN-DTI (Bachelor's Thesis)  
**Phase:** Week 1 - POC Setup  

---

## Dataset Overview

### Statistics
- **Total interactions:** 1000
- **Unique drugs:** 20
- **Unique proteins:** 50
- **Average pairs per drug:** 50.0
- **Average pairs per protein:** 20.0

### Train/Val/Test Split
- **Train:** 699 pairs (69.9%)
- **Val:** 151 pairs (15.1%)
- **Test:** 150 pairs (15.0%)

### Affinity Distribution (Y)
- **Mean:** 8033.72 ± 3699.17
- **Range:** [0.20, 10000.00]
- **Median:** 10000.00

---

## Drug Selection Rationale

### Criteria
- **Molecular weight:** 277.8 - 493.6 Da (drug-like compounds)
- **Chemical diversity:** Selected across LogP range for diverse scaffolds
- **Count:** 20 drugs

### Properties
| Property | Min | Mean | Max |
|----------|-----|------|-----|
| MW (Da) | 277.8 | 400.2 | 493.6 |
| LogP | 0.28 | 3.66 | 5.87 |
| TPSA (Ų) | 39.3 | 82.3 | 137.1 |
| Atoms | 19 | 29 | 37 |

### Rationale
- Covers diverse chemical scaffolds for generalization testing
- Balanced molecular properties (size, lipophilicity, polarity)
- Representative of drug-like compounds in kinase studies

---

## Protein Selection Rationale

### Criteria
- **Length:** 244 - 599 residues
- **Categories:** Mix of short/medium/long for diversity
- **Count:** 50 proteins

### Distribution
| Category | Count | Length Range (residues) |
|----------|-------|------------------------|
| Short | 15 | < 350 |
| Medium | 25 | 350-500 |
| Long | 10 | > 500 |

### Properties
| Metric | Min | Mean | Max |
|--------|-----|------|-----|
| Length (residues) | 244 | 417 | 599 |

### Rationale
- **Length diversity:** Tests model's ability to handle different protein sizes
- **Short proteins:** Easier to handle (simpler conformations)
- **Medium proteins:** Standard kinase sizes
- **Long proteins:** Challenge model generalization
- **All from Davis kinase dataset:** Known binding sites and mechanisms
- **Manageable for GNN:** 200-600 residues is feasible for POC

---

## Affinity Distribution Preservation

Stratified splitting ensures affinity distribution is preserved across splits:

| Split | Mean | Std |
|-------|------|-----|
| Full | 8033.72 | 3699.17 |
| Train | 8121.53 | 3627.26 |
| Val | 7991.69 | 3735.99 |
| Test | 7666.83 | 3985.78 |

✓ Distributions are well-balanced

---

## Files Generated

- `poc_davis.csv` - Full POC dataset (1000 pairs)
- `poc_train.csv` - Training set (699 pairs)
- `poc_val.csv` - Validation set (151 pairs)
- `poc_test.csv` - Test set (150 pairs)
- `protein_mapping.csv` - 50 proteins with sequences and lengths
- `drug_smiles.csv` - 20 drugs with SMILES and properties
- `poc_dataset_documentation.txt` - This file

---

## Next Steps

### Week 2-3: Conformational Generation
1. Create FASTA file from protein sequences
2. Use ColabFold to generate 3 conformations per protein
3. Result: 150 structures (50 proteins × 3 conformations)

### Week 4: POC Architecture
1. Build simple GNN for drug graph processing
2. Build simple GNN for protein graph processing
3. Implement temporal attention across conformations
4. Create POC model with simple MLP prediction head

### Week 5: POC Validation
1. Train multi-conformational model
2. Train static baseline (1 conformation only)
3. Compare RMSE: Multi-conf vs Static
4. **Success criteria:** Multi-conf should beat static by ≥5%

### Week 5 Decision Gate
- **PROCEED:** If ≥5% improvement → Continue to full implementation
- **INVESTIGATE:** If 2-5% improvement → Try 5 conformations
- **PIVOT:** If ≤2% improvement → Focus on hierarchy-only

---

## POC Success Criteria

### Quantitative
- Multi-conf RMSE ≥5% better than static
- Baseline RMSE < 0.8
- Pearson correlation > 0.5

### Qualitative
- Attention weights are non-uniform (entropy < 1.5)
- Training is stable (no NaN/Inf)
- Structural diversity: RMSD 1-3 Å between conformations

---

## References

- **Davis et al. (2011):** Large-scale kinase inhibitor activity dataset (KIBA & Davis benchmarks)
- **AlphaFold2:** Structure prediction used for conformation generation
- **POC approach:** Rigorous hypothesis validation before full-scale implementation
