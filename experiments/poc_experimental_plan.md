# POC Experimental Plan & Success Criteria

**Date Created:** 2025-11-19  
**Project:** DynHierGNN-DTI Bachelor's Thesis  
**Timeline:** Week 1-6 (Proof of Concept)

---

## Objective

**Validate core hypothesis:**  
*Multi-conformational protein modeling improves DTI prediction accuracy over single static structures.*

---

## POC Scope

- **Proteins:** 50 diverse kinases (length 200-600 residues)
- **Drugs:** 20 diverse compounds (MW 200-500 Da)
- **Conformations:** 3 per protein (generated via ColabFold)
- **Total structures:** 150 (50 × 3)
- **Interaction pairs:** ~400 in POC dataset
- **Training compute:** Google Colab (free T4 GPU, ~2-3 hours)

---

## Experimental Design

### Experiment 1: Multi-Conformational vs Static (PRIMARY)

**Multi-Conformational Model:**
- Input: Drug SMILES + Protein (3 conformations)
- Architecture: Simple 2-layer GNN + Temporal Attention
- Output: Binding affinity prediction
- Training: 100 epochs with early stopping
- Test RMSE: Record on held-out test set

**Static Baseline:**
- Input: Drug SMILES + Protein (1st conformation only)
- Architecture: Identical to multi-conf except no attention
- Output: Binding affinity prediction
- Training: Same hyperparameters for fair comparison

**Comparison Metric:**
Improvement = (Static_RMSE - Multi_RMSE) / Static_RMSE × 100%



### Experiment 2: Attention Analysis

**Hypothesis:** Model will learn non-uniform attention across conformations

- Extract attention weights for all test set predictions
- Compute attention entropy for each prediction
- Calculate mean entropy across test set
- Expected: Mean entropy < 1.5 (not uniform)

### Experiment 3: Structural Diversity Correlation

**Hypothesis:** Proteins with diverse conformations benefit more from multi-conf

- Measure RMSD between conformations for each protein
- Correlate RMSD with prediction improvement (multi-conf vs static)
- Expected: Positive correlation (more diverse → more improvement)

---

## Success Criteria

### Quantitative Thresholds

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| **Minimum Success** | Multi-conf ≥5% better than static | ✓ = Proceed |
| **Strong Success** | Multi-conf ≥10% better than static | ✓ = Confident proceed |
| **Baseline RMSE** | < 0.8 on POC test set | ✓ = Reasonable model |
| **Correlation** | Pearson r ≥ 0.5 | ✓ = Meaningful predictions |

### Qualitative Criteria

1. **Attention Diversity:**
   - ✓ Attention weights not uniform
   - ✓ Entropy < 1.5
   - ✓ Weights vary meaningfully across proteins

2. **Training Stability:**
   - ✓ No NaN or Inf values
   - ✓ Smooth loss curves
   - ✓ Validation loss decreases
   - ✓ No severe overfitting

3. **Structural Validity:**
   - ✓ RMSD between conformations: 1-3 Å
   - ✓ pLDDT scores > 70
   - ✓ Conformations are structurally diverse

---

## Decision Gate (Week 5)

### ✅ PROCEED (≥5% improvement)

**Decision:** Hypothesis validated, core contribution proven

**Actions:**
1. Scale to full implementation (442 proteins × 6 conformations)
2. Add hierarchy to architecture
3. Continue to Week 7-12 (Phase 1: Full Implementation)
4. Allocate 17 weeks for full model + evaluation

**Confidence:** High - Core idea works, now optimize

---

### ⚠️ INVESTIGATE (2-5% improvement)

**Decision:** Marginal improvement, need more investigation

**Actions:**
1. Try 5 conformations instead of 3
2. Increase MSA masking (20% instead of 10%)
3. Add stronger regularization
4. Run ablation: 1 vs 3 vs 5 conformations
5. Re-evaluate in Week 6 with extended POC

**Confidence:** Medium - Hypothesis may work with tuning

---

### ❌ PIVOT (≤2% improvement)

**Decision:** Hypothesis not validated with POC data

**Actions:**
1. Focus thesis on hierarchy-only (still novel contribution)
2. OR: Use experimental PDB ensembles instead of AlphaFold
3. OR: Combine multi-conf with different architecture
4. Revise thesis scope with advisor
5. Continue to Phase 1 with modified approach

**Confidence:** Low - Need different strategy

---

## Timeline

| Week | Phase | Tasks | Deliverable |
|------|-------|-------|-------------|
| 1 | Setup | Dataset creation, FASTA | POC dataset ready |
| 2-3 | Generation | ColabFold conformations | 150 structures |
| 4 | Architecture | Build POC model | Model code ready |
| 5 | Training | Train & evaluate | Multi-conf vs static results |
| 6 | Decision | Review with advisor | Go/investigate/pivot decision |

---

## Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| ColabFold slow | Medium | Timeline delay | Start immediately, use Kaggle backup |
| No improvement | Low-Medium | Thesis pivot needed | Have hierarchy-only backup plan |
| Overfitting | Medium | Poor generalization | Strong regularization, dropout |
| Poor structures | Low | Invalid hypothesis test | Regenerate with different seeds |

---

## Literature Validation

Similar approaches in recent work show 5-15% improvements:

- **DynamicBind (2023):** 8-12% improvement with conformational ensembles
- **AFsample2 (2024):** Showed conformational diversity improves docking
- **Our target (5-10%):** Conservative, achievable for POC

---

## Week 5 Presentation to Advisor

**Agenda:**
1. Show POC results (multi-conf vs static comparison)
2. Analyze attention weights (non-uniform?)
3. Discuss structural diversity correlation
4. Present decision gate analysis
5. Recommend: Proceed/Investigate/Pivot
6. Get approval for next phase

**Key Point:** POC success = hypothesis validated, not SOTA performance
