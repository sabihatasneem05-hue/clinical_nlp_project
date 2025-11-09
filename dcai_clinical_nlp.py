"""
Data-Centric AI for Clinical NLP - Complete Implementation
Based on: Data-Centric Artificial Intelligence for Textual Understanding 
in Healthcare Decision Systems

Requirements:
pip install numpy pandas scikit-learn scipy

Execute in Visual Studio or any Python IDE
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. SCHEMA AND ONTOLOGY DEFINITIONS
# ============================================================================

class AssertionStatus(Enum):
    """Assertion status for clinical entities"""
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"
    HISTORICAL = "historical"
    CONDITIONAL = "conditional"

class Temporality(Enum):
    """Temporal status of clinical findings"""
    ONGOING = "ongoing"
    PAST = "past"
    RESOLVED = "resolved"
    PLANNED = "planned"

class EntityType(Enum):
    """Clinical entity types"""
    PROBLEM = "problem"
    FINDING = "finding"
    ANATOMY = "anatomy"
    TEST = "test"
    MEASURE = "measure"
    DRUG = "drug"
    DOSE = "dose"
    ROUTE = "route"
    FREQUENCY = "frequency"
    PROCEDURE = "procedure"
    ALLERGEN = "allergen"

@dataclass
class ClinicalEntity:
    """Structured clinical entity with ontology mapping"""
    text_span: str
    start_idx: int
    end_idx: int
    entity_type: EntityType
    lemma: str
    norm_code: str  # SNOMED CT, ICD-10, RxNorm, LOINC
    norm_system: str  # e.g., "SNOMED_CT", "ICD10", "RXNORM"
    assertion: AssertionStatus
    temporality: Temporality
    confidence: float
    evidence_section: str
    experiencer: str = "PATIENT"  # or "OTHER"

@dataclass
class ClinicalRelation:
    """Relations between clinical entities"""
    head_id: str
    tail_id: str
    relation_type: str  # e.g., "DRUG-ADE", "PROBLEM-PROCEDURE"
    evidence_span: str
    confidence: float

# ============================================================================
# 2. NEGATION AND UNCERTAINTY DETECTION
# ============================================================================

class NegationDetector:
    """Handles negation and uncertainty detection with scope rules"""
    
    def __init__(self):
        self.negation_cues = {
            'no', 'not', 'denies', 'denied', 'without', 'negative for',
            'absence of', 'free of', 'rule out', 'ruled out', 'r/o'
        }
        
        self.uncertainty_cues = {
            'possible', 'possibly', 'likely', 'probably', 'suspected',
            'suggestive of', 'cannot exclude', 'may be', 'might be',
            'question of', 'questionable'
        }
        
        self.scope_terminators = {'.', ';', ':', 'but', 'however', 'although', 'except'}
    
    def detect_assertion(self, text: str, entity_span: Tuple[int, int]) -> AssertionStatus:
        """Detect assertion status for an entity"""
        # Get text before entity within scope
        start_idx, end_idx = entity_span
        scope_start = max(0, start_idx - 100)  # Look back 100 chars
        preceding_text = text[scope_start:start_idx].lower()
        
        # Check for scope terminators
        for terminator in self.scope_terminators:
            if terminator in preceding_text:
                last_term = preceding_text.rfind(terminator)
                preceding_text = preceding_text[last_term+1:]
        
        # Check negation first (higher priority)
        for cue in self.negation_cues:
            if cue in preceding_text:
                return AssertionStatus.NEGATED
        
        # Check uncertainty
        for cue in self.uncertainty_cues:
            if cue in preceding_text:
                return AssertionStatus.UNCERTAIN
        
        # Check for historical indicators
        historical_patterns = ['history of', 'h/o', 'past', 'previous', 'prior']
        for pattern in historical_patterns:
            if pattern in preceding_text:
                return AssertionStatus.HISTORICAL
        
        return AssertionStatus.AFFIRMED

# ============================================================================
# 3. PROGRAMMATIC LABELING FUNCTIONS
# ============================================================================

class LabelingFunction:
    """Base class for labeling functions"""
    
    def __init__(self, name: str, accuracy_prior: float = 0.7):
        self.name = name
        self.accuracy_prior = accuracy_prior
        self.coverage = 0.0
    
    def apply(self, text: str) -> Optional[int]:
        """Apply labeling function, return class label or None (abstain)"""
        raise NotImplementedError

class RegexLabelingFunction(LabelingFunction):
    """Regex-based labeling function"""
    
    def __init__(self, name: str, pattern: str, label: int, accuracy_prior: float = 0.7):
        super().__init__(name, accuracy_prior)
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.label = label
    
    def apply(self, text: str) -> Optional[int]:
        if self.pattern.search(text):
            return self.label
        return None

class SectionAwareLF(LabelingFunction):
    """Section-aware labeling function"""
    
    def __init__(self, name: str, section: str, keywords: List[str], label: int):
        super().__init__(name)
        self.section = section
        self.keywords = keywords
        self.label = label
    
    def apply(self, text: str) -> Optional[int]:
        # Simple section detection
        if self.section.lower() in text.lower():
            section_text = self._extract_section(text, self.section)
            if any(kw.lower() in section_text.lower() for kw in self.keywords):
                return self.label
        return None
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract text from specific section"""
        pattern = rf"{section_name}:(.*?)(?=\n[A-Z][a-z]+:|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else ""

# ============================================================================
# 4. WEAK SUPERVISION LABEL MODEL
# ============================================================================

class WeakSupervisionModel:
    """Generative label model for weak supervision"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.class_priors = None
        self.lf_accuracies = None
    
    def fit(self, label_matrix: np.ndarray, lf_priors: Optional[np.ndarray] = None):
        """
        Fit label model using EM algorithm
        
        Args:
            label_matrix: (n_samples, n_lfs) with values in {0, ..., K-1, -1}
                         -1 indicates abstain
            lf_priors: Optional accuracy priors for each LF
        """
        n_samples, n_lfs = label_matrix.shape
        
        # Initialize parameters
        self.class_priors = np.ones(self.num_classes) / self.num_classes
        if lf_priors is not None:
            self.lf_accuracies = lf_priors
        else:
            self.lf_accuracies = np.ones(n_lfs) * 0.7
        
        # EM iterations
        for iteration in range(20):
            # E-step: Compute posteriors
            posteriors = self._compute_posteriors(label_matrix)
            
            # M-step: Update parameters
            self.class_priors = posteriors.mean(axis=0)
            self._update_lf_accuracies(label_matrix, posteriors)
        
        return self
    
    def _compute_posteriors(self, label_matrix: np.ndarray) -> np.ndarray:
        """Compute posterior P(Y|Lambda) for each sample"""
        n_samples = label_matrix.shape[0]
        posteriors = np.zeros((n_samples, self.num_classes))
        
        for i in range(n_samples):
            votes = label_matrix[i]
            
            # Compute likelihood for each class
            for k in range(self.num_classes):
                log_likelihood = np.log(self.class_priors[k])
                
                for j, vote in enumerate(votes):
                    if vote >= 0:  # Not abstaining
                        if vote == k:
                            log_likelihood += np.log(self.lf_accuracies[j])
                        else:
                            log_likelihood += np.log((1 - self.lf_accuracies[j]) / (self.num_classes - 1))
                
                posteriors[i, k] = np.exp(log_likelihood)
            
            # Normalize
            if posteriors[i].sum() > 0:
                posteriors[i] /= posteriors[i].sum()
            else:
                posteriors[i] = np.ones(self.num_classes) / self.num_classes
        
        return posteriors
    
    def _update_lf_accuracies(self, label_matrix: np.ndarray, posteriors: np.ndarray):
        """Update LF accuracy estimates"""
        n_lfs = label_matrix.shape[1]
        
        for j in range(n_lfs):
            votes = label_matrix[:, j]
            mask = votes >= 0  # Non-abstaining votes
            
            if mask.sum() == 0:
                continue
            
            # Compute expected accuracy
            accuracy = 0.0
            for k in range(self.num_classes):
                class_votes = (votes == k) & mask
                if class_votes.sum() > 0:
                    accuracy += (posteriors[class_votes, k].sum())
            
            self.lf_accuracies[j] = accuracy / mask.sum()
    
    def predict_proba(self, label_matrix: np.ndarray) -> np.ndarray:
        """Get probabilistic labels for unlabeled data"""
        return self._compute_posteriors(label_matrix)

# ============================================================================
# 5. ACTIVE LEARNING
# ============================================================================

class ActiveLearner:
    """Active learning for efficient annotation"""
    
    @staticmethod
    def uncertainty_sampling(probs: np.ndarray, n_samples: int) -> np.ndarray:
        """Select samples with highest entropy"""
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return np.argsort(entropy)[-n_samples:]
    
    @staticmethod
    def disagreement_sampling(label_matrix: np.ndarray, n_samples: int) -> np.ndarray:
        """Select samples where LFs disagree most"""
        vote_counts = np.apply_along_axis(
            lambda x: len(set(x[x >= 0])), axis=1, arr=label_matrix
        )
        return np.argsort(vote_counts)[-n_samples:]
    
    @staticmethod
    def combined_acquisition(probs: np.ndarray, label_matrix: np.ndarray,
                           embeddings: Optional[np.ndarray] = None,
                           edge_case_flags: Optional[np.ndarray] = None,
                           n_samples: int = 100,
                           weights: Dict[str, float] = None) -> np.ndarray:
        """Combined acquisition function"""
        if weights is None:
            weights = {'uncertainty': 0.5, 'disagreement': 0.5}
        
        n = len(probs)
        scores = np.zeros(n)
        
        # Uncertainty component
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        if entropy.max() > 0:
            scores += weights['uncertainty'] * (entropy / entropy.max())
        
        # Disagreement component
        vote_counts = np.apply_along_axis(
            lambda x: len(set(x[x >= 0])), axis=1, arr=label_matrix
        )
        if vote_counts.max() > 0:
            scores += weights['disagreement'] * (vote_counts / vote_counts.max())
        
        return np.argsort(scores)[-n_samples:]

# ============================================================================
# 6. CALIBRATION
# ============================================================================

class TemperatureScaling:
    """Temperature scaling for probability calibration"""
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 50):
        """Find optimal temperature on validation set"""
        try:
            from scipy.optimize import minimize
            
            def nll_loss(T):
                scaled_probs = self._softmax(logits / T[0])
                nll = -np.log(scaled_probs[np.arange(len(labels)), labels] + 1e-10).mean()
                return nll
            
            result = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 10.0)], method='L-BFGS-B')
            self.temperature = result.x[0]
        except Exception as e:
            print(f"Warning: Temperature scaling failed ({e}), using T=1.0")
            self.temperature = 1.0
        return self
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling"""
        return self._softmax(logits / self.temperature)

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier score"""
    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))

# ============================================================================
# 7. EVALUATION METRICS
# ============================================================================

class ClinicalMetrics:
    """Comprehensive evaluation metrics for clinical NLP"""
    
    @staticmethod
    def discrimination_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: np.ndarray) -> Dict[str, float]:
        """Compute discrimination metrics"""
        from sklearn.metrics import (precision_recall_fscore_support, 
                                     roc_auc_score, average_precision_score,
                                     accuracy_score)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
        }
        
        # AUROC and AUPRC for binary or OvR multiclass
        try:
            if y_proba.shape[1] == 2:
                metrics['auroc'] = roc_auc_score(y_true, y_proba[:, 1])
                metrics['auprc'] = average_precision_score(y_true, y_proba[:, 1])
            else:
                metrics['auroc'] = roc_auc_score(y_true, y_proba, 
                                                 multi_class='ovr', average='macro')
        except Exception as e:
            metrics['auroc'] = np.nan
            metrics['auprc'] = np.nan
        
        return metrics
    
    @staticmethod
    def calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Compute calibration metrics"""
        return {
            'ece': compute_ece(y_proba, y_true),
            'brier': compute_brier_score(y_proba, y_true)
        }
    
    @staticmethod
    def fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                        subgroups: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute fairness metrics across subgroups"""
        results = {}
        
        for group in np.unique(subgroups):
            mask = subgroups == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            
            tp = np.sum((group_true == 1) & (group_pred == 1))
            tn = np.sum((group_true == 0) & (group_pred == 0))
            fp = np.sum((group_true == 0) & (group_pred == 1))
            fn = np.sum((group_true == 1) & (group_pred == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            results[str(group)] = {
                'tpr': tpr,
                'fpr': fpr,
                'fnr': fnr,
                'n_samples': int(mask.sum())
            }
        
        # Compute disparities
        tprs = [v['tpr'] for v in results.values()]
        fprs = [v['fpr'] for v in results.values()]
        
        results['disparities'] = {
            'delta_tpr': max(tprs) - min(tprs),
            'delta_fpr': max(fprs) - min(fprs)
        }
        
        return results

# ============================================================================
# 8. COMPLETE PIPELINE
# ============================================================================

class ClinicalNLPPipeline:
    """End-to-end DCAI pipeline for clinical NLP"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.labeling_functions: List[LabelingFunction] = []
        self.weak_supervision = WeakSupervisionModel(num_classes)
        self.negation_detector = NegationDetector()
        self.calibrator = TemperatureScaling()
        
    def add_labeling_function(self, lf: LabelingFunction):
        """Add a labeling function to the pipeline"""
        self.labeling_functions.append(lf)
    
    def apply_lfs(self, texts: List[str]) -> np.ndarray:
        """Apply all labeling functions to texts"""
        n_texts = len(texts)
        n_lfs = len(self.labeling_functions)
        label_matrix = np.full((n_texts, n_lfs), -1, dtype=int)
        
        for i, text in enumerate(texts):
            for j, lf in enumerate(self.labeling_functions):
                result = lf.apply(text)
                if result is not None:
                    label_matrix[i, j] = result
        
        # Compute coverage for each LF
        for j, lf in enumerate(self.labeling_functions):
            coverage = (label_matrix[:, j] >= 0).mean()
            lf.coverage = coverage
        
        return label_matrix
    
    def fit_label_model(self, label_matrix: np.ndarray):
        """Fit weak supervision model"""
        lf_priors = np.array([lf.accuracy_prior for lf in self.labeling_functions])
        self.weak_supervision.fit(label_matrix, lf_priors)
        return self
    
    def get_probabilistic_labels(self, label_matrix: np.ndarray) -> np.ndarray:
        """Get soft labels from weak supervision"""
        return self.weak_supervision.predict_proba(label_matrix)
    
    def calibrate(self, logits: np.ndarray, labels: np.ndarray):
        """Calibrate model probabilities"""
        self.calibrator.fit(logits, labels)
        return self
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_proba: np.ndarray, subgroups: Optional[np.ndarray] = None) -> Dict:
        """Comprehensive evaluation"""
        results = {
            'discrimination': ClinicalMetrics.discrimination_metrics(
                y_true, y_pred, y_proba),
            'calibration': ClinicalMetrics.calibration_metrics(y_true, y_proba)
        }
        
        if subgroups is not None:
            results['fairness'] = ClinicalMetrics.fairness_metrics(
                y_true, y_pred, subgroups)
        
        return results

# ============================================================================
# 9. DEMONSTRATION: ICD CODING USE CASE
# ============================================================================

def demonstrate_icd_coding():
    """Comprehensive demonstration of ICD code suggestion pipeline"""
    
    print("="*80)
    print("DATA-CENTRIC AI FOR CLINICAL NLP")
    print("Use Case: ICD Code Suggestion from Discharge Summaries")
    print("="*80)
    print()
    
    # Initialize pipeline
    print("Step 1: Initializing pipeline with 5 diagnostic classes...")
    pipeline = ClinicalNLPPipeline(num_classes=5)
    
    # Define class labels
    class_names = {
        0: "E11 - Type 2 Diabetes Mellitus",
        1: "I21 - Acute Myocardial Infarction",
        2: "J18 - Pneumonia",
        3: "N18 - Chronic Kidney Disease",
        4: "Other/No Primary Diagnosis"
    }
    
    # Add labeling functions
    print("\nStep 2: Adding programmatic labeling functions...")
    
    # Diabetes LFs
    pipeline.add_labeling_function(
        RegexLabelingFunction(
            name="diabetes_keywords",
            pattern=r"diabetes|diabetic|hyperglycemia|A1C|metformin|insulin",
            label=0,
            accuracy_prior=0.85
        )
    )
    
    pipeline.add_labeling_function(
        SectionAwareLF(
            name="assessment_diabetes",
            section="Assessment",
            keywords=["diabetes", "DM", "diabetic", "glucose"],
            label=0
        )
    )
    
    # Cardiac LFs
    pipeline.add_labeling_function(
        RegexLabelingFunction(
            name="cardiac_keywords",
            pattern=r"myocardial infarction|MI|heart attack|chest pain|angina|STEMI|NSTEMI",
            label=1,
            accuracy_prior=0.88
        )
    )
    
    pipeline.add_labeling_function(
        SectionAwareLF(
            name="assessment_cardiac",
            section="Assessment",
            keywords=["cardiac", "coronary", "MI", "infarction"],
            label=1
        )
    )
    
    # Pneumonia LFs
    pipeline.add_labeling_function(
        RegexLabelingFunction(
            name="pneumonia_keywords",
            pattern=r"pneumonia|infiltrate|consolidation|respiratory infection",
            label=2,
            accuracy_prior=0.82
        )
    )
    
    # Kidney disease LFs
    pipeline.add_labeling_function(
        RegexLabelingFunction(
            name="ckd_keywords",
            pattern=r"chronic kidney disease|CKD|renal failure|ESRD|dialysis|creatinine",
            label=3,
            accuracy_prior=0.80
        )
    )
    
    print(f"  ✓ Added {len(pipeline.labeling_functions)} labeling functions")
    
    # Create realistic clinical text samples
    print("\nStep 3: Processing clinical discharge summaries...")
    
    sample_texts = [
        """
        DISCHARGE SUMMARY
        Patient: 65yo M
        
        Chief Complaint: Chest pain
        
        History of Present Illness: Patient presented with acute onset chest pain radiating 
        to left arm. History of diabetes mellitus type 2 and hypertension.
        
        Assessment: Acute myocardial infarction (STEMI). Patient also has diabetes mellitus 
        type 2, currently on metformin.
        
        Plan: Cardiac catheterization performed. Continue cardiac medications and diabetes 
        management. Follow-up in cardiology clinic.
        """,
        
        """
        DISCHARGE SUMMARY
        Patient: 72yo F
        
        Chief Complaint: Shortness of breath, fever
        
        History: Patient with chronic kidney disease stage 4, not on dialysis. Presented 
        with fever, productive cough, and dyspnea for 3 days.
        
        Assessment: Community-acquired pneumonia with right lower lobe infiltrate on chest 
        X-ray. Chronic kidney disease stable.
        
        Plan: IV antibiotics for pneumonia. Renal function monitoring. Discharge on oral 
        antibiotics with pulmonology follow-up.
        """,
        
        """
        DISCHARGE SUMMARY
        Patient: 58yo M
        
        Chief Complaint: Elevated blood sugar
        
        History: Patient with poorly controlled diabetes mellitus type 2. A1C of 9.2%. 
        No chest pain. No evidence of myocardial infarction on EKG.
        
        Assessment: Diabetes mellitus type 2, uncontrolled. Patient denies cardiac symptoms.
        
        Plan: Adjust insulin regimen. Diabetes education. Follow-up with endocrinology 
        in 2 weeks.
        """,
        
        """
        DISCHARGE SUMMARY
        Patient: 80yo F
        
        Chief Complaint: Weakness, poor appetite
        
        History: Patient with ESRD on hemodialysis. Presented with general malaise. 
        No acute events. Possible pneumonia ruled out with negative chest X-ray.
        
        Assessment: Chronic kidney disease stage 5, end-stage renal disease on dialysis.
        
        Plan: Continue dialysis schedule. Nutritional support. Nephrology follow-up.
        """,
        
        """
        DISCHARGE SUMMARY
        Patient: 45yo M
        
        Chief Complaint: Annual physical examination
        
        History: Patient for routine check-up. No active complaints. Past history of 
        hypertension, well controlled. No diabetes. No cardiac disease.
        
        Assessment: No acute findings. Hypertension controlled on current medications.
        
        Plan: Continue current medications. Routine follow-up in 6 months.
        """
    ]
    
    # True labels for evaluation
    true_labels = np.array([1, 2, 0, 3, 4])  # MI, Pneumonia, Diabetes, CKD, Other
    
    print(f"  ✓ Loaded {len(sample_texts)} discharge summaries")
    
    # Apply labeling functions
    print("\nStep 4: Applying labeling functions...")
    label_matrix = pipeline.apply_lfs(sample_texts)
    
    print("\n  Label Matrix (rows=documents, cols=LFs, -1=abstain):")
    print("  ", "-" * 60)
    df_labels = pd.DataFrame(
        label_matrix,
        columns=[lf.name[:20] for lf in pipeline.labeling_functions],
        index=[f"Doc {i+1}" for i in range(len(sample_texts))]
    )
    print(df_labels.to_string())
    
    # Show LF statistics
    print("\n  Labeling Function Statistics:")
    print("  ", "-" * 60)
    for lf in pipeline.labeling_functions:
        print(f"    {lf.name:30s} | Coverage: {lf.coverage:5.2%} | "
              f"Prior Accuracy: {lf.accuracy_prior:5.2%}")
    
    # Fit weak supervision model
    print("\nStep 5: Training weak supervision label model (EM algorithm)...")
    pipeline.fit_label_model(label_matrix)
    
    # Get probabilistic labels
    soft_labels = pipeline.get_probabilistic_labels(label_matrix)
    
    print("\n  Probabilistic Labels (Class Probabilities):")
    print("  ", "-" * 60)
    df_probs = pd.DataFrame(
        soft_labels,
        columns=[f"Class {i}" for i in range(5)],
        index=[f"Doc {i+1}" for i in range(len(sample_texts))]
    )
    print(df_probs.round(3).to_string())
    
    # Get predictions
    predictions = np.argmax(soft_labels, axis=1)
    
    print("\n  Final Predictions vs Ground Truth:")
    print("  ", "-" * 60)
    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        status = "✓" if pred == true else "✗"
        print(f"    Doc {i+1}: Predicted={class_names[pred][:30]:30s} | "
              f"True={class_names[true][:30]:30s} {status}")
    
    # Comprehensive evaluation
    print("\nStep 6: Comprehensive Evaluation...")
    print("  ", "-" * 60)
    
    # Discrimination metrics
    metrics = ClinicalMetrics.discrimination_metrics(true_labels, predictions, soft_labels)
    print("\n  DISCRIMINATION METRICS:")
    for key, value in metrics.items():
        if not np.isnan(value):
            print(f"    {key:20s}: {value:.4f}")
    
    # Calibration metrics
    cal_metrics = ClinicalMetrics.calibration_metrics(true_labels, soft_labels)
    print("\n  CALIBRATION METRICS:")
    for key, value in cal_metrics.items():
        print(f"    {key:20s}: {value:.4f}")
    
    # Demonstrate active learning
    print("\nStep 7: Active Learning - Identifying samples for annotation...")
    print("  ", "-" * 60)
    
    # Uncertainty sampling
    uncertain_samples = ActiveLearner.uncertainty_sampling(soft_labels, n_samples=2)
    print(f"\n  Most uncertain samples (for clinician review):")
    for idx in uncertain_samples:
        entropy = -np.sum(soft_labels[idx] * np.log(soft_labels[idx] + 1e-10))
        top_class = np.argmax(soft_labels[idx])
        print(f"    Doc {idx+1}: Entropy={entropy:.3f}, Top prediction={class_names[top_class][:40]}")
    
    # Disagreement sampling
    disagreement_samples = ActiveLearner.disagreement_sampling(label_matrix, n_samples=2)
    print(f"\n  Samples with most LF disagreement:")
    for idx in disagreement_samples:
        votes = label_matrix[idx]
        n_votes = (votes >= 0).sum()
        unique_votes = len(set(votes[votes >= 0]))
        print(f"    Doc {idx+1}: {n_votes} LFs voted, {unique_votes} different labels")
    
    # Demonstrate negation detection
    print("\nStep 8: Negation & Uncertainty Detection Demo...")
    print("  ", "-" * 60)
    
    test_phrases = [
        ("No chest pain reported", (3, 13)),
        ("Patient denies myocardial infarction", (15, 37)),
        ("Possible pneumonia on imaging", (9, 18)),
        ("History of diabetes mellitus", (11, 28)),
        ("Patient has acute kidney injury", (12, 31))
    ]
    
    detector = NegationDetector()
    print("\n  Clinical phrase assertion detection:")
    for phrase, span in test_phrases:
        assertion = detector.detect_assertion(phrase, span)
        entity = phrase[span[0]:span[1]]
        print(f"    '{entity}' in '{phrase}'")
        print(f"      → Assertion: {assertion.value.upper()}")
    
    return pipeline, soft_labels, true_labels

# ============================================================================
# 10. ADVANCED DEMONSTRATION: ADVERSE DRUG EVENT DETECTION
# ============================================================================

def demonstrate_ade_detection():
    """Demonstrate adverse drug event extraction"""
    
    print("\n\n")
    print("="*80)
    print("ADVANCED USE CASE: Adverse Drug Event (ADE) Detection")
    print("="*80)
    print()
    
    # Binary classification: ADE present or not
    pipeline = ClinicalNLPPipeline(num_classes=2)
    
    print("Step 1: Configuring ADE detection labeling functions...")
    
    # ADE-specific LFs
    pipeline.add_labeling_function(
        RegexLabelingFunction(
            name="ade_explicit",
            pattern=r"adverse (drug )?event|ADE|drug reaction|medication reaction",
            label=1,
            accuracy_prior=0.95
        )
    )
    
    pipeline.add_labeling_function(
        RegexLabelingFunction(
            name="causality_phrases",
            pattern=r"(after|following|secondary to|due to).{0,30}(medication|drug|started)",
            label=1,
            accuracy_prior=0.75
        )
    )
    
    pipeline.add_labeling_function(
        RegexLabelingFunction(
            name="common_ades",
            pattern=r"rash|urticaria|anaphylaxis|hypotension|bradycardia|nausea.{0,20}(drug|medication)",
            label=1,
            accuracy_prior=0.70
        )
    )
    
    pipeline.add_labeling_function(
        RegexLabelingFunction(
            name="no_ade",
            pattern=r"no (known )?drug allerg|no adverse|tolerated (well|without)",
            label=0,
            accuracy_prior=0.85
        )
    )
    
    print(f"  ✓ Configured {len(pipeline.labeling_functions)} ADE-specific LFs")
    
    # Sample progress notes
    ade_texts = [
        """
        Progress Note - Day 2
        Patient developed urticaria and pruritus 2 hours after starting penicillin.
        Medication discontinued. Adverse drug event documented. Started antihistamines.
        """,
        
        """
        Progress Note - Day 1
        Patient tolerating new medications well. No adverse reactions noted.
        Vital signs stable. No known drug allergies.
        """,
        
        """
        Progress Note - Day 3
        Severe hypotension observed following administration of ACE inhibitor.
        Blood pressure dropped to 80/50. Medication held. Possible ADE.
        """,
        
        """
        Progress Note - Day 1
        New anticoagulation started. Patient counseled on side effects.
        No immediate reactions. Continue monitoring.
        """,
        
        """
        Progress Note - Day 2
        Anaphylactic reaction to contrast dye during imaging. Code blue called.
        Patient stabilized with epinephrine. Drug allergy added to chart.
        """
    ]
    
    ade_labels = np.array([1, 0, 1, 0, 1])  # True ADE status
    
    print(f"\nStep 2: Processing {len(ade_texts)} progress notes...")
    
    # Apply LFs
    label_matrix = pipeline.apply_lfs(ade_texts)
    
    print("\n  Label Matrix:")
    df_ade_labels = pd.DataFrame(
        label_matrix,
        columns=[lf.name for lf in pipeline.labeling_functions],
        index=[f"Note {i+1}" for i in range(len(ade_texts))]
    )
    print(df_ade_labels.to_string())
    
    # Fit model
    print("\nStep 3: Training weak supervision model...")
    pipeline.fit_label_model(label_matrix)
    soft_labels = pipeline.get_probabilistic_labels(label_matrix)
    
    predictions = (soft_labels[:, 1] > 0.5).astype(int)
    
    print("\n  Predictions (ADE Detection):")
    print("  ", "-" * 60)
    for i, (pred, true, prob) in enumerate(zip(predictions, ade_labels, soft_labels[:, 1])):
        status = "✓" if pred == true else "✗"
        ade_status = "ADE PRESENT" if pred == 1 else "NO ADE"
        print(f"    Note {i+1}: {ade_status:15s} (confidence: {prob:.3f}) {status}")
    
    # Evaluation
    print("\nStep 4: Safety-Critical Evaluation...")
    metrics = ClinicalMetrics.discrimination_metrics(ade_labels, predictions, soft_labels)
    
    print("\n  Performance Metrics:")
    print(f"    Accuracy:  {metrics['accuracy']:.3f}")
    print(f"    Precision: {metrics['precision_macro']:.3f}")
    print(f"    Recall:    {metrics['recall_macro']:.3f}")
    print(f"    F1-Score:  {metrics['f1_macro']:.3f}")
    
    # Safety check
    false_negatives = np.sum((ade_labels == 1) & (predictions == 0))
    false_positives = np.sum((ade_labels == 0) & (predictions == 1))
    
    print("\n  SAFETY ANALYSIS:")
    print(f"    False Negatives (Missed ADEs):     {false_negatives} ⚠")
    print(f"    False Positives (False Alarms):    {false_positives}")
    print(f"    Sentinel Cases (Critical Misses):  {false_negatives}")
    
    if false_negatives > 0:
        print("\n  ⚠ WARNING: System missed critical ADE cases!")
        print("    Action required: Review false negatives before deployment")
    
    return pipeline

# ============================================================================
# 11. FAIRNESS ANALYSIS DEMONSTRATION
# ============================================================================

def demonstrate_fairness_analysis():
    """Demonstrate fairness evaluation across subgroups"""
    
    print("\n\n")
    print("="*80)
    print("FAIRNESS ANALYSIS: Performance Across Patient Subgroups")
    print("="*80)
    print()
    
    # Simulate predictions for different demographic groups
    np.random.seed(42)
    n_samples = 200
    
    # Create subgroups
    age_groups = np.random.choice(['18-45', '46-65', '65+'], size=n_samples, p=[0.3, 0.4, 0.3])
    sex_groups = np.random.choice(['M', 'F'], size=n_samples, p=[0.5, 0.5])
    
    # Simulate ground truth and predictions with intentional bias
    y_true = np.random.binomial(1, 0.3, size=n_samples)
    
    # Introduce disparities (simulating documentation bias)
    y_pred = y_true.copy()
    
    # Add noise with different rates by group
    for i in range(n_samples):
        if age_groups[i] == '65+':
            # Lower accuracy for elderly
            if np.random.random() < 0.15:
                y_pred[i] = 1 - y_pred[i]
        elif age_groups[i] == '18-45':
            if np.random.random() < 0.08:
                y_pred[i] = 1 - y_pred[i]
        else:
            if np.random.random() < 0.10:
                y_pred[i] = 1 - y_pred[i]
    
    print("Step 1: Analyzing performance across age groups...")
    age_fairness = ClinicalMetrics.fairness_metrics(y_true, y_pred, age_groups)
    
    print("\n  Age Group Performance:")
    print("  ", "-" * 60)
    for group, stats in age_fairness.items():
        if group != 'disparities':
            print(f"\n    {group}:")
            print(f"      Sample Size:  {stats['n_samples']}")
            print(f"      TPR (Recall): {stats['tpr']:.3f}")
            print(f"      FPR:          {stats['fpr']:.3f}")
            print(f"      FNR:          {stats['fnr']:.3f}")
    
    print("\n  DISPARITY METRICS:")
    print(f"    ΔTrue Positive Rate:  {age_fairness['disparities']['delta_tpr']:.3f}")
    print(f"    ΔFalse Positive Rate: {age_fairness['disparities']['delta_fpr']:.3f}")
    
    # Check thresholds
    tpr_disparity = age_fairness['disparities']['delta_tpr']
    fpr_disparity = age_fairness['disparities']['delta_fpr']
    
    print("\n  FAIRNESS ASSESSMENT:")
    if tpr_disparity > 0.05 or fpr_disparity > 0.05:
        print("    ⚠ WARNING: Disparities exceed 5% threshold!")
        print("    Recommended actions:")
        print("      - Stratified sampling to balance training data")
        print("      - Subgroup-specific thresholds")
        print("      - Targeted data augmentation for underperforming groups")
    else:
        print("    ✓ Fairness constraints satisfied")
    
    # Analyze by sex
    print("\n\nStep 2: Analyzing performance by sex...")
    sex_fairness = ClinicalMetrics.fairness_metrics(y_true, y_pred, sex_groups)
    
    print("\n  Sex-Stratified Performance:")
    print("  ", "-" * 60)
    for group, stats in sex_fairness.items():
        if group != 'disparities':
            print(f"\n    {group}:")
            print(f"      Sample Size:  {stats['n_samples']}")
            print(f"      TPR:          {stats['tpr']:.3f}")
            print(f"      FPR:          {stats['fpr']:.3f}")
    
    print("\n  SEX-BASED DISPARITIES:")
    print(f"    ΔTPR: {sex_fairness['disparities']['delta_tpr']:.3f}")
    print(f"    ΔFPR: {sex_fairness['disparities']['delta_fpr']:.3f}")

# ============================================================================
# 12. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print(" DATA-CENTRIC AI FOR CLINICAL NLP - COMPLETE DEMONSTRATION")
    print(" Based on research paper implementation")
    print("="*80)
    
    try:
        # Use Case 1: ICD Coding
        pipeline1, soft_labels, true_labels = demonstrate_icd_coding()
        
        # Use Case 2: ADE Detection
        pipeline2 = demonstrate_ade_detection()
        
        # Use Case 3: Fairness Analysis
        demonstrate_fairness_analysis()
        
        # Summary
        print("\n\n")
        print("="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        print()
        print("✓ Successfully demonstrated:")
        print("  1. ICD code suggestion with weak supervision")
        print("  2. Adverse drug event detection")
        print("  3. Fairness analysis across patient subgroups")
        print("  4. Negation and uncertainty detection")
        print("  5. Active learning for efficient annotation")
        print("  6. Calibration and comprehensive evaluation metrics")
        print()
        print("Key Data-Centric AI Principles Implemented:")
        print("  • Programmatic labeling with multiple weak signals")
        print("  • Generative label model (EM algorithm)")
        print("  • Active learning for sample selection")
        print("  • Comprehensive evaluation (discrimination, calibration, fairness)")
        print("  • Safety-critical assessment for clinical deployment")
        print()
        print("="*80)
        print("Implementation ready for Visual Studio execution!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()