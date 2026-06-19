CONTEXT SOURCE: context/literature/00_syntheses/roter_faden_related_work.md
CONTEXT REASON: user
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
# Roter Faden: Related Work Chapter

> Detailed argumentative thread for each section. For every point, the supporting papers are listed with their specific contribution to that argument. This document is the skeleton you write from — each bullet cluster maps to roughly one paragraph.

---

## Guiding Thesis Statement

NeuroSGG constructs scene graphs through a **decoupled, training-free, neurosymbolic pipeline**: SAM 3 provides mask-grounded perception → deterministic node canonicalization stabilizes open-vocabulary entities → an LLM proposes relations over geometric context → deterministic symbolic rules validate, repair, and complete the graph. The related work chapter must explain **why each of these four design decisions is necessary** by tracing what prior work achieved and what it left unsolved.

---

## Section 1: Scene Graph Generation — Definition, Task, and Foundations

**Purpose:** Establish the scene graph abstraction, the SGG task, and the foundational closed-set work. End by showing that the closed-set assumption + box-based grounding are the two structural limitations this thesis removes.

**Transition into this section:** *(from introduction)* "To ground the thesis contribution, it is necessary to understand what scene graphs represent, how the field formalized their generation, and which assumptions became embedded in the standard pipeline."

---

### 1.1 What is a scene graph?

**Points to make:**
- A scene graph is a directed, labeled graph $G = (V, E)$ where nodes are localized visual entities (objects, regions, stuff) and edges carry predicate labels encoding relationships.
- The atomic unit is the subject–predicate–object triplet: $(s, p, o) \in V \times \mathcal{P} \times V$.
- The representation is more expressive than object detection because it captures *how* entities relate, not just *that* they exist.
- Combinatorial challenge: $n$ objects → $O(n^2)$ candidate pairs before predicate classification.

**Supporting literature:**
| Paper | What it contributes to this point |
|---|---|
| [Visual Relationship Detection](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/visual_relationship_detection/summary.md) (Lu et al., 2016) | Formalized localized $(s, p, o)$ triplets as a vision task. Showed that `person-riding-bicycle` cannot be reduced to detecting `person` and `bicycle` independently — the predicate carries relational meaning. Introduced the VRD dataset (5K images, 100 object / 70 predicate classes). |
| [Visual Genome](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/visual_genome/summary.md) (Krishna et al., 2017) | Made scene graphs a central data structure: 108K images with dense annotations for objects, attributes, relationships, region descriptions, and QA. Linked localization, language, and relational structure in one resource. |
| [SGG Survey](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/sgg_survey/summary.md) (2024) | Provides the canonical categorization of SGG methods and confirms the standard task definition used across the field. |

---

### 1.2 Task decomposition: PredCls, SGCls, SGDet

**Points to make:**
- The field separates the full task into diagnostic subproblems to isolate error sources:
  - **Predicate Classification (PredCls):** ground-truth boxes + labels given → predict predicates only.
  - **Scene Graph Classification (SGCls):** ground-truth boxes given → predict both object labels and predicates.
  - **Scene Graph Detection (SGDet):** detect objects, assign labels, predict predicates — fully end-to-end.
- These decompositions are useful because they reveal *where* a pipeline fails: grounding? labeling? relation prediction?
- For the thesis, this decomposition is important because NeuroSGG's modular pipeline naturally separates these — SAM 3 handles detection, canonicalization handles labeling, and the LLM + validation handles relations.

**Supporting literature:**
| Paper | What it contributes |
|---|---|
| [Iterative Message Passing](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/iterative_message_passing/summary.md) (Xu et al., 2017) | Helped establish PredCls / SGCls / SGDet as the standard evaluation protocol for SGG. |
| [Neural Motifs](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/neural_motifs/summary.md) (Zellers et al., 2018) | Uses the three evaluation modes to demonstrate that much of the "performance" in PredCls comes from label-pair frequency, not visual understanding. |

---

### 1.3 Foundational closed-set models

**Points to make:**
- **Iterative Message Passing** (2017): First end-to-end model treating SGG as structured prediction. Nodes and edges exchange information through recurrent message passing → joint inference is better than independent pair classification. *Thesis lesson:* graph consistency matters, but IMP is tied to detector boxes + fixed label spaces.
- **Neural Motifs** (2018): The most influential critique of closed-set SGG. Showed that VG predicates are highly biased — the most frequent predicate for an object pair is often a strong baseline. The stacked-motif architecture uses bidirectional LSTMs for global context. *Thesis lesson:* high recall can reflect dataset motifs, not image-grounded understanding. This motivates the thesis's emphasis on separating frequency priors from geometric/symbolic evidence.
- **Graph R-CNN** (2018): Introduced the Relation Proposal Network (RePN) to prune the $O(n^2)$ candidate space before classification, plus attentional GCN for context. *Thesis lesson:* not all object pairs are worth classifying — NeuroSGG uses spatial filters for candidate pair generation, which is conceptually similar.
- **DETR** (Carion et al., 2020): Eliminated hand-designed components (anchors, NMS) in object detection by reformulating it as direct set prediction with a transformer encoder-decoder and bipartite matching loss. *Thesis lesson:* DETR's set-prediction paradigm became the architectural template for later one-stage SGG models (RelTR, PSGTR, OvSGTR). The bipartite matching loss is also reused conceptually in node canonicalization (matching SAM 3 candidates to ground-truth objects).
- **RelTR** (2022): Extends DETR's set-prediction paradigm to scene graphs — directly predicts a sparse set of $(s, p, o)$ triplets with coupled subject/object queries and triplet attention, instead of scoring every pair. *Thesis lesson:* shows modern SGG can avoid dense enumeration, but still learns a closed-vocabulary relation classifier.

**Supporting literature:**
| Paper | Specific detail to mention |
|---|---|
| [Iterative Message Passing](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/iterative_message_passing/summary.md) | End-to-end, uses RPN + message passing GRU. Joint inference > independent prediction. |
| [Neural Motifs](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/neural_motifs/summary.md) | Frequency baseline: given object labels, most frequent predicate per pair achieves ~33% R@1 on PredCls. Stacked motif network with bidirectional LSTM context. |
| [Graph R-CNN](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/graph_rcnn/summary.md) | RePN prunes ~96% of candidate edges via learned relatedness scores before aGCN refines the graph. |
| [DETR](file:///Users/jaenix/Documents/master_thesis/context/literature/07_foundational/detr/summary.md) (Carion et al., 2020) | Set prediction with transformers. Bipartite matching loss. No anchors, no NMS. Architectural ancestor of RelTR, PSGTR, OvSGTR. |
| [RelTR](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/reltr/summary.md) | Extends DETR to SGG: coupled subject/object queries, triplet attention in the decoder. Still closed-vocab VG150. |

---

### 1.4 The VG150 benchmark and its limitations

**Points to make:**
- Visual Genome's raw annotations are free-form, noisy, and long-tailed → the field converged on **VG150** (150 object classes, 50 predicates) as the standard evaluation subset.
- VG150 made benchmarking tractable and created a shared comparison point (IMP, Neural Motifs, Graph R-CNN, RelTR all report on it).
- **But VG150 also baked in problems:** the 50 predicates are dominated by a few (e.g., `on`, `has`, `near`), which encourages frequency-exploiting models.
- VG150 uses bounding boxes, not masks → not suitable as the primary benchmark for a SAM 3 mask-first pipeline.
- Introduce **PSG** (Yang et al., 2022) as the mask-grounded alternative: 49K images, panoptic segmentation, 56 predicates, covers things + stuff.
- Introduce **Fair PSGG** (Lorenz et al., 2024) as the evaluation correction: existing PSG metrics inflate scores through duplicate predictions and unfair matching → SingleMPO protocol fixes this.
- **Do not prescribe the thesis evaluation strategy here.** Simply state: "The implications of this benchmark landscape for the thesis's evaluation protocol are discussed in Chapter X."

**Supporting literature:**
| Paper | What it contributes |
|---|---|
| [Visual Genome](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/visual_genome/summary.md) | 108K images, 2.8M relationships. VG150 reduction = most-used subset. Free-form labels need canonicalization. |
| [Neural Motifs](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/neural_motifs/summary.md) | Exposed VG150's predicate frequency bias — sets up the §2 argument. |
| [Panoptic SGG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/panoptic_sgg/summary.md) (Yang et al., 2022) | Introduces PSG task: mask-grounded nodes (things + stuff), 49K images, 133 object categories, 56 relation predicates. Two-stage baseline (PSGFormer) and one-stage baseline (PSGTR). |
| [Fair PSGG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/fair_psgg/summary.md) (Lorenz et al., 2024) | Shows that standard PSG evaluation is unfair: duplicate predictions, biased matching. Proposes SingleMPO and fair ranking. Introduces Pair-Net as a strong new baseline. Critical for any mask-first pipeline that may produce overlapping/duplicate masks. |
| [Visual Relationship Detection](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/visual_relationship_detection/summary.md) | VRD (5K images, 100/70 vocab) as the smaller precursor benchmark. |

---

### 1.5 Transition out of §1

> "The foundational SGG pipeline — object detector, fixed label space, dense pair enumeration, and closed-vocabulary predicate classification — created a shared experimental framework for the field. But it also embedded two structural assumptions that later work progressively questioned: first, that benchmark recall reflects genuine relational understanding rather than dataset statistics; and second, that the vocabulary of visual entities and predicates can be fixed at training time. Before addressing the vocabulary constraint (§3), it is necessary to examine a deeper problem within the closed-set framework itself: the extent to which predicate-frequency bias undermines the reliability of SGG evaluation."

---

## Section 2: Bias, Long Tail, and Commonsense in Closed-Set SGG

**Purpose:** Explain why high Recall@K can come from dataset priors rather than genuine relational understanding. Establish that external knowledge / commonsense can help — but the thesis solves this differently (symbolic validation, not re-balanced training).

**Transition into this section:** *(from §1.5 above)*

---

### 2.1 The frequency-bias problem

**Points to make:**
- Neural Motifs (2018) showed that VG predicates are highly skewed: a handful of generic predicates (`on`, `has`, `near`, `wearing`) dominate, and subject–object label statistics alone are surprisingly predictive.
- **Unbiased SGG / TDE** (Tang et al., 2020): Formalized the bias problem using causal inference. Proposed Total Direct Effect (TDE) — a counterfactual method that removes the "shortcut" path from object labels to predicate prediction, isolating the visual evidence contribution. Showed that existing models rely heavily on the context prior (the "bad" bias) but that some context is genuinely useful (the "good" bias).
- The distinction between **useful priors** (real commonsense: persons often ride bicycles) and **harmful shortcuts** (always predicting the most frequent predicate regardless of visual evidence) is central to the thesis: NeuroSGG uses geometry and symbolic rules to distinguish these.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [Neural Motifs](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/neural_motifs/summary.md) | Frequency baseline achieves surprisingly high PredCls recall. Establishes the problem. |
| [Unbiased SGG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/unbiased_sgg/summary.md) (Tang et al., 2020) | Causal graph: $X → Y$ with confounding context path. TDE = $Y_{do(X=x)} - Y_{do(X=\emptyset)}$. Model-agnostic plug-in. Dramatically improves mean Recall (mR@K) while slightly reducing Recall@K. Shows that debiasing must balance informativeness vs. bias removal. |

---

### 2.2 External knowledge and commonsense as mitigation

**Points to make:**
- Several works tried to improve SGG by injecting external knowledge rather than (only) debiasing training:
  - **External Knowledge + Reconstruction** (Gu et al., 2019): Used ConceptNet commonsense embeddings to refine object and phrase features. Also added an auxiliary image-reconstruction loss to regularize noisy VG annotations. *Key insight:* commonsense knowledge helps predicate prediction, but the knowledge source and the visual model are still entangled in training.
  - **GB-Net** (Zareian et al., 2020): Reframes SGG as building a bridge between an image-conditioned scene graph and a commonsense knowledge graph (ConceptNet). Uses Graph Bridging Network to dynamically transfer knowledge. *Key insight:* the "bridge" metaphor is useful — a knowledge graph can inform which predicates are plausible for a given object pair without retraining the visual model.
  - **Visual Commonsense** (Wang et al., 2020): Learns a separate visual-commonsense model over scene graph statistics and fuses its predictions with visual evidence. Directly addresses the question: "when should the model trust visual evidence vs. graph plausibility?" *Key insight:* this is exactly the question NeuroSGG answers with deterministic rules — but NeuroSGG does it through explicit geometry validation rather than a learned fusion.
  - **HiKER-SGG** (2024): Adds hierarchical commonsense knowledge with three semantic levels (entity, predicate, triplet) and uses this to make SGG more robust under image corruption. Introduces VG-C (corrupted VG) as a robustness benchmark. *Key insight:* hierarchical knowledge improves both clean and corrupted performance, suggesting that structured knowledge representation is valuable even when perception quality varies.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [External Knowledge + Reconstruction](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/external_knowledge_reconstruction/summary.md) | ConceptNet embeddings improve feature refinement. Reconstruction loss regularizes noisy labels. R@100 improves ~3% on VG150 SGDet. |
| [GB-Net](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/gb_net/summary.md) | Bridge between image graph and ConceptNet. Message passing transfers knowledge dynamically. Improves R@50 by 3–5% on VG150. |
| [Visual Commonsense](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/visual_commonsense/summary.md) | Learned commonsense model fused with visual features. Addresses trust question: when to believe visual evidence vs. prior knowledge. |
| [HiKER-SGG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/hiker_sgg/summary.md) | Three-level hierarchical knowledge (entity → predicate → triplet). VG-C benchmark for corruption robustness. Hierarchical structure helps even under noise. |

---

### 2.3 The lesson for the thesis

**Points to make (one paragraph, not multiple):**
- The bias/commonsense literature shows that (a) raw predicate frequency is a real signal but a dangerous shortcut, (b) external knowledge can improve prediction, and (c) the question of when to trust evidence vs. priors is central.
- NeuroSGG inherits these lessons but solves them differently: instead of re-balanced training or learned commonsense fusion, it uses **deterministic symbolic validation** — geometry rules check whether proposed relations are physically plausible, inverse/transitivity constraints enforce logical consistency, and contradiction detection flags implausible outputs. This is closer to the commonsense idea than to the debiasing idea.

---

### 2.4 Transition out of §2

> "Debiasing and commonsense injection improve predicate quality within a fixed vocabulary. But they do not address the more fundamental limitation: the assumption that the vocabulary itself is closed. Real images contain objects and relationships outside any training set, and modern perception models can name entities in open-ended language. The next section examines how recent work has begun to relax this vocabulary constraint."

---

## Section 3: Open-Vocabulary and Foundation-Model SGG

**Purpose:** Position NeuroSGG against the methods that directly address vocabulary limitations. This is the most competitive landscape — structure it carefully.

**Transition into this section:** *(from §2.4 above)*

---

### 3.1 Caption-based and pseudo-label approaches

**Points to make:**
- **PGSG / Pixels-to-Graphs** (Li et al., 2024): Formulates open-vocabulary SGG as image-to-sequence generation with a VLM. The VLM generates a textual scene graph which is then grounded back to image regions. *Strength:* truly open vocabulary — any text the VLM can produce. *Limitation:* VLM-generated graphs are noisy, hallucination-prone, and hard to validate geometrically because the generation is autoregressive text, not structured prediction.
- **OpenPSG** (Zhou et al., 2024): First open-set panoptic SGG framework. Uses OpenSeeD for open-set segmentation + BLIP-2 to auto-regressively predict relations for arbitrary object pairs. Two modes: object-pair classification (IC mode) and image captioning (Cap mode). *Strength:* extends PSG to open-set. *Limitation:* still relies on an LMM to predict relations — no geometric validation, and autoregressive decoding can hallucinate relations.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [PGSG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/pgsg_pixels_to_graphs/summary.md) | VLM generates graph as text → parsed into triplets → grounded to image regions. No explicit geometric validation. VLM hallucination is the main failure mode. |
| [OpenPSG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/openpsg/summary.md) | OpenSeeD segmenter + BLIP-2 for relation prediction. IC mode: classify cropped object pairs. Cap mode: caption-based. Open-set but still LMM-dependent for relations. |

---

### 3.2 CLIP/VLM-driven relation models with learned components

**Points to make:**
- **OvSGTR** (Chen et al., 2024): Defines four formal open-vocabulary SGG settings (open-object, open-predicate, open-object+predicate, open-combination). Proposes a DETR-like model that aligns visual node/edge features with text concepts using CLIP-based visual-concept alignment. Uses Relation Retention Module to prevent catastrophic forgetting of base relations when learning novel ones. *Strength:* the formal OV-SGG taxonomy is valuable; the architecture is principled. *Limitation:* still trains a relation model — needs SGG-specific supervision. Not training-free.
- **Universal SGG / USG** (Wu et al., CVPR 2025): Modality-invariant framework for images, video, text, and 3D data. Uses SAM 2 for pseudo-masks via box prompts. Object Associator aligns cross-modal objects. Text-Centric Scene Contrasting Learning anchors visual features to textual ontologies. *Strength:* cross-modal capability, modality-invariant. *Limitation:* still trains the Object Associator and relation head — not zero-shot for relations.
- **REACT++** (2026): Decoupled Two-Stage architecture with frozen YOLO perception + Transformer relation head. DAMP (Detection-Anchored Multi-scale Pooling) extracts features without RoI Align. CARPE (Cross-Attention Rotary Prototype Embedding) replaces MLP relation heads. *Strength:* formally decoupled, real-time (54ms), excellent engineering. *Limitation:* closed-vocabulary relation model trained on VG/PSG. The architecture is the right pattern but the relation head is still learned + fixed-vocab.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [OvSGTR](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/ovsgtr/summary.md) | 4 OV settings. CLIP alignment. Relation Retention via knowledge distillation. Competitive on VG150 + open-vocab splits. |
| [Universal SGG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/universal_sgg/summary.md) | SAM 2 pseudo-masks. Object Associator + Text-Centric Contrasting. Multi-modal: image, video, text, 3D. |
| [REACT++](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/react_plus_plus/summary.md) | DTS architecture. DAMP for feature extraction from frozen ViT. CARPE for cross-attention relation prediction. 54ms inference. Selected as thesis baseline for its decoupled architecture. |

---

### 3.3 Training-free orchestration

**Points to make:**
- **SAMJAM** (2025): Zero-shot, training-free video SGG for egocentric kitchen videos. Fuses Gemini (for frame-level semantic graphs) with SAM 2 (for geometric mask tracking). Uses discrete bipartite matching to align VLM-generated semantic nodes to SAM 2 spatio-temporal masks. *Strength:* proves that training-free SGG is viable by combining two foundation models through algorithmic alignment. *Limitation:* the VLM generates the graph without geometric validation — SAMJAM trusts the VLM's relations. No symbolic constraint checking, no contradiction detection, no relation repair.
- **NeuroSGG workshop paper** (own work, 2026): Training-free open-vocabulary SGG: local LLM proposes SAM 3 prompts → deterministic canonicalization (mask IoU dedup, schema mapping, audit trail) → LLM predicts relations given geometry context → deterministic validation (geometry checks, inverse/transitivity rules, contradiction detection). *Key contrast vs. SAMJAM:* NeuroSGG adds the symbolic validation layer that SAMJAM lacks. *Key contrast vs. PGSG/OpenPSG:* NeuroSGG separates semantic decisions (LLM) from geometric/logical validation (deterministic code), preventing hallucinated relations from reaching the final graph.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [SAMJAM](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/samjam/summary.md) | Gemini + SAM 2. Bipartite matching. Zero-shot. EPIC-KITCHENS. No geometric validation of relations. |
| [NeuroSGG Workshop](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/neurosgg_open_vocab_workshop/summary.md) | Ollama/SAM3. Deterministic canonicalization + validation. VG150-constrained and open-vocabulary modes. Geometry-based predicate scoring. |

---

### 3.4 LLM/VLM-based scene understanding (why prompting alone is not enough)

**Points to make:**
- Multiple papers now use LLMs/VLMs to generate scene descriptions or graphs (PGSG, OpenPSG, SAMJAM all rely on language models for relation prediction).
- The recurring problem: **language models hallucinate relations.** They can propose `cat sitting on cloud` because it is linguistically plausible, even when it is visually/physically impossible.
- **LLM4SGG** (2024): Directly investigates using LLMs (GPT-4, LLaMA) to predict scene graph predicates given object pairs. Shows that LLMs carry strong relational commonsense and can predict rare predicates better than frequency-biased trained models. However, the LLM has no visual input — it relies entirely on object-label pairs, which means it cannot distinguish visual context and hallucinates relations when labels are ambiguous. *Key insight:* LLMs are powerful predicate proposers but dangerous as sole predicate deciders. NeuroSGG uses the LLM in exactly this role — proposer, not decider — with geometry as the validator.
- **Tarot-SAM3** (2026) demonstrates the same problem from the perception side: naively coupling SAM 3 with an MLLM causes the output to be dominated by the MLLM's hallucination tendencies. Their solution: use SAM 3's geometric boundaries as a structural bottleneck that filters the MLLM's reasoning. *Key insight:* complex reasoning must be layered *on top of* strict geometric tracking, never entangled with it.
- NeuroSGG applies the same principle more broadly: the LLM proposes relations, but deterministic geometry rules and symbolic constraints validate them before they enter the final graph.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [LLM4SGG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/llm4sgg/summary.md) | Uses LLMs (GPT-4, LLaMA) as predicate predictors given object-label pairs. Strong on rare predicates, but no visual input → hallucinates when labels are ambiguous. LLM as proposer, not sole decider. |
| [Tarot-SAM3](file:///Users/jaenix/Documents/master_thesis/context/literature/01_perception_frontends/tarot_sam3/summary.md) | Training-free. MLLM proposes, SAM 3 validates geometrically. Prevents hallucination by enforcing spatial bottleneck. |
| [PGSG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/pgsg_pixels_to_graphs/summary.md) | (reused) VLM generates text graph — no geometric check → hallucination-prone. |
| [OpenPSG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/openpsg/summary.md) | (reused) BLIP-2 predicts relations autoregressively — can hallucinate. |

---

### 3.5 Transition out of §3

> "Open-vocabulary SGG methods have progressively relaxed the vocabulary constraint, from caption-derived pseudo-labels to CLIP-aligned relation heads to fully training-free foundation-model orchestration. But the strongest methods still either train a relation model or trust a language model's output without systematic validation. The thesis's response is to combine training-free perception with deterministic symbolic validation — which requires two building blocks: a strong promptable perception frontend (§4) and a reasoning layer that can validate and repair graph structure (§5)."

---

## Section 4: Promptable Segmentation and Visual Grounding

**Purpose:** Justify SAM 3 and the prompting interface as the perception layer. Explain what Promptable Concept Segmentation gives that earlier approaches couldn't.

**Transition into this section:** *(from §3.5 above)*

---

### 4.1 The SAM lineage: from class-agnostic masks to concept segmentation

**Points to make:**
- **SAM / SAM 2** (2023–2024): Established the promptable segmentation paradigm — class-agnostic masks conditioned on geometric prompts (points, boxes). Powerful for segmentation but **lacked semantic understanding**: the model produces masks but doesn't know *what* the object is. For SGG, this requires an expensive secondary classification stage (e.g., CLIP).
- **SAM 3** (late 2025): Introduces **Promptable Concept Segmentation (PCS)**: accepts short noun phrases or visual exemplars as prompts and exhaustively detects, segments, and tracks all matching instances. Pre-trained on SA-Co (4M+ unique concepts, 220× larger than previous benchmarks). Hard negative mining distinguishes visually similar but conceptually distinct objects (lobster vs. crab).
- **Key architectural innovation — presence head:** SAM 3 decouples recognition from localization. A global presence token predicts whether a concept exists in the image *at all*; individual queries then localize. This drastically reduces hallucinated masks for absent concepts — critical for SGG because it means the candidate node set is clean.
- **Object Multiplexing (SAM 3.1):** Processes all tracked objects simultaneously in a shared-memory approach → 2× throughput, and forces the model to understand spatial distribution of all objects jointly.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [SAM 3 Concepts](file:///Users/jaenix/Documents/master_thesis/context/literature/01_perception_frontends/sam3_concepts/summary.md) | PCS engine. SA-Co dataset (4M concepts). Presence head decouples recognition from localization. Hard negative mining. |
| [Tarot-SAM3](file:///Users/jaenix/Documents/master_thesis/context/literature/01_perception_frontends/tarot_sam3/summary.md) | Shows SAM 3 PCS works well for short noun phrases but struggles with long/implicit referring expressions → motivates the LLM as prompt generator rather than feeding complex descriptions directly to SAM 3. |
| [SAM 3D](file:///Users/jaenix/Documents/master_thesis/context/literature/01_perception_frontends/sam3d/summary.md) | Extends SAM 3 to 3D — relevant as future work for the thesis if facade graphs move to 3D reconstruction. |

---

### 4.2 Why SAM 3 and not alternatives?

**Points to make:**
- **vs. Grounding DINO / GLIP:** These are strong open-vocabulary object detectors that accept text prompts, but they produce **bounding boxes, not masks**. Grounding DINO (Liu et al., 2024) fuses DINO with grounded pre-training to achieve open-set detection with text queries. GLIP (Li et al., 2022) unifies phrase grounding and object detection through language-image pre-training. Both are powerful for localization, but for PSG-style evaluation and for computing geometric features (IoU, containment, boundary contact), masks are strictly more informative than boxes. SAM 3's PCS also handles **multi-instance exhaustive detection** natively ("find all windows"), whereas Grounding DINO/GLIP return a confidence-ranked list that may miss low-confidence instances.
- **vs. OpenSeeD (used by OpenPSG):** OpenSeeD is strong for open-set segmentation but is a trained model on specific datasets — SAM 3's PCS is more general and zero-shot.
- **vs. SAM 2 (used by SAMJAM, USG):** SAM 2 requires a secondary classifier for semantic labels. SAM 3's PCS provides both masks and concept labels in one pass, eliminating the classification bottleneck.
- For NeuroSGG specifically: the LLM generates concept prompts (noun phrases like "window", "door", "balcony") → SAM 3 grounds them → canonicalization stabilizes them. This is only possible because SAM 3 natively understands concept-level prompts.

**Supporting literature:**
| Paper | What it contributes to this comparison |
|---|---|
| [Grounding DINO](file:///Users/jaenix/Documents/master_thesis/context/literature/01_perception_frontends/grounding_dino/summary.md) (Liu et al., 2024) | Open-set object detection with text prompts. Fuses DINO with grounded pre-training. Strong zero-shot detector but box-only output — no masks. |
| [GLIP](file:///Users/jaenix/Documents/master_thesis/context/literature/01_perception_frontends/glip/summary.md) (Li et al., 2022) | Unifies phrase grounding and detection via language-image pre-training. Deep fusion of language and vision. Box-only; needs SAM or similar for masks. |
| [SAMJAM](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/samjam/summary.md) | Uses SAM 2 — needs Gemini for semantic labels. Demonstrates the limitation that NeuroSGG avoids by using SAM 3. |
| [OpenPSG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/openpsg/summary.md) | Uses OpenSeeD — trained open-set segmenter. Less general than SAM 3 PCS. |
| [USG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/universal_sgg/summary.md) | Uses SAM 2 pseudo-masks via box prompts. Still needs separate semantic association. |

---

### 4.3 Node construction: from masks to canonical graph nodes

**Points to make:**
- SAM 3 outputs are raw candidates: mask, label, confidence, source prompt. These are **not yet graph nodes.** The same object can be returned multiple times from different prompts, small fragments may appear as detections, and prompt labels may not match the thesis schema.
- **Node canonicalization** is the "Stage 1.5" that creates a clean contract between perception and graph construction: merge duplicates (mask IoU), reject fragments (area/confidence thresholds), map prompt labels to schema, assign stable IDs.
- **Set-of-Mark (SoM) prompting** (Yang et al., 2023) is directly relevant here: SoM overlays numbered visual markers (alphanumeric tags, masks, boxes) onto image regions, turning the visual scene into a structured "marked" representation that LLMs/VLMs can reference by index. *Thesis use:* NeuroSGG uses a SoM-style approach when presenting canonical nodes to the LLM for relation prediction — each node gets a numbered region in the Set-of-Marks overlay, and the LLM references node IDs rather than describing spatial locations in free text. This drastically reduces hallucination and enables structured relation output.
- This idea has precedents in the literature:
  - Visual Genome already needed canonicalization of free-form crowdsourced labels.
  - Fair PSGG showed that duplicate predictions distort evaluation — one canonical node per real object is essential.
  - SAMJAM's bipartite matching is a form of canonicalization (aligning VLM nodes to SAM masks).

**Supporting literature:**
| Paper | What it contributes |
|---|---|
| [Set-of-Mark Prompting](file:///Users/jaenix/Documents/master_thesis/context/literature/05_visual_grounding/set_of_mark_prompting/summary.md) (Yang et al., 2023) | Overlays numbered visual markers on image regions for LLM/VLM reference. Enables structured visual grounding via index-based referencing. Directly used in NeuroSGG's relation prediction context. |
| [Fair PSGG](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/fair_psgg/summary.md) | Duplicate masks → duplicate relations → inflated scores. Motivates strict canonicalization. |
| [SAMJAM](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/samjam/summary.md) | Bipartite matching as a canonicalization mechanism. |
| [Visual Genome](file:///Users/jaenix/Documents/master_thesis/context/literature/02_scene_graph_generation/visual_genome/summary.md) | Free-form labels need canonicalization (VG150 is itself a canonicalization). |

---

### 4.4 Visual grounding context

**Points to make (brief — only if these papers add something the above doesn't):**
- **LASP** (2025): Language-to-Space Programming for training-free 3D visual grounding. Uses spatial programming to ground language in 3D space without training. *Relevance:* similar philosophy to NeuroSGG — geometry-driven, training-free grounding. Conceptual parallel, not direct ancestor.
- **Zero-shot REC** (2025): Debiased cross-modal attention for zero-shot referring expression comprehension. *Relevance:* shows that zero-shot grounding is feasible but attention-based approaches still struggle with hallucination.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [LASP](file:///Users/jaenix/Documents/master_thesis/context/literature/05_visual_grounding/lasp/summary.md) | Training-free, geometry-driven, 3D. Spatial programming language. |
| [Zero-shot REC](file:///Users/jaenix/Documents/master_thesis/context/literature/05_visual_grounding/zeroshot_rec/summary.md) | Debiased attention maps + code-based reasoning. Zero-shot. |

---

### 4.5 Transition out of §4

> "Promptable concept segmentation provides mask-grounded, semantically labeled visual entities without training a task-specific detector. But perception alone does not produce a scene graph: the candidate nodes must be validated, their relations inferred, and the resulting graph checked for logical consistency. This validation and reasoning layer is the thesis's central contribution, and its design draws on neurosymbolic methods that combine neural perception with symbolic constraint checking."

---

## Section 5: Neurosymbolic Reasoning and Constraint-Based Validation

**Purpose:** Motivate the reasoning/validation layer. Clarify that the thesis sits on the **deterministic/rule-based end** of the neurosymbolic spectrum, not the differentiable-logic end.

**Transition into this section:** *(from §4.5 above)*

---

### 5.1 The neurosymbolic spectrum

**Points to make:**
- **Neurosymbolic AI survey** (Hitzler et al., 2023): Provides the "Why, What, How" taxonomy. Key distinction for the thesis: **Type 1** (symbolic → neural: rules constrain neural outputs) vs. **Type 6** (neural → symbolic → neural: neural perception feeds symbolic reasoning which feeds back). NeuroSGG is closer to Type 1/6: neural perception (SAM 3) + neural semantic reasoning (LLM) → symbolic validation (geometry rules, constraint propagation) → refined graph.
- **NeSy KG Reasoning survey** (2024): Classifies neurosymbolic knowledge graph reasoning into: (a) logically informed embeddings, (b) embeddings trained with logical constraints, (c) rule-learning approaches. *Relevance to thesis:* NeuroSGG's constraint rules (inverse relations, transitivity, contradiction detection) are closest to (c) but are hand-specified rather than learned. The thesis should acknowledge this and position rule learning as future work.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [NeSy AI Survey](file:///Users/jaenix/Documents/master_thesis/context/literature/04_neurosymbolic_reasoning/neurosymbolic_ai_survey/summary.md) | Type 1–6 taxonomy. Positions thesis on the symbolic-validation end. |
| [NeSy KG Reasoning](file:///Users/jaenix/Documents/master_thesis/context/literature/04_neurosymbolic_reasoning/neurosymbolic_kg_reasoning/summary.md) | Three-way classification. NeuroSGG closest to rule-based category but uses hand-specified rather than learned rules. |

---

### 5.2 The LLM-as-controller paradigm

**Points to make:**
- Before discussing constraint validation specifically, it is important to acknowledge the broader paradigm that NeuroSGG's architecture belongs to: **using an LLM/VLM as a controller that generates executable programs over visual primitives**, rather than as a direct answer generator.
- **ViperGPT** (Surís et al., 2023): Generates Python programs from natural-language queries, where the programs call visual API modules (object detection, depth estimation, etc.) to answer visual questions. The LLM never sees pixels — it reasons over API outputs. *Key insight:* separating the reasoning controller (LLM) from the visual primitives (API modules) enables compositional, interpretable, and auditable visual reasoning. NeuroSGG follows this pattern: the LLM generates prompts and proposes relations, but SAM 3 and deterministic geometry code execute the visual and spatial operations.
- **VisualProgramming** (Gupta & Kembhavi, 2023): Similar paradigm — an LLM generates modular visual programs that compose vision modules (object detection, segmentation, VQA) into multi-step reasoning chains. *Key insight:* programs are inspectable and debuggable, unlike end-to-end neural reasoning. NeuroSGG's pipeline — prompt generation → segmentation → canonicalization → relation proposal → validation — is itself a visual program, with the LLM controlling semantic decisions and deterministic code controlling spatial/logical operations.
- *Thesis positioning:* NeuroSGG extends the LLM-as-controller paradigm from visual question answering to scene graph construction. The key addition is the **symbolic validation layer**: ViperGPT and VisualProgramming trust the program output; NeuroSGG adds constraint checking that can reject, repair, or complete the LLM's proposals.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [ViperGPT](file:///Users/jaenix/Documents/master_thesis/context/literature/04_neurosymbolic_reasoning/vipergpt/summary.md) | LLM generates Python programs over visual API modules. Compositional, interpretable visual reasoning. LLM as controller, not perceiver. |
| [VisualProgramming](file:///Users/jaenix/Documents/master_thesis/context/literature/04_neurosymbolic_reasoning/visual_programming/summary.md) | LLM generates modular visual programs composing vision modules. Programs are inspectable/debuggable. Multi-step reasoning chains. |

---

### 5.3 Constraint-based relation validation (the closest ancestors)

**Points to make:**
- **LARC** (2023): Language-Augmented Relation Constraints. Uses linguistic knowledge to derive constraints on scene graph relations — e.g., symmetry constraints (if `A next_to B` then `B next_to A`), sparsity constraints (a predicate can appear at most $k$ times per subject), and plausibility constraints (certain subject-predicate-object combinations are impossible). Applied as post-hoc constraints or during training. *This is the most direct ancestor for NeuroSGG's validation layer.* The thesis's geometry rules (inverse relations, transitivity, contradiction detection) are an instance of LARC-style relation constraints, but derived from spatial geometry rather than language statistics.
- **NAVER** (2025): Neurosymbolic Compositional Automaton for Visual Grounding. Uses ProbLog (probabilistic logic programming) to compose visual primitive detectors with logical rules. The automaton decomposes complex referring expressions into compositional sub-programs. *Relevance:* demonstrates that symbolic program execution over neural primitives can achieve strong grounding — the same architecture pattern as NeuroSGG (LLM proposes, deterministic code validates). But NAVER targets referring expression grounding, not scene graph construction.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [LARC](file:///Users/jaenix/Documents/master_thesis/context/literature/03_scene_graph_reasoning/larc/summary.md) | Symmetry, sparsity, plausibility constraints on scene graph relations. Language-derived. Post-hoc or during training. Most direct ancestor for thesis validation layer. |
| [NAVER](file:///Users/jaenix/Documents/master_thesis/context/literature/04_neurosymbolic_reasoning/naver/summary.md) | ProbLog + neural primitives. Compositional automaton for visual grounding. Demonstrates symbolic execution over neural outputs. |

---

### 5.4 Neuro-symbolic grounding in 3D (conceptual parallels)

**Points to make:**
- **NS3D** (2023): Neuro-Symbolic Grounding of 3D Objects and Relations. Learns neural concept primitives for object attributes and spatial relations in 3D, then composes them with symbolic programs to answer questions. *Relevance:* similar architecture pattern — neural perception + symbolic composition. But NS3D learns the primitives, while NeuroSGG's geometry rules are hand-specified. NS3D also targets 3D question answering, not 2D scene graph construction.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [NS3D](file:///Users/jaenix/Documents/master_thesis/context/literature/04_neurosymbolic_reasoning/ns3d/summary.md) | Learned neural primitives + symbolic programs. 3D grounding + QA. Conceptual parallel to NeuroSGG's architecture. |

---

### 5.5 What the thesis's reasoning layer actually does

**Points to make (one paragraph summarizing the design, not the literature):**
- The thesis implements **deterministic constraint-based validation**:
  - **Inverse relation enforcement:** if `A above B`, add `B below A`.
  - **Transitivity:** if `A part_of B` and `B part_of C`, infer `A part_of C`.
  - **Contradiction detection:** flag `A above B` ∧ `B above A`, or `A part_of B` ∧ `B part_of A`.
  - **Geometry-based plausibility:** check that `above` relations respect centroid $y$-coordinates, that `inside` relations respect mask containment ratios, etc.
- This combines elements from multiple ancestors: LARC's constraint types, ViperGPT/VisualProgramming's program-over-primitives pattern, and NAVER's symbolic-execution-over-neural-outputs principle. But it is deliberately simpler — **explainable, fast, and sufficient** for the spatial/structural predicates that dominate scene graphs. The thesis positions learned constraints and probabilistic logic as future extensions.

---

### 5.6 Transition out of §5

> "The neurosymbolic validation layer completes the thesis pipeline: promptable segmentation grounds open-vocabulary entities as masks, canonicalization creates stable nodes, the LLM proposes relations given geometric context, and symbolic rules validate the result. This pipeline is designed primarily for general scene understanding, but its mask-first, geometry-driven architecture is especially well-suited to domains with strong spatial regularity — such as architectural facades."

---

## Section 6: Application Context (Optional — 1 page max)

**Purpose:** If facade evaluation is confirmed, briefly introduce the application domain. If not, fold this into a motivation paragraph in §1 or the introduction.

---

### 6.1 Facade interpretation as a transfer case

**Points to make:**
- Architectural facades exhibit strong spatial regularity: windows form rows/columns, doors appear at ground level, balconies attach to facades, elements repeat with translational symmetry.
- This regularity makes facades a natural test case for geometry-driven scene graph reasoning: spatial predicates like `aligned_with`, `same_row_as`, `part_of` are directly verifiable from mask geometry.
- **Translational Symmetry Facades** (2022): Explicitly models translational symmetry in facade parsing for 3D reconstruction. *Relevance:* the symmetry detection idea maps to the thesis's `same_row_as` / `same_column_as` / `repeated_with` predicates.
- **TrueCity** (2026): Real + simulated urban data for cross-domain 3D scene understanding. *Relevance:* potential evaluation dataset for urban/facade transfer experiments.
- Existing facade datasets (CMP Facade, eTRIMS, LabelMeFacade) are segmentation datasets, not SGG datasets → the thesis may need a small custom relation annotation layer.

**Supporting literature:**
| Paper | Specific detail |
|---|---|
| [Translational Symmetry Facades](file:///Users/jaenix/Documents/master_thesis/context/literature/06_application_domains/translational_symmetry_facades/summary.md) | Symmetry-aware facade parsing. Translational symmetry detection. 3D reconstruction. |
| [TrueCity](file:///Users/jaenix/Documents/master_thesis/context/literature/06_application_domains/truecity/summary.md) | Real + simulated urban data. Cross-domain 3D. |
| [Thesis Proposal](file:///Users/jaenix/Documents/master_thesis/context/literature/06_application_domains/thesis_proposal_de/summary.md) | Original framing: SGG + neurosymbolic grounding for facade interpretation. |

---

## Chapter Closing

The final paragraph of the related work chapter should tie back to the thesis statement:

> "Across all five threads — foundational SGG, predicate bias, open-vocabulary methods, promptable segmentation, and neurosymbolic reasoning — a consistent gap emerges: no existing method combines training-free open-vocabulary perception with deterministic symbolic validation of graph structure. Caption-based methods hallucinate relations; learned models require task-specific training; zero-shot orchestration methods like SAMJAM trust the language model without geometric verification. NeuroSGG addresses this gap through a decoupled pipeline where each stage has a clear responsibility: SAM 3 grounds concepts as masks, canonicalization creates stable nodes, the LLM proposes relations given geometric context, and symbolic rules validate, repair, and complete the graph. The following chapter details this pipeline."

---

## Summary: Papers per Section

| Section | Papers | Approx. pages |
|---|---|---|
| §1 Scene Graph Foundations | VRD, Visual Genome, IMP, Neural Motifs, Graph R-CNN, **DETR**, RelTR, PSG, Fair PSGG, SGG Survey | 3–4 |
| §2 Bias & Commonsense | Neural Motifs (reused), Unbiased SGG, External Knowledge, GB-Net, Visual Commonsense, HiKER-SGG | 2–3 |
| §3 Open-Vocab & FM SGG | PGSG, OpenPSG, OvSGTR, USG, REACT++, SAMJAM, NeuroSGG workshop, Tarot-SAM3, **LLM4SGG** | 3–4 |
| §4 Promptable Segmentation | SAM 3, Tarot-SAM3, SAM 3D, **Grounding DINO**, **GLIP**, **Set-of-Mark**, Fair PSGG (reused), SAMJAM (reused), LASP, Zero-shot REC | 2–3 |
| §5 Neurosymbolic Reasoning | NeSy AI Survey, NeSy KG Survey, **ViperGPT**, **VisualProgramming**, LARC, NAVER, NS3D | 3–4 |
| §6 Application Context | Translational Symmetry, TrueCity, Thesis Proposal | 0.5–1 |
| **Total** | **~37 unique papers cited** | **~15–20 pages** |
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/01_perception_frontends/glip/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "glip"
title: "Grounded Language-Image Pre-Training"
authors: ["Liunian Harold Li", "Pengchuan Zhang", "Haotian Zhang", "Jianwei Yang", "Chunyuan Li", "Yiwu Zhong", "Lijuan Wang", "Lu Yuan", "Lei Zhang", "Jenq-Neng Hwang", "Kai-Wei Chang", "Jianfeng Gao"]
year: 2022
venue: "CVPR"
doi: "10.1109/CVPR52688.2022.01069"
url: "https://arxiv.org/abs/2112.03857"
bibtex_key: "li2022glip"
primary_topic: "01_perception_frontends"
related_topics: ["05_visual_grounding"]
relevance: "medium"
status: "read"
tags: ["grounding", "open-vocabulary-detection", "pretraining", "phrase-grounding", "perception-frontend"]
---

# Grounded Language-Image Pre-Training

## TL;DR
GLIP unifies object detection and phrase grounding during pre-training, producing language-aware object-level features that transfer well to zero-shot and few-shot detection tasks.

## Problem Addressed
Object detectors are usually trained around fixed label sets, while phrase grounding models localize textual phrases. GLIP argues these should be treated as one task: object detection can be written as phrase grounding over a prompt containing the target categories.

## Key Contributions
- Reformulates object detection as grounding between region features and words/phrases in a text prompt.
- Trains jointly on detection data and grounding data.
- Scales pre-training with 27M grounding examples, including web image-text pairs with pseudo grounding boxes.
- Shows strong zero-shot and few-shot transfer across object-level recognition tasks.
- Establishes an important precursor to later open-set detectors such as Grounding DINO.

## Method Summary
GLIP takes an image and a text prompt. Instead of predicting a fixed classifier output for every detected box, the model aligns region features with phrase/token features from the prompt. This allows the same model to behave as either a detector or a phrase grounding model.

The method uses:

1. a visual encoder for region features;
2. a language encoder for prompt features;
3. cross-modal fusion between visual and textual representations;
4. region-word alignment and localization losses;
5. self-training over large image-text data to scale concept coverage.

## Key Results
- Reports 49.8 AP on COCO zero-shot and 26.9 AP on LVIS zero-shot in the abstract setting.
- After COCO fine-tuning, reports 60.8 AP on validation and 61.5 AP on test-dev.
- Shows strong transfer to 13 downstream object detection datasets, with one-shot GLIP rivaling fully supervised detector baselines.

## Relevance to My Thesis
Medium relevance. GLIP is not a scene graph method, but it is useful background for open-vocabulary perception frontends. It helps explain why language-conditioned detection is a realistic alternative to fixed detector vocabularies and why detection/grounding can be treated in one framework.

For the thesis, GLIP is most useful in the perception alternatives section and as historical context for Grounding DINO. It is less directly useful than Grounding DINO or SAM for implementation because newer systems are more commonly used as plug-in open-set localizers.

## Key Takeaways / Ideas to Reuse
- Cite GLIP for the detection-as-grounding reformulation.
- Use it to motivate language-aware object/node proposal modules.
- Do not conflate open-vocabulary detection with scene graph generation; GLIP localizes objects, not relations.
- Phrase grounding data can expand object concept coverage beyond manually curated detection labels.

## Limitations / Open Questions
- Box-centric rather than mask-centric.
- Does not construct graphs or predict predicates.
- Large-scale pre-training is expensive and not directly reusable inside a thesis-scale method.
- Open-vocabulary localization still depends on prompt design and pre-training distribution.

## Related Papers
- [[grounding_dino]] extends the open-set detector line with DINO-style architecture and stronger grounded pre-training.
- [[set_of_mark_prompting]] solves a complementary problem: making already proposed regions easy for an LMM to refer to.
- [[detr]] provides architectural context for later transformer detectors.

## BibTeX
```bibtex
@inproceedings{li2022glip,
  title={Grounded Language-Image Pre-Training},
  author={Li, Liunian Harold and Zhang, Pengchuan and Zhang, Haotian and Yang, Jianwei and Li, Chunyuan and Zhong, Yiwu and Wang, Lijuan and Yuan, Lu and Zhang, Lei and Hwang, Jenq-Neng and Chang, Kai-Wei and Gao, Jianfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={10965--10975},
  year={2022}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/01_perception_frontends/grounding_dino/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "grounding_dino"
title: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
authors: ["Shilong Liu", "Zhaoyang Zeng", "Tianhe Ren", "Feng Li", "Hao Zhang", "Jie Yang", "Qing Jiang", "Chunyuan Li", "Jianwei Yang", "Hang Su", "Jun Zhu", "Lei Zhang"]
year: 2024
venue: "ECCV"
doi: ""
url: "https://arxiv.org/abs/2303.05499"
bibtex_key: "liu2024grounding"
primary_topic: "01_perception_frontends"
related_topics: ["05_visual_grounding", "02_scene_graph_generation"]
relevance: "high"
status: "read"
tags: ["open-set-detection", "grounding", "vision-language", "dino", "perception-frontend"]
---

# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

## TL;DR
Grounding DINO turns a strong DETR-style detector into an open-set detector by tightly fusing image features with text prompts and grounded pre-training.

## Problem Addressed
Classic object detectors predict a fixed training vocabulary. A thesis pipeline that needs domain-specific facade elements, arbitrary user concepts, or promptable object categories needs a perception frontend that can localize objects described by language, not only COCO/VG classes.

## Key Contributions
- Introduces an open-set detector that accepts category names or referring expressions as text input.
- Builds on DINO-style detection with grounded pre-training over detection, grounding, and caption data.
- Uses tight cross-modal fusion through feature enhancement, language-guided query selection, and a cross-modality decoder.
- Evaluates both open-set object detection and referring object detection, not only closed-set detection.
- Reports strong zero-shot transfer, including 52.5 AP on COCO zero-shot detection and 26.1 mean AP on ODinW zero-shot in the paper.

## Method Summary
Grounding DINO injects language into a transformer detector. Text prompts are encoded and fused with image features before final detection, allowing object queries and region predictions to be conditioned on arbitrary textual concepts.

The architecture has three important fusion points:

1. **Feature enhancer:** combines visual and textual features early enough to shape region representations.
2. **Language-guided query selection:** initializes object queries using text-relevant visual features.
3. **Cross-modality decoder:** refines boxes with joint image-text reasoning.

The model is pre-trained on mixed detection, grounding, and caption-derived data to improve concept coverage.

## Key Results
- Strong zero-shot detection on COCO, LVIS, and ODinW.
- Strong referring object detection on RefCOCO/+/g style benchmarks.
- Often used in practice as a language-conditioned detector before SAM-style segmentation.

## Relevance to My Thesis
High relevance as a perception alternative or companion to SAM-style segmentation. It is not an SGG model, but it can produce language-conditioned boxes that help define candidate nodes before graph construction.

For the thesis, Grounding DINO is useful in a comparison section for Stage 1 perception frontends: SAM provides masks but weak semantics; Grounding DINO provides text-conditioned boxes and labels but not mask-native graph nodes. A practical pipeline may combine Grounding DINO for concept localization with SAM for masks.

## Key Takeaways / Ideas to Reuse
- Use Grounding DINO as the main citation for promptable open-set object detection.
- Distinguish object localization from relation prediction; open-set objects do not solve open-set relations.
- Consider a Grounding DINO + SAM baseline when comparing perception alternatives.
- Text-conditioned detection can reduce the node proposal space before relation construction.
- Referring-expression detection results support language-conditioned localization for non-COCO concepts.

## Limitations / Open Questions
- Outputs boxes rather than full panoptic scene graph nodes.
- Does not predict relations or graph structure.
- Prompt wording can affect detections.
- Open-set detection quality may vary by domain and by concept granularity.

## Related Papers
- [[glip]] is the earlier grounded language-image pre-training model that unifies detection and phrase grounding.
- [[detr]] is part of the architectural lineage behind DETR/DINO-style set prediction.
- [[reltr]] adapts DETR-style ideas to relation triplet prediction.
- [[set_of_mark_prompting]] offers a different way to make regions referenceable to an LMM after detection/segmentation.

## BibTeX
```bibtex
@inproceedings{liu2024grounding,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Jiang, Qing and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and Zhang, Lei},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/01_perception_frontends/sam3_concepts/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "sam3_concepts"
title: "SAM 3: Segment Anything with Concepts"
authors: ["Meta AI"]
year: 2025
venue: "arXiv"
doi: ""
url: ""
bibtex_key: "sam3_concepts_2025"
primary_topic: "01_perception_frontends"
related_topics: ["02_scene_graph_generation"]
relevance: "high"
status: "unread"
tags: ["sam", "segmentation", "concepts", "perception", "foundation-model", "open-vocabulary"]
---

# SAM 3: Segment Anything with Concepts

## TL;DR
SAM 3 introduces Promptable Concept Segmentation (PCS), enabling the model to detect, segment, and track all instances of an open-vocabulary concept across images and videos using text or image exemplar prompts.

## Problem Addressed
Previous SAM models relied primarily on geometric and interactive prompts (points/boxes) to segment specific, localized objects. They lacked the built-in capability to perform open-vocabulary semantic segmentation of *all* occurrences of a specific concept (e.g., "all windows") across a scene in a zero-shot manner.

## Key Results
- Demonstrated significant performance gains on open-vocabulary promptable concept segmentation benchmarks over prior systems.
- Successfully unified detection and tracking without degrading the interactive capabilities of SAM 2.
- Released the SA-Co (Segment Anything with Concepts) dataset, a large-scale benchmark with over 4 million unique concept labels.

## Key Contributions
- **Promptable Concept Segmentation (PCS):** Shifting from "pointing" at pixels to "naming" concepts using text or image exemplars.
- **Unified Architecture:** Combines an image-level detector and a memory-based video tracker using a shared backbone.
- **SA-Co Dataset:** A massive new dataset containing millions of concept labels including hard negatives, curated with AI annotators and verifiers.

## Method Summary
SAM 3 extends the SAM architecture by adding a "presence head" to improve detection accuracy and discriminate between closely related text prompts. It operates as a single foundation model that processes both text (e.g., "yellow school bus") and image exemplar prompts, alongside traditional geometric prompts. It integrates an image-level detector with a memory-based video tracker to ensure consistent tracking across frames. The model was trained on the SA-Co dataset using a data engine that focuses human verification effort on hard failures.

## Relevance to My Thesis
This is the concept-enriched variant of SAM — directly relevant to Stage 1 (Perception Frontend) and Stage 1.5 (Node Canonicalization). If SAM 3 can output concept labels alongside masks, it may close the gap between raw segmentation and semantic node initialization, reducing the work needed in the canonicalization stage.

## Key Takeaways / Ideas to Reuse
- **Direct Semantic Initialization:** Use SAM 3's text-prompting to extract all instances of architectural concepts (e.g., "window", "door") in one pass, initializing semantic node types in the scene graph directly.
- **Presence Detection:** Leverage the new presence head to reliably check if a concept (e.g., "balcony") exists in the scene before attempting to segment it, which can act as a structural prior.
- **Exemplar Prompting:** Use image exemplars to segment domain-specific or novel architectural elements without needing a predefined text vocabulary.

## Limitations / Open Questions
- How rich are the concept labels compared to a standalone classifier?
- Does concept output quality degrade on out-of-distribution domains (facades)?

## Related Papers
- [[sam3d]] — sibling paper: SAM 3D extension
- [[fair_psgg]] — downstream SGG that consumes perception outputs
- [[ns3d]] — neurosymbolic reasoning over concept-enriched inputs

## BibTeX
```bibtex
@article{sam3_concepts_2025,
  title={SAM 3: Segment Anything with Concepts},
  author={Meta AI},
  journal={arXiv preprint},
  year={2025}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/01_perception_frontends/sam3d/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "sam3d"
title: "SAM 3D: 3Dfy Anything in Images"
authors: ["Xingyu Chen", "Fu-Jen Chu", "Pierre Gleize", "Kevin J. Liang", "Alexander Sax", "Hao Tang", "Weiyao Wang", "Michelle Guo", "Thibaut Hardin", "Xiang Li", "Aohan Lin", "Jia-Wei Liu", "Ziqi Ma", "Anushka Sagar", "Bowen Song", "Xiaodong Wang", "Jianing Yang", "Bowen Zhang", "Piotr Dollár", "Georgia Gkioxari", "Matt Feiszli", "Malik Jitendra"]
year: 2025
venue: "arXiv"
doi: ""
url: "https://arxiv.org/abs/2511.16624"
bibtex_key: "chen2025sam3d"
primary_topic: "01_perception_frontends"
related_topics: ["04_neurosymbolic_reasoning"]
relevance: "high"
status: "unread"
tags: ["sam", "3d", "segmentation", "perception", "foundation-model"]
---

# SAM 3D: 3Dfy Anything in Images

## TL;DR
TODO — One-sentence summary.

## Problem Addressed
TODO

## Key Contributions
- TODO

## Method Summary
TODO

## Key Results
- TODO

## Relevance to My Thesis
SAM 3 / SAM 3D is the preferred perception frontend for the thesis pipeline (Stage 1). It provides strong, promptable segmentation that can generalize across domains. Understanding its output format and limitations is essential for designing the node canonicalisation module (Stage 1.5).

## Key Takeaways / Ideas to Reuse
- TODO

## Limitations / Open Questions
- No semantic labels — classification must happen separately
- No relational output — graph construction is entirely downstream

## Related Papers
- [[ns3d]] — NS3D also operates in 3D space
- [[truecity]] — Potential evaluation domain for 3D perception

## BibTeX
```bibtex
@article{chen2025sam3d,
  title={SAM 3D: 3Dfy Anything in Images},
  author={Chen, Xingyu and Chu, Fu-Jen and Gleize, Pierre and Liang, Kevin J. and Sax, Alexander and Tang, Hao and Wang, Weiyao and Guo, Michelle and Hardin, Thibaut and Li, Xiang and Lin, Aohan and Liu, Jia-Wei and Ma, Ziqi and Sagar, Anushka and Song, Bowen and Wang, Xiaodong and Yang, Jianing and Zhang, Bowen and Doll{\'a}r, Piotr and Gkioxari, Georgia and Feiszli, Matt and Malik, Jitendra},
  journal={arXiv preprint arXiv:2511.16624},
  year={2025}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/01_perception_frontends/tarot_sam3/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "tarot_sam3"
title: "Tarot-SAM3: Training-free SAM3 for Referring Expression Segmentation"
authors: ["TODO"]
year: 2026
venue: "arXiv"
doi: ""
url: ""
bibtex_key: "tarot_sam3_2026"
primary_topic: "01_perception_frontends"
related_topics: ["04_neurosymbolic_reasoning"]
relevance: "high"
status: "unread"
tags: ["sam3", "referring-expression", "multimodal", "reasoning"]
---

# Tarot-SAM3: Training-free SAM3 for Referring Expression Segmentation

## TL;DR
Tarot-SAM3 is a training-free framework that combines SAM 3's Promptable Concept Segmentation with the reasoning logic of Multimodal LLMs, using SAM 3's rigorous geometric boundaries to prevent MLLM hallucinations.

## Problem Addressed
Referring Expression Segmentation (RES) models typically rely on massive, annotated datasets and struggle to generalize to *both* explicit and implicit expressions. While SAM 3 has strong zero-shot capabilities, it cannot natively handle complex or implicit reasoning. Existing attempts to couple Multimodal Large Language Models (MLLMs) with SAM 3 are naive: they rely entirely on the MLLM to parse the prompt, allowing language hallucinations to dictate the mask, and they treat SAM 3 as a static predictor without refining its output.

## Key Results
- Achieves State-of-the-Art (SOTA) performance among training-free methods on explicit RES benchmarks (e.g., 75.5 gIoU on RefCOCO testA).
- Surpasses even *fine-tuned* models on implicit/reasoning-based benchmarks (e.g., 74.3 gIoU on the ReasonSeg test set).
- Demonstrates exceptional robustness in open-world scenarios without any additional supervision.

## Key Contributions
- A novel training-free framework that handles both explicit and implicit referring expressions by uncoupling linguistic reasoning from visual grounding.
- **Expression Reasoning Interpreter (ERI):** A phase that structures MLLM reasoning into discrete prompt types (text, box, point) to prevent hallucinated generation.
- **Mask Self-Refining (MSR):** A phase that leverages DINOv3's feature coherence to actively detect and correct over- or under-segmentation mistakes made by SAM 3.

## Method Summary
The framework operates in two distinct phases:
1. **Expression Reasoning Interpreter (ERI):** It reformulates complex expressions into heterogeneous prompts. It extracts the target and refer objects, generates an evaluation criterion map, and rephrases the prompt. It then generates multiple mask candidates from SAM 3 (using text and bounding box prompts) and performs a lightweight consistency filter (e.g. text mask must overlap with bounding box).
2. **Mask Self-Refining (MSR):** It uses the MLLM to select the globally preferred mask. Crucially, it then uses DINOv3 similarity maps across the discriminative regions of the predicted mask to detect inconsistencies. If it detects under-segmentation or over-segmentation, it shifts or adds positive/negative point prompts and re-queries SAM 3 for a refined mask.

## Relevance to My Thesis
Validates the methodology-first approach: complex reasoning must be layered *on top of* strict, deterministic geometric tracking. It provides a blueprint for using SAM 3 as a structural bottleneck to ground multimodal reasoning.

## Key Takeaways / Ideas to Reuse
- **Structural Bottleneck:** MLLM acts as a semantic orchestrator defining scene logic, while SAM 3 strictly enforces the physical boundaries of nodes.
- **Canonicalization:** Highlights the necessity of a canonicalization filter between raw MLLM perception and geometric tracking.
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/external_knowledge_reconstruction/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "external_knowledge_reconstruction"
title: "Scene Graph Generation with External Knowledge and Image Reconstruction"
authors: ["Jiuxiang Gu", "Handong Zhao", "Zhe Lin", "Sheng Li", "Jianfei Cai", "Mingyang Ling"]
year: 2019
venue: "CVPR"
doi: ""
url: "https://arxiv.org/abs/1904.00560"
bibtex_key: "gu2019external"
primary_topic: "02_scene_graph_generation"
related_topics: ["03_scene_graph_reasoning"]
relevance: "high"
status: "read"
tags: ["scene-graph", "external-knowledge", "commonsense", "conceptnet", "image-reconstruction", "dmn"]
---

# Scene Graph Generation with External Knowledge and Image Reconstruction

## TL;DR
Gu et al. improve SGG by refining object and phrase features with ConceptNet commonsense and by adding an auxiliary image-reconstruction loss that regularizes noisy object annotations.

## Problem Addressed
Scene graph datasets are long-tailed, incomplete, and noisy. Predicate labels are strongly tied to object labels, but large datasets such as Visual Genome often miss object annotations or include weak/noisy proposals. The paper targets both relation bias and missing-object noise.

## Key Contributions
- Introduces a knowledge-based feature refinement module that retrieves ConceptNet facts for predicted object labels.
- Uses a Dynamic Memory Network to perform multi-hop attention over retrieved facts and inject the resulting commonsense signal into object and phrase features.
- Adds an image-level auxiliary reconstruction branch during training so the graph model is penalized when object predictions cannot reconstruct the visual content.
- Shows that knowledge and reconstruction are complementary: external facts help semantic plausibility, while image reconstruction helps with missing/noisy object annotations.

## Method Summary
The model follows a proposal-based SGG pipeline. It first generates object and subgraph proposals, refines object/subgraph features jointly, then predicts object categories and pairwise predicates.

The external-knowledge path predicts an object label, retrieves top ConceptNet facts involving that object, encodes those symbolic triples as sentence/fact embeddings, and uses a Dynamic Memory Network to attend to useful facts. The resulting memory vector refines the object features before object and relation classification.

The reconstruction path is a training-only regularizer. It reconstructs the input image from detected object labels and bounding boxes, which encourages object predictions to preserve image-level content even when annotations are sparse.

## Key Results
- On VRD, the full KB-GAN model improves SGGen Rec@50/100 from 18.16/22.30 for the baseline to 20.31/25.01.
- On VG-MSDN, the full model improves SGGen Rec@50/100 over Factorizable Net from 13.06/16.47 to 13.65/17.57.
- Ablations show both the ConceptNet knowledge branch and image reconstruction branch contribute gains, with the knowledge branch especially improving object detection mAP.

## Relevance to My Thesis
This is one of the clearest early examples of injecting external commonsense into SGG. For the thesis, it supports the idea that Stage 2/3 should not rely only on visual pair features. It also warns that external knowledge should be grounded in visual evidence: commonsense improves plausibility, but a separate visual regularizer is needed to counter noisy or missing labels.

The image-reconstruction branch is less directly reusable with a SAM-style frontend, but the principle is useful: graph predictions should be checked against image-level consistency, not only against sparse graph annotations.

## Key Takeaways / Ideas to Reuse
- Retrieve symbolic facts only after node canonicalization has produced plausible labels; raw SAM masks alone are not enough.
- Use external knowledge as feature/context refinement, not as a replacement for geometric evidence.
- Add an image- or mask-consistency check so reasoning cannot hallucinate plausible but visually unsupported nodes.
- Consider a small domain-specific knowledge base for facades instead of broad ConceptNet facts, because architectural relations require spatial/structural precision.

## Limitations / Open Questions
- ConceptNet facts are incomplete and generic; they may not contain the facade-specific relations needed by the thesis.
- The method depends on predicted object labels, so early node-label errors can retrieve irrelevant facts.
- The reconstruction branch uses object labels and boxes, not mask-grounded SAM outputs.
- Evaluation uses recall-based SGG metrics; it does not isolate graph reasoning quality or consistency.

## Related Papers
- [[gb_net]] generalizes the idea by treating scene graphs and commonsense graphs as one bridged heterogeneous graph.
- [[visual_commonsense]] learns commonsense directly from scene graph corpora instead of relying on ConceptNet.
- [[hiker_sgg]] adds hierarchical external knowledge for robustness under corruptions.
- [[unbiased_sgg]] is relevant for the long-tail predicate bias that this paper tries to mitigate.

## BibTeX
```bibtex
@inproceedings{gu2019external,
  title={Scene Graph Generation with External Knowledge and Image Reconstruction},
  author={Gu, Jiuxiang and Zhao, Handong and Lin, Zhe and Li, Sheng and Cai, Jianfei and Ling, Mingyang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1969--1978},
  year={2019}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/fair_psgg/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "fair_psgg"
title: "A Fair Ranking and New Model for Panoptic Scene Graph Generation"
authors: ["Julian Lorenz", "Alexander Pest", "Daniel Kienzle", "Katja Ludwig", "Rainer Lienhart"]
year: 2024
venue: "ECCV 2024"
doi: ""
url: "https://arxiv.org/abs/2404.03450"
bibtex_key: "lorenz2024fair"
primary_topic: "02_scene_graph_generation"
related_topics: ["03_scene_graph_reasoning"]
relevance: "high"
status: "unread"
tags: ["panoptic", "scene-graph", "benchmark", "decoupled", "fair-evaluation"]
---

# A Fair Ranking and New Model for Panoptic Scene Graph Generation

## TL;DR
TODO — One-sentence summary.

## Problem Addressed
TODO

## Key Contributions
- TODO

## Method Summary
TODO

## Key Results
- TODO

## Relevance to My Thesis
Provides a strong benchmark baseline for panoptic scene graph generation. The decoupled evaluation approach (segment first, then predict relations) aligns directly with the thesis pipeline. Can serve as the initial testing baseline for Stage 2.

## Key Takeaways / Ideas to Reuse
- Decoupled evaluation: separate segmentation quality from relation prediction quality
- TODO

## Limitations / Open Questions
- TODO

## Related Papers
- [[universal_sgg]] — Another SGG approach
- [[iterative_message_passing]] — Foundational SGG method

## BibTeX
```bibtex
@inproceedings{lorenz2024fair,
  title={A fair ranking and new model for panoptic scene graph generation},
  author={Lorenz, Julian and Pest, Alexander and Kienzle, Daniel and Ludwig, Katja and Lienhart, Rainer},
  booktitle={Computer Vision--ECCV 2024: 18th European Conference, Milan, Italy, September 29--October 4, 2024, Proceedings, Part LXI},
  pages={148--164},
  year={2024},
  organization={Springer}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/gb_net/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "gb_net"
title: "Bridging Knowledge Graphs to Generate Scene Graphs"
authors: ["Alireza Zareian", "Svebor Karaman", "Shih-Fu Chang"]
year: 2020
venue: "ECCV"
doi: ""
url: "https://arxiv.org/abs/2001.02314"
bibtex_key: "zareian2020bridging"
primary_topic: "02_scene_graph_generation"
related_topics: ["03_scene_graph_reasoning", "04_neurosymbolic_reasoning"]
relevance: "high"
status: "read"
tags: ["scene-graph", "knowledge-graph", "commonsense", "message-passing", "gb-net", "bridging"]
---

# Bridging Knowledge Graphs to Generate Scene Graphs

## TL;DR
GB-Net reframes SGG as building a dynamic bridge between an image-conditioned scene graph and a commonsense knowledge graph, then uses message passing to refine both node labels and bridge edges.

## Problem Addressed
Earlier knowledge-enhanced SGG methods either use simple frequency priors or retrieve facts as unstructured triples. They do not fully exploit the graph structure of commonsense knowledge. This paper asks how to treat a scene graph as an instantiation of a commonsense knowledge graph and reason across both structures jointly.

## Key Contributions
- Provides a unified formulation of scene graphs and commonsense graphs as knowledge graphs with entity nodes, predicate nodes, and typed edges.
- Reformulates SGG as inference of "bridge" edges from scene entity/predicate instances to commonsense entity/predicate classes.
- Introduces Graph Bridging Network (GB-Net), which iteratively propagates messages within the scene graph, within the commonsense graph, and across bridge edges.
- Uses pairwise matching to connect scene nodes to class nodes rather than ordinary independent classifiers.

## Method Summary
GB-Net initializes potential entity and predicate nodes from an image. It also maintains a commonsense graph whose nodes are entity and predicate classes and whose edges encode general facts/relations. The key variable is a set of bridge edges linking scene instances to commonsense classes.

During each iteration, GB-Net:

1. updates scene and commonsense node representations through graph message passing,
2. compares scene nodes with class nodes to update bridge probabilities,
3. uses the refined bridge to improve the next message-passing round.

This makes object and predicate classification a graph-alignment problem rather than a set of independent softmax decisions.

## Key Results
- GB-Net improves both Recall and mean Recall on Visual Genome compared with prior SGG methods under the paper's settings.
- The class-balanced GB-Net variant improves average mean Recall further while largely preserving ordinary Recall.
- Ablations show removing commonsense knowledge and reducing message-passing/bridge-refinement steps both hurt performance.

## Relevance to My Thesis
This is highly relevant to the missing middle layer between perception outputs and a clean graph. The thesis can reuse the conceptual framing even if the implementation is different: a scene graph node can be linked to an ontology/commonsense class, and graph reasoning can operate over the combined instance-level and class-level graph.

For a SAM-based thesis pipeline, GB-Net suggests that Stage 1.5 node canonicalization should not just assign labels. It should create explicit alignments between visual instances and a schema/ontology, with uncertainty preserved as edge weights.

## Key Takeaways / Ideas to Reuse
- Model canonicalization as bridge inference between instance nodes and ontology/class nodes.
- Keep predicate nodes explicit where useful; treating relations as first-class nodes can make reasoning over relation classes easier.
- Use iterative refinement: visual graph, ontology graph, and bridge confidences should update each other.
- For facades, maintain a small commonsense/domain graph with classes like `facade_element`, `opening`, `window`, `door`, `balcony`, `storey`, and predicate families such as spatial, structural, and grouping.
- Use mean Recall/per-predicate analysis because commonsense priors can otherwise over-reward frequent relations.

## Limitations / Open Questions
- GB-Net is still box-based and Visual Genome-oriented.
- It depends on the quality and coverage of the commonsense graph.
- The method improves SGG metrics but does not directly define a separate graph validation/refinement task.
- It does not address open-vocabulary relation labels or modern segmentation-foundation-model outputs.

## Related Papers
- [[external_knowledge_reconstruction]] is an earlier ConceptNet/DMN approach that treats knowledge as retrieved facts.
- [[hiker_sgg]] extends the bridged-knowledge idea with hierarchical superclass reasoning and robustness testing.
- [[neurosymbolic_kg_reasoning]] gives the broader taxonomy for graph reasoning over KGs.
- [[visual_commonsense]] learns graph plausibility as a separate module.

## BibTeX
```bibtex
@inproceedings{zareian2020bridging,
  title={Bridging Knowledge Graphs to Generate Scene Graphs},
  author={Zareian, Alireza and Karaman, Svebor and Chang, Shih-Fu},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={606--623},
  year={2020}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/graph_rcnn/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "graph_rcnn"
title: "Graph R-CNN for Scene Graph Generation"
authors: ["Jianwei Yang", "Jiasen Lu", "Stefan Lee", "Dhruv Batra", "Devi Parikh"]
year: 2018
venue: "ECCV"
doi: ""
url: "https://arxiv.org/abs/1808.00191"
bibtex_key: "yang2018graph"
primary_topic: "02_scene_graph_generation"
related_topics: []
relevance: "medium"
status: "read"
tags: ["scene-graph", "gcn", "relation-proposal"]
---

# Graph R-CNN for Scene Graph Generation

## TL;DR
Graph R-CNN is an effective scene graph generation model that introduces a Relation Proposal Network (RePN) to intelligently prune the quadratic number of potential relationships and an attentional Graph Convolutional Network (aGCN) to capture contextual information between objects and relations.

## Problem Addressed
The naive approach to scene graph generation considers every pair of object nodes as a potential relationship, which scales quadratically and is computationally impractical. Simply randomly sub-sampling edges is inefficient because interactions in real-world scenes are highly structured and non-random.

## Key Contributions
- **Relation Proposal Network (RePN):** Computes "relatedness" scores to prune unlikely scene graph connections rather than relying on random sampling.
- **Attentional Graph Convolutional Network (aGCN):** Propagates higher-order context through the sparsified graph, updating representations while modulating information flow across unreliable edges via learned attention.
- **SGGen+ Metric:** A more holistic evaluation metric that computes total recall for singleton entities, pair entities, and triplets, rather than strictly triplets.

## Method Summary
The model works in three stages:
1. **Object Node Extraction:** Uses a standard Faster R-CNN pipeline.
2. **Relationship Edge Pruning (RePN):** Uses an asymmetric kernel function on the object class distributions to efficiently score and prune O(N²) object pairs, keeping only highly probable relationships.
3. **Graph Context Integration (aGCN):** Operates on the pruned candidate graph. It learns to route contextual information between objects and relations adaptively.

## Relevance to My Thesis
Graph R-CNN is foundational for understanding how to handle the combinatorial explosion of potential edges in scene graph generation. The RePN concept is highly relevant for Stage 2 (Relation Prediction) as a filtering mechanism to prune impossible object combinations before applying heavier relation heads (like CARPE).

## Limitations
- Relies heavily on bounding boxes and two-stage architectures, which are slower and less geometrically precise than the panoptic mask-based approaches (like SAM 3) used in the thesis.
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/hiker_sgg/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "hiker_sgg"
title: "HiKER-SGG: Hierarchical Knowledge Enhanced Robust Scene Graph Generation"
authors: ["Ce Zhang", "Simon Stepputtis", "Joseph Campbell", "Katia Sycara", "Yaqi Xie"]
year: 2024
venue: "arXiv"
doi: ""
url: "https://arxiv.org/abs/2403.12033"
bibtex_key: "zhang2024hiker"
primary_topic: "02_scene_graph_generation"
related_topics: ["03_scene_graph_reasoning", "04_neurosymbolic_reasoning"]
relevance: "high"
status: "read"
tags: ["scene-graph", "hierarchical-knowledge", "robustness", "corruption", "commonsense", "visual-genome"]
---

# HiKER-SGG: Hierarchical Knowledge Enhanced Robust Scene Graph Generation

## TL;DR
HiKER-SGG adds hierarchical commonsense knowledge and hierarchical prediction heads to make SGG more robust on clean and corrupted images, introducing VG-C as a corruption benchmark for scene graphs.

## Problem Addressed
Most SGG methods assume clean images, but real images suffer from fog, snow, glare, water drops, blur, dust, and other corruptions. Under corruption, fine-grained visual features may fail even when broader semantic categories remain inferable. The paper asks whether hierarchical external knowledge can guide predictions from coarse to fine classes and improve robustness.

## Key Contributions
- Introduces robust SGG under real-world image corruptions as an explicit benchmark problem.
- Creates corrupted Visual Genome (VG-C), with 20 procedurally generated corruptions.
- Builds hierarchical commonsense graphs using external knowledge plus automatically discovered superclass structures.
- Uses hierarchical inference to predict coarse entity/predicate categories before finer classes.
- Shows improved clean-image performance and lower degradation under corruptions compared with other knowledge-graph SGG methods.

## Method Summary
HiKER-SGG follows a two-stage SGG pattern. It first initializes scene entity and predicate nodes from an object detector and union-box features. It then builds a hierarchical commonsense graph containing entity classes, predicate classes, superclass nodes, and hierarchy edges.

The hierarchy is discovered from a weighted combination of semantic similarity (GloVe embeddings) and pattern similarity (confusion patterns from a baseline model). Scene graph nodes are bridged to commonsense/hierarchy nodes, message passing updates the combined graph, and hierarchical prediction heads first select coarse categories before narrowing to child classes.

## Key Results
- On clean Visual Genome, HiKER-SGG outperforms GB-Net and EB-Net + EOA on PredCls and SGCls mean Recall under both unconstrained and constrained graph settings.
- On VG-C, HiKER-SGG reports higher average mean Recall and smaller degradation across 20 corruption types than GB-Net and EB-Net.
- Ablations show predicate hierarchy, entity hierarchy, discovered rather than manually fixed hierarchies, and adaptive refinement each contribute.
- The method adds modest training/parameter overhead relative to other knowledge-graph SGG baselines.

## Relevance to My Thesis
High relevance because the thesis needs robust graph construction from imperfect perception. The exact detector setup is older than the intended SAM pipeline, but the hierarchy idea transfers directly: graph nodes and predicates should be organized at multiple semantic levels so reasoning can fall back from fine labels to coarse structural categories.

For facades, this is especially useful. A corrupted or occluded element might be confidently identified as an "opening" before deciding whether it is a window, door, balcony opening, or shadow. Similarly, predicates can be grouped into spatial, structural, grouping, and semantic families before selecting fine relation labels.

## Key Takeaways / Ideas to Reuse
- Add hierarchy to the thesis schema: node types and predicate types should have superclass/family structure.
- Use coarse-to-fine reasoning when perception is uncertain.
- Evaluate graph robustness under image perturbations or synthetic perception noise, not only on clean benchmark outputs.
- Treat hierarchy as both a prediction aid and a reasoning constraint: child labels must be compatible with parent labels.
- Consider corruption/noise stress tests for Stage 1.5 and Stage 2 outputs.

## Limitations / Open Questions
- The model is still built around Faster R-CNN-style boxes and Visual Genome labels.
- VG-C uses synthetic corruptions; facade/urban noise may differ from those transformations.
- Hierarchical class discovery uses semantic and confusion similarity from existing datasets, which may not transfer to a small custom facade schema.
- It improves robustness but does not fully separate graph refinement as a standalone post-hoc reasoning module.

## Related Papers
- [[gb_net]] is the direct graph-bridging predecessor.
- [[external_knowledge_reconstruction]] shows external knowledge can refine SGG features.
- [[visual_commonsense]] is complementary because it learns commonsense plausibility separately.
- [[neurosymbolic_kg_reasoning]] provides the broader taxonomy for hierarchical/constraint-based KG reasoning.

## BibTeX
```bibtex
@article{zhang2024hiker,
  title={HiKER-SGG: Hierarchical Knowledge Enhanced Robust Scene Graph Generation},
  author={Zhang, Ce and Stepputtis, Simon and Campbell, Joseph and Sycara, Katia and Xie, Yaqi},
  journal={arXiv preprint arXiv:2403.12033},
  year={2024}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/iterative_message_passing/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "iterative_message_passing"
title: "Scene Graph Generation by Iterative Message Passing"
authors: ["Danfei Xu", "Yuke Zhu", "Christopher B. Choy", "Fei-Fei Li"]
year: 2017
venue: "CVPR 2017"
doi: ""
url: "https://arxiv.org/abs/1701.02426"
bibtex_key: "xu2017iterative"
primary_topic: "02_scene_graph_generation"
related_topics: []
relevance: "medium"
status: "read"
tags: ["scene-graph", "message-passing", "foundational", "visual-genome", "sparse-annotations"]
---

# Scene Graph Generation by Iterative Message Passing

## TL;DR
Iterative Message Passing is an early end-to-end SGG model that treats objects and pairwise relations as a jointly inferred graph and refines node and edge predictions through recurrent message passing.

## Problem Addressed
Object detectors recognize isolated entities but miss the relational structure that distinguishes semantically different scenes with similar objects. Earlier visual relationship methods often predicted each object pair independently, ignoring graph context and the mutual constraints between object labels, boxes, and relation predicates.

## Key Contributions
- Formulates scene graph generation as joint inference over object classes, box refinements, and directed relationship predicates.
- Uses a region proposal network to generate candidate boxes and a graph inference module to refine object and edge predictions.
- Introduces a primal-dual message passing scheme in which node GRUs and edge GRUs exchange contextual information over the scene graph topology.
- Establishes three evaluation settings that became standard in SGG: predicate classification, scene graph classification, and scene graph generation.
- Evaluates on a Visual Genome-derived SGG dataset and on NYU Depth v2 support relations, showing that contextual graph inference helps both sparse semantic relations and denser support-relation prediction.

## Method Summary
Given an image, the model first proposes object boxes. For each box it predicts an object category and bounding-box offsets, and for each ordered object pair it predicts a relationship predicate. Instead of classifying each pair in isolation, the model maintains hidden states for object nodes and relation edges, then iteratively exchanges messages between them. This lets object and predicate decisions influence each other during inference.

The Visual Genome experiment uses the frequent 150 object categories and 50 predicates, which is one origin of the VG150-style closed-vocabulary setup. The paper also notes that Visual Genome relation annotations are sparse: only a small fraction of all possible object pairs are labeled with a relationship.

## Key Results
- Outperforms a visual relationship detection baseline on Visual Genome-derived scene graph generation tasks.
- Reports improvements across predicate classification, scene graph classification, and scene graph generation.
- Shows an 18% gain on predicate classification R@100 over the independent baseline in the reported setup.
- Generalizes the message-passing formulation to NYU Depth v2 support relation graphs.

## Relevance to My Thesis
Foundational SGG method that established the message-passing paradigm for scene graph generation. Important for understanding the evolution of SGG methods and as a historical baseline.

## Key Takeaways / Ideas to Reuse
- Scene graph generation is not only relation classification; object, box, and predicate decisions interact.
- Context can improve predicate prediction, especially when local visual evidence is ambiguous.
- The standard SGG evaluation modes separate relation-only behavior from full end-to-end detection.
- R@K was adopted partly because Visual Genome annotations are sparse and incomplete, but this also makes false positives hard to interpret.
- The closed 150-object / 50-predicate setup is useful historically, not necessarily ideal for a SAM3 open-vocabulary pipeline.

## Limitations / Open Questions
- Uses bounding boxes rather than segmentation masks.
- Depends on a fixed object and predicate vocabulary.
- Predicts relations over proposal boxes, so duplicate/fragmented object proposals can propagate into graph errors.
- Visual Genome's sparse annotations mean correct but unlabeled predicted relations may be penalized or ignored depending on the metric.

## Related Papers
- [[visual_relationship_detection]] motivates the triplet formulation but predicts pairs independently.
- [[visual_genome]] supplies the dense image-level scene graph annotations.
- [[neural_motifs]] later shows that VG relation labels are strongly predictable from object-pair labels.
- [[reltr]] reformulates SGG as one-stage transformer set prediction.
- [[panoptic_sgg]] shifts graph grounding from boxes to masks.

## BibTeX
```bibtex
@inproceedings{xu2017iterative,
  title={Iterative Message Passing for Scene Graph Generation},
  author={Xu, Danfei and Zhu, Yuke and Choy, Christopher B. and Fei-Fei, Li},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2544--2553},
  year={2017}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/llm4sgg/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "llm4sgg"
title: "LLM4SGG: Large Language Models for Weakly Supervised Scene Graph Generation"
authors: ["Kibum Kim", "Kanghoon Yoon", "Jaehyeong Jeon", "Yeonjun In", "Jinyoung Moon", "Donghyun Kim", "Chanyoung Park"]
year: 2024
venue: "CVPR"
doi: ""
url: "https://arxiv.org/abs/2310.10404"
bibtex_key: "kim2024llm4sgg"
primary_topic: "02_scene_graph_generation"
related_topics: ["04_neurosymbolic_reasoning", "05_visual_grounding"]
relevance: "high"
status: "read"
tags: ["scene-graph", "weak-supervision", "llm", "caption-triplets", "open-vocabulary", "pseudo-labels"]
---

# LLM4SGG: Large Language Models for Weakly Supervised Scene Graph Generation

## TL;DR
LLM4SGG uses an LLM with chain-of-thought and few-shot prompts to extract and align caption triplets for weakly supervised SGG, improving pseudo-label quality before a conventional grounding/training stage.

## Problem Addressed
Fully supervised SGG depends on expensive dense scene graph annotations. Weakly supervised SGG tries to use image-caption pairs instead, but common parser-plus-knowledge-base pipelines create poor pseudo labels:

- **Semantic over-simplification:** fine-grained caption predicates such as "lying on" collapse into coarse predicates such as "on".
- **Low-density scene graphs:** many triplets are discarded when parser outputs cannot be aligned to the target object/predicate vocabulary.

These failures matter because the downstream SGG model can only learn relations that survive pseudo-label extraction.

## Key Contributions
- Identifies semantic over-simplification and low-density scene graphs as overlooked failure modes in weakly supervised SGG.
- Replaces rule-based triplet extraction and static KB alignment with LLM-guided triplet extraction and entity/predicate alignment.
- Uses chain-of-thought prompting and in-context examples rather than fine-tuning the LLM.
- Improves weakly supervised SGDet results on Visual Genome and GQA, especially on mean Recall@K, indicating better coverage of long-tail predicates.
- Shows data efficiency: the method remains useful with a smaller subset of captioned training images.

## Method Summary
LLM4SGG keeps the broad weak-supervision pipeline: captions are converted into unlocalized triplets, triplets are aligned to target classes, entities are grounded to image regions, and the resulting localized triplets train an SGG model.

The change is in the triplet formation stage. Instead of relying on a scene parser and WordNet-style alignment, the method prompts an LLM to:

1. paraphrase captions when useful to expose fine-grained relations;
2. extract subject-predicate-object triplets;
3. align extracted entity names to target object categories;
4. align extracted predicate phrases to target predicate classes.

The localized pseudo labels are then passed into existing WSSGG training pipelines, so the paper is best understood as a supervision-improvement module, not a new direct inference architecture.

## Key Results
- Reports stronger Recall@K and mean Recall@K than prior weakly supervised SGG methods on Visual Genome.
- Shows gains on GQA, suggesting that the LLM-based triplet formation is not limited to one dataset.
- Ablations support both major claims: LLM-based extraction improves predicate coverage, and LLM-based alignment reduces discarded triplets.
- Smaller LLM experiments indicate the approach is possible beyond the largest hosted models, but quality drops when the language model is weaker.

## Relevance to My Thesis
High relevance as the closest supplied LLM-based SGG paper. It confirms that LLMs are useful for graph construction, but its exact role differs from the thesis pipeline: LLM4SGG uses an LLM offline to build pseudo-labels for training, while the thesis uses an LLM or VLM at inference time to propose or canonicalize graph structure under geometric validation.

This distinction is important for related work. LLM4SGG is a direct citation for "LLMs in SGG", but not a direct baseline for a training-free, SAM-grounded, code-validated graph constructor.

## Key Takeaways / Ideas to Reuse
- Cite LLM4SGG when claiming that LLMs have already been used to improve SGG supervision.
- Use the semantic over-simplification failure mode as motivation for avoiding flat, generic predicates in the thesis relation schema.
- Treat LLM outputs as proposals or pseudo labels that still require visual grounding and validation.
- Mean Recall@K is the right metric to discuss when evaluating whether an LLM helps long-tail predicate coverage.
- Keep the distinction between caption-derived relation knowledge and image-grounded relation inference explicit.

## Limitations / Open Questions
- The method still trains a downstream SGG model and does not directly construct a scene graph at inference time.
- It is tied to target vocabularies such as Visual Genome/GQA, so it is not fully open-vocabulary in the thesis sense.
- LLM prompting can add cost and may require careful prompt engineering.
- Grounding is still delegated to region/object detectors; mask-grounded graph construction is not the focus.

## Related Papers
- [[pgsg_pixels_to_graphs]] also uses VLMs for open-vocabulary scene graph generation but generates graph sequences directly from images.
- [[openpsg]] uses large multimodal models for open-set panoptic SGG.
- [[set_of_mark_prompting]] is relevant if the thesis uses numbered regions to make LMM graph proposals grounded and referenceable.
- [[vipergpt]] and [[visual_programming]] provide the "LLM proposes executable steps, tools/code validate" design pattern.

## BibTeX
```bibtex
@inproceedings{kim2024llm4sgg,
  title={LLM4SGG: Large Language Models for Weakly Supervised Scene Graph Generation},
  author={Kim, Kibum and Yoon, Kanghoon and Jeon, Jaehyeong and In, Yeonjun and Moon, Jinyoung and Kim, Donghyun and Park, Chanyoung},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/neural_motifs/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "neural_motifs"
title: "Neural Motifs: Scene Graph Parsing with Global Context"
authors: ["Rowan Zellers", "Mark Yatskar", "Sam Thomson", "Yejin Choi"]
year: 2018
venue: "CVPR"
doi: ""
url: "https://arxiv.org/abs/1711.06640"
bibtex_key: "zellers2018neural"
primary_topic: "02_scene_graph_generation"
related_topics: ["03_scene_graph_reasoning"]
relevance: "high"
status: "read"
tags: ["scene-graph", "visual-genome", "motifs", "context-bias"]
---

# Neural Motifs: Scene Graph Parsing with Global Context

## TL;DR
Neural Motifs shows that Visual Genome scene graphs contain strong recurring object-relation patterns, and that object labels alone are often highly predictive of predicates.

## Problem Addressed
Scene graph generation is usually presented as visual relation recognition, but Visual Genome has strong statistical regularities. The paper asks how much relation prediction is driven by recurring graph motifs and object-label priors rather than direct visual evidence.

## Key Contributions
- Provides a quantitative analysis of motifs in Visual Genome scene graphs.
- Shows that object labels are highly predictive of relation labels, while relation labels are much less predictive of object labels.
- Introduces strong frequency baselines that predict the most common relation for a subject-object class pair.
- Proposes Stacked Motif Networks, an architecture that uses global context to stage box, object, and relation prediction.

## Method Summary
The model builds on Faster R-CNN object proposals. It contextualizes detected regions using bidirectional LSTMs, predicts object labels conditioned on global context and previous labels, then predicts relations from contextualized subject, object, and image/union features. The key idea is to model scene graph parsing as a structured prediction problem where object and relation decisions depend on global graph context.

## Key Results
- The paper reports that over half of Visual Genome graphs contain recurring motifs involving at least two relations.
- A frequency baseline based on object-pair labels improves over previous state of the art, showing the strength of dataset priors.
- Stacked Motif Networks further improve over this strong baseline by modeling higher-order context.

## Relevance to My Thesis
This paper is essential for understanding the failure mode of a SAM 3 + relation-head pipeline: even with excellent masks, a learned relation predictor can collapse into object-label priors such as "person wearing shirt" or generic predicates such as "on" and "has". It motivates separating geometric evidence, semantic priors, and reasoning constraints in the thesis pipeline.

## Key Takeaways / Ideas to Reuse
- Treat object-label priors as both useful context and a source of bias.
- Compare any relation predictor against a simple frequency or rule baseline.
- For facade scenes, expect strong motifs such as window-on-facade, door-below-window, and balcony-attached-to-building.
- Use global scene context after SAM 3 node extraction, but avoid letting context overwrite geometric evidence.

## Limitations / Open Questions
- The method is still tied to bounding-box object detection rather than mask-grounded panoptic nodes.
- Its strength partly comes from Visual Genome biases, which may not transfer cleanly to architectural/facade domains.
- The paper improves SGG accuracy but does not solve open-vocabulary node discovery or domain-specific relation definitions.

## Related Papers
- [[visual_genome]] — dataset whose biases and motifs are analyzed.
- [[unbiased_sgg]] — later work that directly targets biased predicate prediction.
- [[graph_rcnn]] — another foundational two-stage SGG architecture.
- [[panoptic_sgg]] — mask-based task formulation closer to SAM 3 outputs.

## BibTeX
```bibtex
@inproceedings{zellers2018neural,
  title={Neural Motifs: Scene Graph Parsing with Global Context},
  author={Zellers, Rowan and Yatskar, Mark and Thomson, Sam and Choi, Yejin},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={5831--5840},
  year={2018}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/neurosgg_open_vocab_workshop/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "neurosgg_open_vocab_workshop"
title: "Neurosymbolic Scene Graph Generation in the Open-Vocabulary Setting"
authors: ["Lukas Arzoumanidis", "Jannik Matijevic", "Youness Dehbi"]
year: 2026
venue: "IJCAI Workshop submission"
doi: ""
url: ""
bibtex_key: "arzoumanidis2026neurosgg"
primary_topic: "02_scene_graph_generation"
related_topics: ["01_perception_frontends", "03_scene_graph_reasoning", "04_neurosymbolic_reasoning"]
relevance: "high"
status: "read"
tags: ["own-work", "open-vocabulary", "training-free", "sam3", "llm", "set-of-marks", "vg150"]
source_pdf: "/Users/jaenix/Downloads/IJCAI__Neurosymbolic_Scene_Graph_Generation_in_the_Open-Vocabulary_Setting_authors.pdf"
---

# Neurosymbolic Scene Graph Generation in the Open-Vocabulary Setting

## TL;DR
NeuroSGG is a training-free open-vocabulary SGG pipeline where the LLM handles semantic decisions and deterministic Python code handles grounding, geometry, validation, and graph assembly.

## Problem Addressed
Most scene graph generation systems are trained and evaluated in closed vocabularies such as VG150. This is convenient for benchmarking, but it is a poor fit for open-world images where object labels and relation predicates are not known in advance. Recent open-vocabulary SGG methods relax this assumption, but still usually require SGG-specific training, weak supervision, or relation-aware pretraining.

## Key Contributions
- Introduces a training-free pipeline for open-vocabulary SGG with a strict semantic/geometric split.
- Uses an LLM to generate SAM3 object prompts, create canonicalization mappings, and predict relation triplets.
- Uses SAM3 concept segmentation for grounded instance masks.
- Builds a Set-of-Marks relation context from numbered bounding boxes so the LLM can refer to concrete segmented instances.
- Validates and assembles relations deterministically, ensuring every graph node is grounded in SAM3 output.
- Provides a preliminary VG150 analysis showing that exact-match closed-vocabulary metrics underestimate expressive open-vocabulary predictions.

## Method Summary
The pipeline has eight stages: LLM prompt generation, SAM3 segmentation, LLM canonicalization config, deterministic canonicalization, Set-of-Marks context construction, LLM relation prediction, geometric validation, and final graph assembly.

The architectural principle is the most important thesis carry-over: semantic choices are made by the LLM, while geometry-sensitive decisions remain deterministic and auditable. This creates a clean boundary between open-vocabulary semantics and grounded graph construction.

## Key Results
- Preliminary VG150 evaluation on 10 images reports node recall around 52.8-53.1%.
- Gemma 4 (31B) reaches 7.8% triplet recall; Qwen 3.5 (122B) reaches 4.7% in the reported setup.
- The submitted paper compares Gemma 4's triplet recall with OvSGTR's reported Mean R@50 of 7.4%, while emphasizing that the metrics are not directly equivalent because this method is training-free and evaluated on a small preliminary subset.
- False predicate errors concentrate on frequent VG150 predicates such as `on`, `in`, `near`, and `has`, supporting the claim that VG150's annotation conventions penalize richer open-vocabulary predicates.

## Relevance to My Thesis
This is now the thesis implementation checkpoint. It turns the earlier methodology-first idea into a runnable pipeline and clarifies the thesis contribution:

- not a new detector;
- not a closed-set relation classifier;
- a neurosymbolic orchestration layer that separates semantic prediction from deterministic grounding and validation;
- a basis for studying how to evaluate open-vocabulary scene graphs fairly.

The thesis should build from this by expanding evaluation, adding geometry-supervised predicate scoring or reranking, and defining a vocabulary-normalized evaluation protocol.

## Key Takeaways / Ideas to Reuse
- The related-work chapter can use the same four buckets as the workshop paper: closed-set SGG, open-vocabulary SGG, text-prompted segmentation, and Set-of-Marks prompting.
- The methodology chapter should center the semantic/geometric split as the core design principle.
- VG150 should be framed as an engineering diagnostic, not as the final thesis-level benchmark.
- PSG or mask-grounded evaluation is still the more natural long-term benchmark for a SAM3-based pipeline.

## Limitations / Open Questions
- The evaluation subset is too small for final thesis claims.
- The current relation stage relies heavily on LLM predicate choice, so semantically equivalent predicates can be counted as false positives under exact-match scoring.
- Geometry is attached as audit features, but it is not yet used to actively score, rerank, or repair relation predictions.
- Open-vocabulary evaluation needs synonym and predicate-normalization machinery.

## Related Papers
- [[visual_genome]] and [[neural_motifs]] explain the VG150 benchmark and its frequency biases.
- [[ovsgtr]] is the strongest trained open-vocabulary comparison point.
- [[pgsg_pixels_to_graphs]] and [[openpsg]] are closest VLM/open-set SGG relatives.
- [[sam3_concepts]] supplies the concept-segmentation frontend.
- [[larc]] and [[naver]] are useful next reads for the reasoning and constraint layer.

## BibTeX
```bibtex
@inproceedings{arzoumanidis2026neurosgg,
  title={Neurosymbolic Scene Graph Generation in the Open-Vocabulary Setting},
  author={Arzoumanidis, Lukas and Matijevic, Jannik and Dehbi, Youness},
  booktitle={IJCAI Workshop submission},
  year={2026}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/openpsg/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "openpsg"
title: "OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models"
authors: ["Zijian Zhou", "Zheng Zhu", "Holger Caesar", "Miaojing Shi"]
year: 2024
venue: "arXiv"
doi: ""
url: "https://arxiv.org/abs/2407.11213"
bibtex_key: "zhou2024openpsg"
primary_topic: "02_scene_graph_generation"
related_topics: ["01_perception_frontends"]
relevance: "high"
status: "read"
tags: ["open-set", "panoptic-scene-graph", "lmm", "autoregressive"]
---

# OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models

## TL;DR
OpenPSG is the first framework for open-set Panoptic Scene Graph (PSG) Generation. It uses an open-set segmenter (OpenSeeD) and leverages Large Multimodal Models (LMMs, like BLIP-2) to auto-regressively predict relations for object pairs that are not limited to a predefined vocabulary.

## Problem Addressed
Traditional PSG models only predict closed-set, predefined object and relation categories. Open-set relation prediction is extremely complex due to the exponential combinations of novel objects and unknown interactions.

## Key Contributions
- **First Open-Set PSG Task:** Formulates the problem of predicting relations beyond predefined sets in panoptic scene graphs.
- **RelQ-Former (Relation Query Transformer):** Uses pair feature extraction queries and cross-attention on masks to efficiently extract interaction features. Uses a relation existence query to filter out irrelevant pairs and save computation.
- **Multimodal Relation Decoder:** Uses an LMM to auto-regressively predict relations. Introduces "generation instructions" and "judgement instructions" to guide the LMM in assigning open-vocabulary predicates.

## Method Summary
1. **Object Segmenter:** OpenSeeD extracts masks, categories, and visual features.
2. **RelQ-Former:** Constructs subject-object pair masks. Uses cross-attention to focus on regions where interactions occur. A selector uses a relation existence query to prune pairs that likely have no relationship.
3. **Multimodal Relation Decoder:** Surviving pairs are fed into a frozen LMM decoder alongside text instructions (e.g., "What are the relations between [subject] and [object]?") to output open-vocabulary relations.

## Relevance to My Thesis
Highly relevant to **Stage 1.5 (Node Canonicalization)** and **Stage 2 (Relation Head)**. OpenPSG provides a blueprint for how to bridge geometric masks with LMM-based open-vocabulary semantics. The idea of using an existence query to filter pairs before running expensive auto-regressive decoding aligns perfectly with the goal of creating an efficient, decoupled thesis architecture.

## Limitations
- Autoregressive decoding via LMMs for every valid object pair is inherently slow, highlighting the computational bottlenecks of directly using LLMs/LMMs in Stage 2 without lightweight adapter heads.
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/ovsgtr/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "ovsgtr"
title: "Expanding Scene Graph Boundaries: Fully Open-vocabulary Scene Graph Generation via Visual-Concept Alignment and Retention"
authors: ["Zuyao Chen", "Jinlin Wu", "Zhen Lei", "Zhaoxiang Zhang", "Chang Wen Chen"]
year: 2024
venue: "arXiv"
doi: ""
url: "https://arxiv.org/abs/2311.10988"
bibtex_key: "chen2024ovsgtr"
primary_topic: "02_scene_graph_generation"
related_topics: ["01_perception_frontends"]
relevance: "high"
status: "read"
tags: ["open-vocabulary", "scene-graph", "visual-concept-alignment", "knowledge-distillation", "relations", "ovsgtr"]
---

# Expanding Scene Graph Boundaries: Fully Open-vocabulary Scene Graph Generation via Visual-Concept Alignment and Retention

## TL;DR
OvSGTR defines four open-vocabulary SGG settings and proposes a DETR-like model that aligns visual node/edge features with text concepts while retaining relation knowledge through distillation.

## Problem Addressed
Open-vocabulary SGG is often object-centric: methods may recognize unseen objects but still assume a closed relation vocabulary. This paper argues that real open-vocabulary SGG must consider unseen nodes, unseen edges, and their combination. It studies the hardest case where both object and relation categories are unseen during training.

## Key Contributions
- Defines four SGG settings: closed-set SGG, open-vocabulary object SGG (OvD-SGG), open-vocabulary relation SGG (OvR-SGG), and fully open-vocabulary object+relation SGG (OvD+R-SGG).
- Proposes OvSGTR, a DETR-like architecture with frozen image backbone, frozen text encoder, and a transformer graph decoder.
- Replaces fixed classifiers with visual-concept alignment for both nodes and edges.
- Uses image-caption data for weakly supervised relation-aware pretraining.
- Adds visual-concept retention via knowledge distillation to reduce catastrophic forgetting of relation concepts during fine-tuning.

## Method Summary
OvSGTR predicts object nodes and relations in a unified transformer setup. Object features are aligned to text embeddings of object concepts. Relation features are built from subject features, object features, and a learned relation query, then aligned to text embeddings of relation concepts.

Because relation-aware visual-language pretraining is scarce, the model parses image captions into pseudo triplets and uses them for weakly supervised pretraining. During SGG fine-tuning, a teacher model pretrained on caption-derived relations distills relation feature space into the student, preserving open-vocabulary relation knowledge when supervised annotations omit novel relation classes.

## Key Results
- In closed-set VG150 SGDet, OvSGTR outperforms listed baselines on R@20/50/100 and mR@20/50/100 while using fewer trainable parameters than some heavier baselines.
- In OvD-SGG, OvSGTR substantially improves novel-object recall over VS3 under the reported split.
- In OvR-SGG, visual-concept retention is critical: without distillation, novel-relation recall is near zero; with retention, OvSGTR reports large gains on novel relations.
- In OvD+R-SGG, the method improves all reported joint, novel-object, and novel-relation metrics, but the task remains significantly harder than the other settings.

## Relevance to My Thesis
High relevance because it clarifies what "open vocabulary" actually means for scene graphs. For the thesis, it is not enough to support open-vocabulary object names from SAM/VLMs if relation families remain closed and brittle. The paper also makes catastrophic forgetting of relation concepts a concrete risk when fine-tuning relation heads on narrow datasets.

For a methodology-first thesis, the four-setting taxonomy is useful in the related work and evaluation design: object openness, relation openness, and combined openness should be discussed separately.

## Key Takeaways / Ideas to Reuse
- Separate open-vocabulary node evaluation from open-vocabulary relation evaluation.
- Align visual relation features to textual relation concepts instead of using only fixed predicate classifiers.
- Use caption-derived pseudo triplets as weak relation supervision when scene graph annotations are scarce.
- Preserve relation concept space during fine-tuning; otherwise novel relation recall can collapse.
- For facade transfer, relation openness may matter more than object openness because spatial/structural predicates are domain-specific.

## Limitations / Open Questions
- Caption-parsed pseudo triplets are noisy and unlocalized.
- The method is still evaluated mostly on VG-style object boxes and relation labels, not mask-grounded PSG/SAM outputs.
- Fully open-vocabulary SGG remains difficult, especially when both objects and relations are novel.
- Visual-concept alignment does not by itself enforce graph consistency, hierarchy, or geometric constraints.

## Related Papers
- [[pgsg_pixels_to_graphs]] is a VLM generation alternative for open-vocabulary SGG.
- [[openpsg]] handles open-set panoptic SGG with LMM relation decoding.
- [[universal_sgg]] also argues for text-centric alignment across modalities.
- [[unbiased_sgg]] remains relevant because open-vocabulary relation learning still interacts with long-tail bias.

## BibTeX
```bibtex
@article{chen2024ovsgtr,
  title={Expanding Scene Graph Boundaries: Fully Open-vocabulary Scene Graph Generation via Visual-Concept Alignment and Retention},
  author={Chen, Zuyao and Wu, Jinlin and Lei, Zhen and Zhang, Zhaoxiang and Chen, Chang Wen},
  journal={arXiv preprint arXiv:2311.10988},
  year={2024}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/panoptic_sgg/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "panoptic_sgg"
title: "Panoptic Scene Graph Generation"
authors: ["Jingkang Yang", "Yi Zhe Ang", "Zujin Guo", "Kaiyang Zhou", "Wayne Zhang", "Ziwei Liu"]
year: 2022
venue: "ECCV"
doi: ""
url: "https://arxiv.org/abs/2207.11247"
bibtex_key: "yang2022panoptic"
primary_topic: "02_scene_graph_generation"
related_topics: []
relevance: "high"
status: "read"
tags: ["panoptic-scene-graph", "dataset", "transformer"]
---

# Panoptic Scene Graph Generation

## TL;DR
This paper introduces the new task of Panoptic Scene Graph Generation (PSG), which replaces rigid, overlapping bounding boxes with precise panoptic segmentation masks. It releases a high-quality dataset of 49k images and provides strong one-stage and two-stage baselines.

## Problem Addressed
Current Scene Graph Generation (SGG) tasks rely on bounding boxes, which are coarse, contain noisy background pixels, and cannot properly ground amorphous background "stuff" (like sky, pavement). Additionally, datasets like Visual Genome suffer from redundant, trivial, and ambiguous predicate labels.

## Key Contributions
- **New Task & Dataset:** The PSG dataset (based on COCO and VG) features 133 object classes (things + stuff) and 56 rigorously defined, non-overlapping predicate classes, all grounded by panoptic masks.
- **Two-Stage Baselines:** Adapted classic SGG methods (IMP, MOTIFS, VCTree) to use Panoptic FPN features.
- **One-Stage Baselines (PSGTR & PSGFormer):** DETR-based architectures. PSGTR predicts triplets directly via queries. PSGFormer explicitly separates object queries and relation queries, linking them via a prompting-like cross-attention matching block.

## Method Summary (PSGFormer)
PSGFormer is a one-stage model that uses two Transformer decoders: one for objects and one for relations. To connect them, it uses a Query Matching Block that acts like a fill-in-the-blank prompt: it uses the relation query to select the most suitable subject and object queries via cosine similarity, forming the final triplet.

## Relevance to My Thesis
This is the foundational paper that shifts the paradigm from BBox-based SGG to Mask-based SGG (PSG). Because the thesis uses SAM 3 (which outputs masks, not boxes), the PSG task formulation is the exact problem the thesis is solving. The concept of explicit bipartite matching between relation queries and object queries in PSGFormer is highly relevant to designing Stage 2 relation heads.

## Limitations
- One-stage models like PSGTR struggle to produce high-quality panoptic masks compared to dedicated segmentation models, justifying the thesis's choice of a Decoupled Two-Stage (DTS) architecture where SAM 3 handles perception perfectly.
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/pgsg_pixels_to_graphs/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "pgsg_pixels_to_graphs"
title: "From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models"
authors: ["Rongjie Li", "Songyang Zhang", "Dahua Lin", "Kai Chen", "Xuming He"]
year: 2024
venue: "CVPR"
doi: ""
url: "https://arxiv.org/abs/2404.00906"
bibtex_key: "li2024pixels"
primary_topic: "02_scene_graph_generation"
related_topics: ["01_perception_frontends", "05_visual_grounding"]
relevance: "high"
status: "read"
tags: ["open-vocabulary", "scene-graph", "vlm", "sequence-generation", "relation-grounding", "pgsg"]
---

# From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models

## TL;DR
PGSG formulates open-vocabulary SGG as image-to-sequence generation with a VLM, then converts generated graph text back into grounded relation triplets.

## Problem Addressed
Classic SGG methods are closed-set and struggle with novel predicates. Many open-vocabulary methods handle only unseen entities or predicate classification with given entity pairs. PGSG targets a more general setting: generating scene graphs with known and novel visual relation triplets directly from pixels.

## Key Contributions
- Reformulates SGG as image-to-graph sequence generation using generative VLMs.
- Introduces scene graph sequence prompts with relation-aware tokens `[ENT]` and `[REL]`.
- Adds a relationship construction module that grounds generated entity tokens to bounding boxes and converts vocabulary tokens into SGG category labels.
- Shows that SGG supervision can transfer back to downstream vision-language tasks such as VQA and visual grounding.

## Method Summary
PGSG uses a BLIP-style VLM. The text decoder is prompted to generate a sequence of subject-predicate-object triplets, written in natural language with special entity and relation markers. A relationship construction module then recovers grounded graph structure:

1. **Entity grounding:** hidden states around `[ENT]` tokens attend back to image features and predict entity boxes.
2. **Category conversion:** vocabulary-space token predictions are mapped into object and predicate category spaces.
3. **VLM transfer:** the SGG-trained VLM can initialize downstream VL tasks.

The model is evaluated in open-vocabulary predicate SGG and also tested under closed-vocabulary SGG, zero-shot triplets, VQA, captioning, and referring expression grounding.

## Key Results
- On PSG, PGSG reports large gains over SGTR-style baselines for whole-category and novel-predicate metrics under the paper's open-vocabulary split.
- On OpenImages V6, PGSG improves mR@100 and R@100 compared with SGTR.
- On downstream tasks, SGG training improves BLIP performance on GQA, especially relation and object questions, and improves RefCOCO-style visual grounding.
- The authors note weaker closed-vocabulary SGG performance compared with high-resolution traditional pipelines, partly due to VLM input resolution and single-stage training.

## Relevance to My Thesis
High relevance for open-vocabulary graph construction and the SAM/VLM interface. PGSG shows a route from foundation-model perception/language representations into explicit scene graph structures. However, it is not a clean fit as the main thesis architecture because it generates graphs through autoregressive text and then re-parses them, which makes geometry and consistency harder to control.

For the thesis, PGSG is best treated as a strong related-work precedent for VLM-based open-vocabulary relations and as evidence that explicit relation supervision improves VL reasoning.

## Key Takeaways / Ideas to Reuse
- Use explicit graph tokens or structured output constraints when asking a VLM to produce graph relations.
- Do not trust text generation alone; generated relations need a grounding/conversion step.
- VLMs can provide open-vocabulary predicate candidates, but final graph assembly should preserve geometry, confidence, and schema constraints.
- For a SAM-based pipeline, use VLM-generated relations as proposals that Stage 3 validates, not as final graph truth.
- Downstream gains on GQA/visual grounding support the argument that explicit graph structure benefits reasoning.

## Limitations / Open Questions
- Autoregressive graph generation can be slow and hard to guarantee valid.
- Generated sequences need heuristic parsing and can duplicate or omit triplets.
- Entity grounding uses boxes rather than masks, and input resolution limits small-object detection.
- Closed-vocabulary SGG results do not dominate high-resolution detector-based methods.

## Related Papers
- [[openpsg]] also uses large multimodal models for open-set panoptic SGG but with pair filtering and relation decoding.
- [[ovsgtr]] studies fully open-vocabulary nodes and relations with visual-concept alignment rather than pure sequence generation.
- [[universal_sgg]] is another cross-modal/open-vocabulary SGG direction.
- [[samjam]] is relevant for the idea of using VLM semantics plus SAM-style grounding/tracking.

## BibTeX
```bibtex
@inproceedings{li2024pixels,
  title={From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models},
  author={Li, Rongjie and Zhang, Songyang and Lin, Dahua and Chen, Kai and He, Xuming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/react_plus_plus/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "react_plus_plus"
title: "REACT++: Efficient Cross-Attention for Real-Time Scene Graph Generation"
authors: ["TODO"]
year: 2026
venue: "arXiv"
doi: ""
url: ""
bibtex_key: "react_plus_plus_2026"
primary_topic: "02_scene_graph_generation"
related_topics: []
relevance: "high"
status: "unread"
tags: ["scene-graph", "cross-attention", "real-time", "sam", "decoupled"]
---

# REACT++: Efficient Cross-Attention for Real-Time Scene Graph Generation

## TL;DR
REACT++ introduces a Decoupled Two-Stage (DTS) architecture utilizing Detection-Anchored Multi-scale Pooling (DAMP) and Cross-Attention Rotary Prototype Embedding (CARPE) for efficient scene graph generation from a frozen perception backbone.

## Problem Addressed
Real-time Scene Graph Generation (SGG) for robotics and embodied agents requires balancing low latency, Object Detection (OD) accuracy, and Relation Prediction (RelPred) accuracy. Current Two-Stage architectures (using Faster R-CNN) are highly accurate but too slow for real-time applications and suffer from computational bottlenecks like RoI Align. Additionally, traditional relation heads struggle to encode the inherent *asymmetry* of visual relations efficiently.

## Key Results
- Achieves the highest inference speed among existing SGG models.
- 20% faster than the previous state-of-the-art real-time SGG model (REACT), while simultaneously gaining a 10% improvement in relation prediction accuracy on average.
- Significantly improves OD accuracy by freezing the YOLO backbone (DTS architecture) rather than co-training a classifier.

## Key Contributions
- **DAMP (Detection-Anchored Multi-Scale Pooling):** A low-cost alternative to RoI Align tailored for one-stage detectors like YOLO.
- **Global Context (AIFI):** Incorporates an Attention-based Intra-scale Feature Interaction module to pool global scene semantics.
- **CARPE (Cross-Attention Rotary Prototype Embedding):** A relation head that replaces rigid MLPs with dynamic cross-attention between objects and an Exponential Moving Average (EMA) predicate prototype bank.
- **Geometric RoPE:** Encodes bounding box geometries (widths, heights, centers, area) directly into the cross-attention queries using Rotary Position Embeddings, avoiding heavy convolutional spatial encoders.

## Method Summary
REACT++ employs a Decoupled Two-Stage (DTS) architecture. In Stage 1, a YOLO backbone extracts visual features using DAMP (a Gaussian-weighted 3x3 neighborhood pooling) and global features using AIFI. The object detector is then frozen. In Stage 2, candidate proposals are filtered using Dynamic Candidate Selection (DCS). Subject and object visual tokens query an EMA bank of semantic predicate prototypes using cross-attention (CARPE). Geometric locations are injected into the attention logits via RoPE, ensuring that spatial biases (e.g., "above", "below") modulate the relation prediction seamlessly.

## Relevance to My Thesis
Provides the explicit blueprint for Stage 2 (Scene Graph Construction). It outlines how to perform feature extraction without RoI Align (using DAMP) and how to formulate relation prediction as a cross-attention problem (using CARPE) rather than relying on legacy MLPs.

## Key Takeaways / Ideas to Reuse
- **DTS Architecture:** Completely freeze the SAM 3 backbone during Stage 2.
- **DAMP Feature Extraction:** Pool features using the spatial indices from Stage 1 rather than cropping.
- **CARPE Relation Head:** Predict edges using cross-attention between node embeddings and learned predicate prototypes.
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/reltr/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "reltr"
title: "RelTR: Relation Transformer for Scene Graph Generation"
authors: ["Yuren Cong", "Michael Ying Yang", "Bodo Rosenhahn"]
year: 2022
venue: "TPAMI 2022"
doi: ""
url: "https://arxiv.org/abs/2201.11460"
bibtex_key: "cong2022reltr"
primary_topic: "02_scene_graph_generation"
related_topics: []
relevance: "medium"
status: "read"
tags: ["scene-graph", "one-stage", "transformer", "detr"]
---

# RelTR: Relation Transformer for Scene Graph Generation

## TL;DR
RelTR is a one-stage, DETR-based Scene Graph Generation model. Unlike traditional two-stage models that first detect objects and then predict relations between them, RelTR directly predicts (subject, predicate, object) triplets from the image using an encoder-decoder transformer architecture.

## Problem Addressed
Traditional two-stage Scene Graph Generation models suffer from computational bottlenecks. Given *n* object proposals, they must evaluate O(*n*²) potential relationships, which is slow and often biased toward common semantic priors rather than actual visual evidence. One-stage models exist for object detection (like DETR), but adapting them to predict complex triplets simultaneously is challenging.

## Key Results
- Demonstrates competitive performance on Visual Genome, Open Images V6, and VRD while maintaining faster inference speeds and fewer parameters than heavy two-stage models.
- Generates sparse scene graphs efficiently using only visual appearance.

## Key Contributions
- **One-Stage Architecture:** Predicts sparse scene graphs directly from visual features without dense O(*n*²) combination of entity proposals.
- **Triplet Decoder:** Uses a novel combination of Coupled Self-Attention (CSA), Decoupled Visual Attention (DVA), and Decoupled Entity Attention (DEA) to process subject and object queries.
- **Set Prediction Loss:** Extends DETR's bipartite matching loss to triplet detection with an IoU-based assignment strategy to handle overlapping or ambiguous relationship proposals.

## Method Summary
RelTR extends the DETR object detection framework. It consists of a feature encoder (CNN), an entity decoder (standard DETR), and a novel **triplet decoder**. The triplet decoder takes in *Nt* pairs of subject and object queries. It uses Coupled Self-Attention so the subject and object queries interact, Decoupled Visual Attention to pull features from the CNN feature map, and Decoupled Entity Attention to pull refined bounding box and class information from the entity decoder. Finally, it uses a feed-forward network to predict the bounding boxes and a convolutional mask head to predict the predicate.

## Relevance to My Thesis
RelTR serves as the standard **baseline** for one-stage end-to-end SGG architectures in the master thesis. While the thesis ultimately focuses on a Decoupled Two-Stage (DTS) architecture (using a frozen SAM 3 backbone for the perception frontend), RelTR is critical for comparison, especially regarding inference speed vs. accuracy trade-offs.

## Key Takeaways / Ideas to Reuse
- **Bipartite Matching for Triplets:** The concept of matching predicted graph edges to ground-truth edges using a bipartite matching loss is highly relevant when training relation heads.
- **Sparse Predictions:** Avoid O(*n*²) exhaustive relation evaluation by treating graph generation as a sparse set prediction problem.

## Limitations / Open Questions
- By generating the graph purely from visual appearance in one stage, RelTR struggles to leverage complex external semantics or LLM-based ontologies (Stage 1.5 Node Canonicalization), making it less flexible than decoupled architectures.
- Object detection accuracy typically drops in one-stage SGG models compared to using a frozen state-of-the-art detector.

## Related Papers
- [[react_plus_plus]] — The Decoupled Two-Stage (DTS) alternative that beats one-stage speeds.
- [[samjam]] — Approaches that use frozen perception models instead.

## BibTeX
```bibtex
@article{cong2022reltr,
  title={RelTR: Relation Transformer for Scene Graph Generation},
  author={Cong, Yuren and Yang, Michael Ying and Rosenhahn, Bodo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/samjam/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "samjam"
title: "SAMJAM: Zero-Shot Video Scene Graph Generation"
authors: ["TODO"]
year: 2025
venue: "arXiv"
doi: ""
url: ""
bibtex_key: "samjam_2025"
primary_topic: "02_scene_graph_generation"
related_topics: []
relevance: "high"
status: "unread"
tags: ["scene-graph", "video", "zero-shot", "sam2", "matching"]
---

# SAMJAM: Zero-Shot Video Scene Graph Generation

## TL;DR
SAMJAM is a zero-shot, training-free pipeline that fuses Vision-Language Models (for semantics) with SAM 2 (for geometric tracking) using discrete bipartite matching algorithms.

## Problem Addressed
Current Vision Language Models (VLMs) like Gemini struggle with the temporal dynamics required for Video Scene Graph Generation (VidSGG). Specifically, they fail to maintain stable object identities across frames (assigning new IDs to the same object) and struggle to produce precise, tight bounding boxes.

## Key Results
- SAMJAM outperforms Gemini 2.0 Flash by 8.33% in mean recall on the EPIC-KITCHENS dataset.
- Achieved a 39.66% mean recall across dynamic video clips (e.g., "pick up cereal bag", "cut bell pepper"), consistently maintaining object IDs across frames where standalone VLMs failed.

## Key Contributions
- A novel 5-stage zero-shot pipeline that successfully marries the semantic reasoning of VLMs (Gemini) with the precise spatiotemporal tracking of VFMs (SAM 2).
- An explicit Intersection-over-Union (IoU) based bipartite matching algorithm to algorithmically ground semantic concepts to pixel-perfect masks.

## Method Summary
The pipeline executes 5 stages per frame: 1) **Mask propagation** using SAM 2's memory bank; 2) **Mask generation & filtering** to find new objects while preventing overlap with propagated masks; 3) **Frame-level SGG** where Gemini independently generates a semantic graph; 4) **Object-mask matching** where Gemini's bounding boxes are mapped to SAM 2's masks using an IoU threshold (>0.1); 5) **Synthesis**, replacing Gemini's unstable IDs and poor boxes with SAM 2's temporally-consistent tracking IDs (tIDs) and precise boundaries.

## Limitations / Open Questions
- **Computational Bottleneck:** The pipeline is heavily bottlenecked by SAM 2's automatic mask generation, taking over 20 seconds per frame.
- **VLM Dependency:** Still vulnerable to large bounding box errors or zero-shot classification mistakes made by Gemini, which can force the matching algorithm to map the semantic node to the wrong geometric mask.

## Relevance to My Thesis
Essential for Stage 1.5 (Node Canonicalization). It proves that Scene Graph Generation can be achieved without joint training by algorithmically aligning semantic nodes (from an external reasoner) directly to temporally consistent, geometrically grounded SAM masks.

## Key Takeaways / Ideas to Reuse
- **Bipartite Matching:** Map abstracted semantic objects directly to explicit SAM mask IDs using IoU and confidence thresholds.
- **Zero-shot Pipeline:** Combines VLM semantic graphs with VFM spatial tracking without the need to fine-tune a coupled object detector.
- **Temporal Consistency:** Rely entirely on SAM 2/3's internal memory tracker to assign temporal object IDs (tIDs), overriding any ID assigned by the reasoning layer.
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/sgg_survey/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "sgg_survey"
title: "Scene Graph Generation: A comprehensive survey"
authors: ["Hongsheng Li", "Guangming Zhu", "Liang Zhang", "Youliang Jiang", "Yixuan Dang", "Haoran Hou", "Peiyi Shen", "Xia Zhao", "Syed Afaq Ali Shah", "Mohammed Bennamoun"]
year: 2024
venue: "Neurocomputing"
doi: ""
url: "https://doi.org/10.1016/j.neucom.2023.127052"
bibtex_key: "li2024scenegraph"
primary_topic: "02_scene_graph_generation"
related_topics: []
relevance: "low"
status: "read"
tags: ["survey", "overview"]
---

# Scene Graph Generation: A comprehensive survey

## TL;DR
A comprehensive 2024 survey of Scene Graph Generation covering 138 papers. It categorizes existing methods based on feature representation (multimodal features, prior info, commonsense knowledge) and feature refinement (message passing, attention mechanisms, visual translation embedding).

## Problem Addressed
With the explosion of SGG research since 2016, there is a need to systematize visual relationship detection methods and understand the underlying mechanisms that successfully handle the long-tailed distribution and intra-class diversity of relations.

## Key Takeaways
- **Feature Representation:** High-quality SGG requires fusing appearance, spatial, semantic, and contextual features. 
- **Priors & Commonsense:** Because relation datasets are long-tailed, models heavily rely on statistical priors (co-occurrences), language priors, and external commonsense knowledge graphs (like ConceptNet) to prune the search space and correct neural predictions.
- **Refinement:** Message passing (via RNNs, GCNs) and attention mechanisms are the standard ways to allow objects and edges to communicate contextual information globally across the image.
- **Evolution:** SGG is expanding from 2D static images into 3D environments and spatio-temporal (video) domains.

## Relevance to My Thesis
As a survey, it provides broad context but lacks specific architectural implementations to adopt. It validates the thesis's choice to use attention mechanisms (like CARPE) over older message-passing architectures (RNNs) for Stage 2 relation prediction.
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/unbiased_sgg/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "unbiased_sgg"
title: "Unbiased Scene Graph Generation from Biased Training"
authors: ["Kaihua Tang", "Yulei Niu", "Jianqiang Huang", "Jiaxin Shi", "Hanwang Zhang"]
year: 2020
venue: "CVPR"
doi: ""
url: "https://arxiv.org/abs/2002.11949"
bibtex_key: "tang2020unbiased"
primary_topic: "02_scene_graph_generation"
related_topics: ["03_scene_graph_reasoning", "04_neurosymbolic_reasoning"]
relevance: "high"
status: "read"
tags: ["scene-graph", "bias", "causal-inference", "mean-recall"]
---

# Unbiased Scene Graph Generation from Biased Training

## TL;DR
This paper introduces a causal debiasing framework for SGG that removes harmful predicate bias with Total Direct Effect while preserving useful contextual priors.

## Problem Addressed
Scene graph generators trained on Visual Genome tend to predict frequent, generic predicates such as "on", "near", and "has" instead of more informative relations. Standard debiasing methods can harm useful object-context priors, so the paper distinguishes good context bias from bad long-tail predicate bias.

## Key Contributions
- Frames SGG predicate bias through a causal graph rather than ordinary likelihood prediction.
- Uses counterfactual inference and Total Direct Effect (TDE) to subtract harmful contextual bias at inference time.
- Provides a Scene Graph Diagnosis toolkit with bias-sensitive metrics.
- Emphasizes mean Recall@K and zero-shot relationship retrieval as better diagnostics than ordinary Recall@K.

## Method Summary
The framework trains a normal biased SGG model, then performs counterfactual inference over the learned causal graph. It compares the predicate prediction in the real setting against a counterfactual setting where the visual content effect is controlled. The final predicate score uses Total Direct Effect to reduce the direct contribution of dataset bias while retaining useful visual/contextual signal. The method is model-agnostic and can be applied to architectures such as MOTIFS, VTransE, and VCTree.

## Key Results
- On Visual Genome, TDE improves mean Recall@K across multiple SGG backbones and tasks.
- The paper evaluates Predicate Classification, Scene Graph Classification, and Scene Graph Detection.
- It also evaluates zero-shot relationship retrieval and sentence-to-graph retrieval to expose graph-level bias.

## Relevance to My Thesis
This is one of the most important papers for the relation-prediction phase. A SAM 3 frontend may produce cleaner nodes than classical detectors, but predicate prediction can still be biased. For facade interpretation, the thesis should explicitly test whether the graph layer predicts meaningful relations like "above", "aligned-with", "part-of", or "supports" rather than collapsing into generic adjacency predicates.

## Key Takeaways / Ideas to Reuse
- Use mean Recall@K or per-predicate analysis, not only aggregate Recall@K.
- Separate visual/geometric evidence from semantic priors in the relation layer.
- Add zero-shot or low-frequency relation tests if the relation vocabulary includes facade-specific predicates.
- Consider causal or rule-based post-processing when learned relation heads overuse common predicates.

## Limitations / Open Questions
- The method is designed around Visual Genome-style closed vocabularies.
- It debiases predicate scoring but does not solve open-vocabulary segmentation, mask canonicalization, or graph schema design.
- Causal debiasing may be harder to justify when the thesis uses hand-defined geometric relations rather than learned VG predicates.

## Related Papers
- [[neural_motifs]] — demonstrates the strength of object-pair priors and motivates the bias problem.
- [[visual_genome]] — source of the long-tail relation distribution.
- [[panoptic_sgg]] — later mask-grounded formulation where predicate bias remains relevant.
- [[openpsg]] — open-set PSG setting where vocabulary bias appears differently.

## BibTeX
```bibtex
@inproceedings{tang2020unbiased,
  title={Unbiased Scene Graph Generation from Biased Training},
  author={Tang, Kaihua and Niu, Yulei and Huang, Jianqiang and Shi, Jiaxin and Zhang, Hanwang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={3716--3725},
  year={2020}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/universal_sgg/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "universal_sgg"
title: "Universal Scene Graph Generation"
authors: ["Shengqiong Wu", "Hao Fei", "Tat-Seng Chua"]
year: 2025
venue: "CVPR 2025"
doi: ""
url: ""
bibtex_key: "wu2025usg"
primary_topic: "02_scene_graph_generation"
related_topics: []
relevance: "medium"
status: "unread"
tags: ["scene-graph", "universal", "generation"]
---

# Universal Scene Graph Generation

## TL;DR
USG is a modality-invariant framework that parses images, video, text, and 3D data utilizing SAM 2 pseudo-masks and a Text-Centric Scene Contrasting Learning Mechanism.

## Problem Addressed
Current Scene Graph Generation (SGG) research is heavily siloed into single modalities (Image SGG, Video SGG, Text SGG, 3D SGG). This prevents the use of complementary strengths: images provide visual detail, text provides abstract/relational descriptions, video provides temporal dynamics, and 3D provides spatial depth. Generating a "Universal" Scene Graph is difficult due to the "modality gap" (aligning the exact same object across fundamentally different feature spaces) and severe domain biases (e.g., 3D data is mostly indoor, video is action-heavy).

## Key Contributions
- Introduces **Universal Scene Graph (USG)**, a new representation capable of fully characterizing scenes from *any* combination of modality inputs.
- Proposes **USG-Par**, an end-to-end parser architecture that handles cross-modal object alignment.
- Introduces a **text-centric scene contrasting learning mechanism** to mitigate domain imbalances by forcing visual objects/relations to align with stable text-space embeddings.

## Method Summary
The USG-Par model consists of five main modules:
1. **Modality-Specific Encoders:** Uses OpenCLIP (text), CLIP-ConvNeXt (image/video), and Point-BERT (3D) to extract contextual features.
2. **Shared Mask Decoder:** A Mask2Former-based architecture generates object queries and extracts features implicitly across modalities.
3. **Object Associator:** Solves the modality gap by projecting object queries into each other's feature spaces to compute cosine similarities, filtering them to construct a cross-modal association matrix.
4. **Relation Proposal Constructor (RPC):** Uses a two-way relation-aware cross-attention mechanism to identify and filter the most promising subject-object pairs, preventing O(N^2) pairwise explosions.
5. **Relation Decoder:** Predicts the final predicate by applying cross-attention between relation queries and the contextualized input features.

## Key Results
- Demonstrates that USG offers a far more powerful and comprehensive scene representation than standalone SGs.
- USG-Par achieves significant performance improvements not only in multimodal parsing but also in *single-modality* scene graph parsing benchmarks.
- Successfully generalizes to unseen scene domains and unseen modality combinations due to the text-centric contrastive learning strategy.

## Relevance to My Thesis
Crucial for bridging the modality gap between language and vision in the Node Canonicalization stage (Stage 1.5). It provides the exact mechanism for "Contrastive Anchoring"—forcing visual representations from SAM to align with text-based ontologies from a VLM.

## Key Takeaways / Ideas to Reuse
- **Object Associator:** To map SAM 3 geometric masks to text-based ontology nodes, project both into a shared latent space and calculate cosine similarity to build an association matrix.
- **Text-Centric Anchoring:** Because text is the most "domain-agnostic" modality, you should anchor visual features to textual representations (rather than the other way around) to prevent domain drift.
- **Relation Proposal Filtering:** Don't compute every N x (N-1) relation pair; use an attention-based filter to only test top-k likely pairs.

## Limitations / Open Questions
- **Scalability of the Associator:** Projecting features back and forth across every modality can be computationally heavy.
- **Text Bias:** Heavily relies on the text modality as the "ground truth" anchor for contrastive learning. If the textual ontology is flawed or overly simple, visual nuances might be suppressed.

## Related Papers
- [[fair_psgg]] — Related SGG benchmark work
- [[iterative_message_passing]] — Foundational SGG method

## BibTeX
```bibtex
@inproceedings{wu2025usg,
  title={Universal Scene Graph Generation},
  author={Wu, Shengqiong and Fei, Hao and Chua, Tat-Seng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END

CONTEXT SOURCE: context/literature/02_scene_graph_generation/visual_commonsense/summary.md
CONTEXT REASON: linked-from:roter_faden_related_work.md
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT START
---
paper_id: "visual_commonsense"
title: "Learning Visual Commonsense for Robust Scene Graph Generation"
authors: ["Alireza Zareian", "Zhecan Wang", "Haoxuan You", "Shih-Fu Chang"]
year: 2020
venue: "ECCV"
doi: ""
url: "https://arxiv.org/abs/2006.09623"
bibtex_key: "zareian2020visualcommonsense"
primary_topic: "02_scene_graph_generation"
related_topics: ["03_scene_graph_reasoning"]
relevance: "high"
status: "read"
tags: ["scene-graph", "commonsense", "robustness", "transformer", "graph-refinement", "fusion"]
---

# Learning Visual Commonsense for Robust Scene Graph Generation

## TL;DR
This paper learns a separate visual-commonsense model over scene graphs and fuses it with perception outputs, directly addressing when to trust visual evidence versus graph plausibility.

## Problem Addressed
SGG models often produce nonsensical graph compositions under visual uncertainty, but commonsense priors can also overcorrect rare but real situations. Existing methods tend to mix perception and commonsense inside one model, making it hard to study or control their relative influence. The paper asks how to learn graph-level commonsense separately and fuse it with perception robustly.

## Key Contributions
- Formalizes visual commonsense as auto-encoding perturbed scene graphs.
- Introduces Global-Local Attention Transformer (GLAT), which combines global transformer attention with local graph-neighborhood attention.
- Trains a graph-commonsense model with BERT-style masking over scene graph entities and predicates.
- Proposes a fusion module that weighs perception and commonsense predictions based on confidence.
- Shows the method can be stacked on top of several SGG models to improve robustness.

## Method Summary
The method separates the system into two models:

1. **Perception model:** an ordinary SGG model that predicts scene graph nodes and predicates from images.
2. **Commonsense model:** GLAT, trained only on graph structure, predicts masked or perturbed graph components usi
---FUSION_BOUNDARY_f29af108e7a6a78a65893037--- CONTEXT END