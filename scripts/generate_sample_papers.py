"""
scripts/generate_sample_papers.py
Generates 50 synthetic research papers, inserts into MongoDB Atlas,
and embeds them into ChromaDB so the chatbot works immediately.

Usage:
    python scripts/generate_sample_papers.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
for d in ["./data/papers", "./data/chromadb", "./data/logs"]:
    Path(d).mkdir(parents=True, exist_ok=True)

from dotenv import load_dotenv
load_dotenv()

import uuid
import random
from datetime import datetime, timedelta
from collections import Counter

from loguru import logger
from backend.database import DatabaseManager, get_papers_collection, get_users_collection


# ─────────────────────────────────────────────────────────────
# 50 Realistic Research Papers across 10 Departments
# ─────────────────────────────────────────────────────────────
PAPERS = [
    # ── Computer Science ──────────────────────────────────────
    {"title": "Transformer-Based Code Generation with Syntax-Aware Attention",
     "abstract": "We introduce SyntaxFormer, a transformer model leveraging abstract syntax trees (AST) as structural priors during code generation. Evaluated on HumanEval and MBPP, SyntaxFormer achieves 82.4% pass@1, outperforming GPT-4 Turbo on algorithmic tasks.",
     "keywords": ["code generation", "transformers", "AST", "LLM", "software engineering"],
     "department": "Computer Science", "authors": ["Dr. Alice Kim", "Prof. David Lee", "Ravi Subramaniam"],
     "publication_year": 2024, "venue": "ACL 2024"},

    {"title": "Federated Learning with Heterogeneous Data Distributions",
     "abstract": "FedAdapt is an adaptive aggregation algorithm for federated learning under statistical heterogeneity. It reduces communication rounds by 40% while achieving within 1% of centralized training accuracy on CIFAR-10 across 100 clients.",
     "keywords": ["federated learning", "distributed ML", "privacy", "optimization"],
     "department": "Computer Science", "authors": ["Dr. Alice Kim", "Dr. Bob Johnson", "Charlie Student"],
     "publication_year": 2024, "venue": "NeurIPS 2024"},

    {"title": "Graph Attention Networks for Knowledge Base Completion",
     "abstract": "GAT-KBC jointly models entity and relation embeddings using graph attention networks for knowledge base completion. Achieves state-of-the-art MRR of 0.38 on FB15k-237 and 0.51 on WN18RR.",
     "keywords": ["knowledge graphs", "graph neural networks", "entity embedding"],
     "department": "Computer Science", "authors": ["Prof. David Lee", "Dr. Priya Sharma", "Alice Lin"],
     "publication_year": 2023, "venue": "EMNLP 2023"},

    {"title": "Differential Privacy in Large Language Model Fine-Tuning",
     "abstract": "DP-LoRA achieves epsilon=3 privacy guarantee while retaining 94% of full fine-tuning performance on GLUE, enabling safe fine-tuning on sensitive healthcare and legal corpora.",
     "keywords": ["differential privacy", "LLM", "fine-tuning", "LoRA", "privacy"],
     "department": "Computer Science", "authors": ["Dr. Priya Sharma", "Dr. Alice Kim", "Prof. Helen Zhang"],
     "publication_year": 2024, "venue": "ICLR 2024"},

    {"title": "Multi-Agent Reinforcement Learning for Traffic Signal Coordination",
     "abstract": "A decentralized MARL system using QMIX for city-scale traffic signal optimization. Reduces average vehicle delay by 34% and fuel consumption by 21% compared to fixed-time control over a 50-intersection grid.",
     "keywords": ["multi-agent RL", "traffic optimization", "QMIX", "smart city"],
     "department": "Computer Science", "authors": ["Prof. David Lee", "Diana Student", "Dr. Alice Kim"],
     "publication_year": 2024, "venue": "AAAI 2024"},

    {"title": "Zero-Shot Cross-Lingual Transfer for 12 Indian Languages",
     "abstract": "IndicBERT-v2 adapts multilingual BERT for NLP tasks across Hindi, Tamil, Telugu, Bengali and 8 other Indian languages. Achieves 91% classification accuracy on XNLI-Indic, outperforming XLM-R by 5.3 points average.",
     "keywords": ["multilingual NLP", "transfer learning", "Indian languages", "BERT"],
     "department": "Computer Science", "authors": ["Dr. Priya Sharma", "Prof. David Lee", "Dr. Alice Kim"],
     "publication_year": 2023, "venue": "ACL 2023"},

    {"title": "Neural Architecture Search for Edge Inference",
     "abstract": "EfficientNAS is a hardware-aware NAS framework for edge devices. Searched models achieve Pareto-optimal accuracy-latency tradeoffs on ImageNet, running at 120 FPS on Raspberry Pi 4 with 73.1% top-1 accuracy.",
     "keywords": ["neural architecture search", "edge computing", "model compression"],
     "department": "Computer Science", "authors": ["Charlie Student", "Dr. Alice Kim", "Prof. Anil Kumar"],
     "publication_year": 2022, "venue": "CVPR 2022"},

    {"title": "Contrastive Learning for Code Clone Detection",
     "abstract": "CodeContrast uses supervised contrastive learning to detect semantically equivalent code clones across 6 programming languages. Achieves 97.2% F1 on BigCloneBench and identifies zero-day vulnerabilities in 3 open-source projects.",
     "keywords": ["contrastive learning", "code clone", "software security"],
     "department": "Computer Science", "authors": ["Dr. Bob Johnson", "Prof. David Lee", "Ravi Subramaniam"],
     "publication_year": 2023, "venue": "ISSTA 2023"},

    # ── Data Science ──────────────────────────────────────────
    {"title": "Explainable AI for Credit Scoring in Microfinance",
     "abstract": "An interpretable ML pipeline combining gradient-boosted trees with SHAP for credit scoring in rural microfinance. Achieves AUC 0.91 on 450,000 loan records with regulatorily compliant explanations in three Indian languages.",
     "keywords": ["explainable AI", "credit scoring", "SHAP", "fairness", "fintech"],
     "department": "Data Science", "authors": ["Diana Student", "Prof. Helen Zhang", "Dr. Alice Kim"],
     "publication_year": 2022, "venue": "ACM FAccT 2022"},

    {"title": "Adaptive Learning Systems Using Knowledge Graphs and LLMs",
     "abstract": "An intelligent tutoring system combining domain knowledge graphs with GPT-4 for personalized STEM education. A 6-month study with 2,400 students showed 28% improvement in exam scores over traditional e-learning.",
     "keywords": ["adaptive learning", "knowledge graphs", "education technology", "LLM"],
     "department": "Data Science", "authors": ["Prof. Helen Zhang", "Diana Student", "Dr. Priya Sharma"],
     "publication_year": 2024, "venue": "AIED 2024"},

    {"title": "Time Series Anomaly Detection in Industrial IoT",
     "abstract": "AnomalyBERT is a transformer-based model for multivariate time series anomaly detection in manufacturing. Achieves F1 of 0.896 on SWAT and 0.851 on WADI, reducing false alarms by 62% over LSTM baselines.",
     "keywords": ["anomaly detection", "time series", "transformer", "IoT", "manufacturing"],
     "department": "Data Science", "authors": ["Prof. Helen Zhang", "Dr. James Wilson", "Diana Student"],
     "publication_year": 2023, "venue": "KDD 2023"},

    {"title": "Causal Inference for Treatment Effect Estimation in Clinical Trials",
     "abstract": "Doubly-robust estimators and TMLE applied to 120,000 patient records reveal a 23% reduction in readmission rates for a novel hypertension intervention, demonstrating unbiased treatment effect estimation from observational data.",
     "keywords": ["causal inference", "treatment effect", "TMLE", "clinical trials"],
     "department": "Data Science", "authors": ["Dr. Anita Gupta", "Prof. Helen Zhang", "Dr. Emma Brown"],
     "publication_year": 2023, "venue": "Journal of Causal Inference"},

    {"title": "Graph-Based Recommendation Systems with Temporal Dynamics",
     "abstract": "TemporalGNN extends graph collaborative filtering with time-aware attention for evolving user preferences. Achieves 12% NDCG improvement over LightGCN on MovieLens-25M and handles cold-start with 89% accuracy.",
     "keywords": ["recommendation systems", "GNN", "temporal modeling", "collaborative filtering"],
     "department": "Data Science", "authors": ["Diana Student", "Charlie Student", "Prof. Helen Zhang"],
     "publication_year": 2024, "venue": "RecSys 2024"},

    {"title": "Self-Supervised Learning for Satellite Image Time Series",
     "abstract": "SatMAE is a masked autoencoder pre-trained on 1.2M Sentinel-2 time series. Fine-tuned on crop mapping and flood detection, it outperforms supervised baselines by 8.4% and 6.1% with only 100 labeled samples.",
     "keywords": ["self-supervised learning", "satellite imagery", "remote sensing", "masked autoencoder"],
     "department": "Data Science", "authors": ["Prof. Helen Zhang", "Dr. Fatima Hassan", "Ravi Subramaniam"],
     "publication_year": 2024, "venue": "IGARSS 2024"},

    # ── Electronics Engineering ───────────────────────────────
    {"title": "Real-Time Object Detection for Autonomous Vehicles on Edge Hardware",
     "abstract": "EdgeYOLO achieves 47 FPS on NVIDIA Jetson Orin with 89.3% mAP on KITTI benchmark, enabling real-time pedestrian and obstacle detection without cloud dependency.",
     "keywords": ["object detection", "autonomous vehicles", "edge computing", "YOLO"],
     "department": "Electronics Engineering", "authors": ["Dr. Bob Johnson", "Prof. Anil Kumar", "Dr. Fatima Hassan"],
     "publication_year": 2023, "venue": "CVPR 2023"},

    {"title": "IoT Sensor Fusion for Smart Grid Demand Forecasting",
     "abstract": "A multi-modal IoT architecture combining smart meters, weather, and social calendars for day-ahead electricity demand forecasting. LSTM-Transformer hybrid achieves 1.8% MAPE across 3 Indian state grids.",
     "keywords": ["IoT", "sensor fusion", "smart grid", "demand forecasting", "energy"],
     "department": "Electronics Engineering", "authors": ["Dr. Fatima Hassan", "Prof. Anil Kumar", "Dr. James Wilson"],
     "publication_year": 2023, "venue": "IEEE Transactions on Smart Grid"},

    {"title": "5G mmWave Beamforming with Deep Reinforcement Learning",
     "abstract": "DRL-BF applies deep Q-learning to adaptive beamforming in 5G mmWave networks, reducing beam alignment overhead by 55% and improving spectral efficiency by 31% at up to 60 km/h user velocity.",
     "keywords": ["5G", "beamforming", "deep reinforcement learning", "mmWave"],
     "department": "Electronics Engineering", "authors": ["Prof. Anil Kumar", "Dr. Fatima Hassan", "Karthik Rajan"],
     "publication_year": 2024, "venue": "IEEE GLOBECOM 2024"},

    {"title": "FPGA Acceleration for Spiking Neural Networks",
     "abstract": "A reconfigurable FPGA accelerator for spiking neural networks targeting neuromorphic workloads. Achieves 5.2x energy efficiency improvement over GPU while matching accuracy on N-MNIST and DVS-CIFAR10.",
     "keywords": ["FPGA", "spiking neural networks", "neuromorphic computing", "hardware acceleration"],
     "department": "Electronics Engineering", "authors": ["Dr. Bob Johnson", "Prof. Anil Kumar", "Charlie Student"],
     "publication_year": 2022, "venue": "DATE 2022"},

    {"title": "Millimeter-Wave Radar for Vital Signs Monitoring",
     "abstract": "A 77 GHz FMCW radar system for non-contact respiration and heart rate monitoring through clothing. Achieves 97.8% respiration and 94.2% heart rate accuracy at 1.5m distance across 48 subjects.",
     "keywords": ["mmWave radar", "vital signs", "non-contact sensing", "FMCW", "healthcare"],
     "department": "Electronics Engineering", "authors": ["Dr. Fatima Hassan", "Dr. Emma Brown", "Prof. Anil Kumar"],
     "publication_year": 2023, "venue": "IEEE Sensors Journal"},

    {"title": "Deep Reinforcement Learning for Robot Manipulation",
     "abstract": "SAC with domain randomization trains robot arm policies for grasping novel objects in cluttered environments. Zero-shot transfer to 150 unseen object categories achieves 83.7% grasp success, enabling factory pick-and-place deployment.",
     "keywords": ["reinforcement learning", "robot manipulation", "SAC", "domain randomization"],
     "department": "Electronics Engineering", "authors": ["Prof. Anil Kumar", "Dr. Bob Johnson", "Charlie Student", "Dr. Alice Kim"],
     "publication_year": 2024, "venue": "ICRA 2024"},

    # ── Biomedical Engineering ────────────────────────────────
    {"title": "Transformer Architecture for 3D Medical Image Segmentation",
     "abstract": "MedSeg3D introduces a hierarchical vision transformer with 3D patch tokenization. On BraTS 2023 and TCIA-Lung, it achieves Dice scores of 0.912 and 0.887, surpassing 3D U-Net by 4.2%.",
     "keywords": ["transformer", "medical imaging", "3D segmentation", "brain tumor"],
     "department": "Biomedical Engineering", "authors": ["Dr. Sarah Chen", "Dr. Raj Patel", "Prof. Maria Santos"],
     "publication_year": 2023, "venue": "MICCAI 2023"},

    {"title": "Graph Neural Networks for Drug-Drug Interaction Prediction",
     "abstract": "DDI-GNN models molecular graphs with heterogeneous GNN for predicting adverse drug-drug interactions. Achieves 96.2% AUROC on DrugBank 5.1.10 and identifies 12 potentially dangerous interactions not previously flagged.",
     "keywords": ["drug interactions", "GNN", "pharmacology", "drug safety", "bioinformatics"],
     "department": "Biomedical Engineering", "authors": ["Dr. Raj Patel", "Prof. Li Wei", "Dr. Emma Brown"],
     "publication_year": 2024, "venue": "Nature Machine Intelligence"},

    {"title": "Non-Invasive Wearable Glucose Monitoring via NIR Spectroscopy",
     "abstract": "A wrist-worn NIR spectroscopy biosensor for continuous non-invasive blood glucose monitoring. Validated on 120 diabetic subjects over 30 days, achieving MARD of 8.7% and 94% ISO 15197 clinical accuracy.",
     "keywords": ["biosensor", "glucose monitoring", "NIR spectroscopy", "wearable", "diabetes"],
     "department": "Biomedical Engineering", "authors": ["Dr. Emma Brown", "Dr. Raj Patel", "Prof. Susan Park"],
     "publication_year": 2023, "venue": "Biosensors and Bioelectronics"},

    {"title": "EEG-Based Motor Imagery Classification for BCI Systems",
     "abstract": "EEGNet-Attention is a compact CNN with attention for motor imagery classification. Achieves 85.3% average accuracy on BCI Competition IV Dataset 2a with only 3,400 parameters, enabling real-time BCI on microcontrollers.",
     "keywords": ["EEG", "brain-computer interface", "motor imagery", "attention", "neural signal processing"],
     "department": "Biomedical Engineering", "authors": ["Dr. Sarah Chen", "Prof. Maria Santos", "Dr. Bob Johnson"],
     "publication_year": 2022, "venue": "IEEE TNSRE 2022"},

    {"title": "Deep Learning for Retinal Disease Detection from OCT Scans",
     "abstract": "RetinalNet applies multi-scale CNN with class activation mapping for automated detection of diabetic macular edema, AMD, and glaucoma. Validated on 108,312 scans from 3 hospitals: 97.4% sensitivity, 96.1% specificity.",
     "keywords": ["retinal disease", "OCT", "deep learning", "diabetic retinopathy", "glaucoma"],
     "department": "Biomedical Engineering", "authors": ["Dr. Raj Patel", "Dr. Emma Brown", "Prof. Susan Park"],
     "publication_year": 2024, "venue": "The Lancet Digital Health"},

    {"title": "Biomechanical Analysis of Gait Using Wearable IMUs and Deep Learning",
     "abstract": "A gait analysis pipeline using 7 body-worn IMUs and bidirectional LSTM to classify 12 gait disorders and predict fall risk. 91.4% sensitivity and 93.7% specificity achieved in a 312-patient clinical study.",
     "keywords": ["gait analysis", "IMU", "wearable sensors", "fall prediction", "biomechanics"],
     "department": "Biomedical Engineering", "authors": ["Dr. Sarah Chen", "Dr. James Wilson", "Prof. Susan Park"],
     "publication_year": 2022, "venue": "IEEE Journal of Biomedical and Health Informatics"},

    {"title": "Multimodal AI for Clinical Decision Support in Emergency Medicine",
     "abstract": "A multimodal CDS system combining EHR data, chest X-rays, and clinical notes for sepsis prediction. Achieves AUROC 0.94 with 6-hour lead time, reducing 30-day mortality by 18% in a prospective trial.",
     "keywords": ["clinical decision support", "multimodal AI", "sepsis", "EHR", "chest X-ray"],
     "department": "Biomedical Engineering", "authors": ["Dr. Emma Brown", "Dr. Sarah Chen", "Prof. Helen Zhang", "Dr. Alice Kim"],
     "publication_year": 2024, "venue": "Nature Medicine"},

    # ── Physics ───────────────────────────────────────────────
    {"title": "Variational Quantum Eigensolvers for Molecular Energy Estimation",
     "abstract": "VQE circuits for ground-state energies of H2, LiH, and BeH2 on IBM quantum hardware. With zero-noise extrapolation, circuits achieve chemical accuracy (<1 kcal/mol) on a 5-qubit processor.",
     "keywords": ["quantum computing", "VQE", "quantum chemistry", "molecular simulation", "NISQ"],
     "department": "Physics", "authors": ["Prof. Robert Feynman Jr", "Dr. Maria Santos", "Arjun Nair"],
     "publication_year": 2023, "venue": "Physical Review Letters"},

    {"title": "Quantum Error Correction with Surface Codes on Superconducting Qubits",
     "abstract": "Fault-tolerant logical qubit encoding using distance-3 and distance-5 surface codes on a 49-qubit superconducting processor. Logical error rate of 2.9e-3 per cycle — below the threshold for scalable quantum computing.",
     "keywords": ["quantum error correction", "surface codes", "superconducting qubits", "fault tolerance"],
     "department": "Physics", "authors": ["Prof. Robert Feynman Jr", "Arjun Nair"],
     "publication_year": 2024, "venue": "Nature Physics"},

    {"title": "Machine Learning Potentials for Molecular Dynamics Simulation",
     "abstract": "A DeepMD potential for water and ionic solutions trained on 2M DFT calculations. Reproduces experimental RDFs and diffusion coefficients within 3% error while running 1000x faster than ab initio MD.",
     "keywords": ["ML potentials", "molecular dynamics", "DFT", "water simulation", "force fields"],
     "department": "Physics", "authors": ["Dr. Maria Santos", "Prof. Robert Feynman Jr", "Dr. Anita Gupta"],
     "publication_year": 2023, "venue": "Journal of Chemical Theory and Computation"},

    {"title": "Quantum Machine Learning for Financial Portfolio Optimization",
     "abstract": "QAOA-enhanced portfolio optimization on a 24-qubit processor achieves 11% better Sharpe ratio compared to classical solvers for 20-asset portfolios within equal computation budget.",
     "keywords": ["quantum machine learning", "portfolio optimization", "QAOA", "quantum finance"],
     "department": "Physics", "authors": ["Prof. Robert Feynman Jr", "Prof. Helen Zhang", "Arjun Nair"],
     "publication_year": 2024, "venue": "Quantum Science and Technology"},

    # ── Mechanical Engineering ────────────────────────────────
    {"title": "Topology Optimization for Additive Manufacturing with ML",
     "abstract": "Physics-informed neural networks combined with SIMP topology optimization for lightweight 3D-printed metal structures. Deep learning surrogate reduces optimization time by 15x within 2% compliance of full FEM.",
     "keywords": ["topology optimization", "additive manufacturing", "PINN", "structural design"],
     "department": "Mechanical Engineering", "authors": ["Dr. James Wilson", "Prof. Kavitha Reddy", "Raj Suresh"],
     "publication_year": 2023, "venue": "Computer Methods in Applied Mechanics"},

    {"title": "Predictive Maintenance for CNC Machines Using Vibration Signals",
     "abstract": "RUL prediction for CNC spindle bearings using vibration and acoustic emission. Bidirectional LSTM with attention achieves MAE of 12.3 hours on PHM 2010, enabling 5-day advance maintenance scheduling.",
     "keywords": ["predictive maintenance", "CNC machine", "vibration analysis", "LSTM", "RUL"],
     "department": "Mechanical Engineering", "authors": ["Dr. James Wilson", "Prof. Kavitha Reddy", "Dr. Fatima Hassan"],
     "publication_year": 2022, "venue": "Mechanical Systems and Signal Processing"},

    {"title": "CFD-ML Hybrid for Turbulent Flow Prediction in Heat Exchangers",
     "abstract": "CNN integrated with RANS solvers for rapid turbulent flow prediction in shell-and-tube heat exchangers. CNN-CFD hybrid achieves 98.1% accuracy vs full CFD at 50x computational speedup.",
     "keywords": ["CFD", "machine learning", "turbulent flow", "heat exchangers", "fluid dynamics"],
     "department": "Mechanical Engineering", "authors": ["Prof. Kavitha Reddy", "Dr. James Wilson", "Raj Suresh"],
     "publication_year": 2024, "venue": "International Journal of Heat and Mass Transfer"},

    # ── Civil Engineering ─────────────────────────────────────
    {"title": "Structural Health Monitoring Using MEMS Sensors and Deep Learning",
     "abstract": "Wireless MEMS accelerometers on a 120m cable-stayed bridge with 1D-CNN for real-time damage detection. 98.4% accuracy and 2m spatial fault localization resolution.",
     "keywords": ["structural health monitoring", "MEMS sensors", "deep learning", "bridge"],
     "department": "Civil Engineering", "authors": ["Dr. James Wilson", "Prof. Kavitha Reddy", "Dr. Bob Johnson"],
     "publication_year": 2022, "venue": "Engineering Structures"},

    {"title": "Smart Concrete with Embedded Carbon Nanotubes for Crack Detection",
     "abstract": "Self-sensing CNT composite concrete enables continuous crack propagation monitoring via electrical impedance. 0.01 mm crack width sensitivity achieved with 94% detection rate.",
     "keywords": ["smart concrete", "carbon nanotubes", "crack detection", "structural monitoring"],
     "department": "Civil Engineering", "authors": ["Prof. Kavitha Reddy", "Dr. Anita Gupta", "Dr. James Wilson"],
     "publication_year": 2023, "venue": "Cement and Concrete Composites"},

    {"title": "Seismic Vulnerability Assessment Using Machine Learning",
     "abstract": "Gradient boosted trees and neural networks on 45,000 buildings from the 2015 Nepal earthquake. Ensemble model achieves 91.2% accuracy in 4-class damage prediction, enabling rapid post-disaster triage.",
     "keywords": ["seismic vulnerability", "machine learning", "earthquake", "structural damage"],
     "department": "Civil Engineering", "authors": ["Dr. James Wilson", "Dr. Anita Gupta", "Prof. Kavitha Reddy"],
     "publication_year": 2023, "venue": "Natural Hazards and Earth System Sciences"},

    # ── Chemistry ─────────────────────────────────────────────
    {"title": "TiO2-Graphene Photocatalysts for Solar Wastewater Treatment",
     "abstract": "TiO2-rGO composites via hydrothermal method achieve 96.8% degradation of ciprofloxacin within 60 min under solar illumination at 10 mg/L concentration.",
     "keywords": ["photocatalysis", "TiO2", "graphene oxide", "wastewater treatment", "solar energy"],
     "department": "Chemistry", "authors": ["Prof. Kavitha Reddy", "Dr. Anita Gupta", "Dr. James Wilson"],
     "publication_year": 2022, "venue": "Applied Catalysis B: Environmental"},

    {"title": "MOF-Based CO2 Capture for Industrial Flue Gas Treatment",
     "abstract": "Novel zirconium-based MOF (Zr-NDC) with record CO2 uptake of 8.2 mmol/g at 1 bar and 298K. Retains 98.7% CO2 capacity over 10 regeneration cycles, demonstrating practical carbon capture applicability.",
     "keywords": ["MOF", "CO2 capture", "carbon sequestration", "porous materials", "climate change"],
     "department": "Chemistry", "authors": ["Dr. Anita Gupta", "Prof. Kavitha Reddy"],
     "publication_year": 2023, "venue": "Journal of the American Chemical Society"},

    {"title": "Electrochemical Nitrogen Fixation at Ambient Conditions",
     "abstract": "Boron-doped graphene electrocatalyst for ambient electrochemical nitrogen fixation. Faradaic efficiency of 34.6% and NH3 yield of 52.4 ug/h/mg at -0.2 V vs RHE — highest reported at room temperature.",
     "keywords": ["nitrogen fixation", "electrocatalysis", "ammonia synthesis", "graphene"],
     "department": "Chemistry", "authors": ["Dr. Anita Gupta", "Prof. Kavitha Reddy", "Dr. Maria Santos"],
     "publication_year": 2024, "venue": "Nature Catalysis"},

    {"title": "Carbon Nanotube Biosensors for Real-Time Pathogen Detection",
     "abstract": "SWCNT field-effect transistors functionalized with SARS-CoV-2 spike antibodies for point-of-care detection. Detects viral antigen at 1 fg/mL in 8 minutes with 99.1% sensitivity and 98.6% specificity.",
     "keywords": ["carbon nanotubes", "biosensor", "COVID-19", "pathogen detection", "FET"],
     "department": "Chemistry", "authors": ["Dr. Anita Gupta", "Dr. Emma Brown", "Prof. Kavitha Reddy"],
     "publication_year": 2022, "venue": "ACS Nano"},

    # ── Mathematics ───────────────────────────────────────────
    {"title": "Convergence Analysis of Adam Optimizer Under Non-Convex Settings",
     "abstract": "Rigorous convergence guarantees for Adam optimizer in non-convex stochastic optimization, proving O(1/sqrt(T)) convergence rate. Validated with faster convergence on ResNet-50 and BERT pretraining.",
     "keywords": ["optimization", "Adam optimizer", "convergence", "stochastic gradient descent"],
     "department": "Mathematics", "authors": ["Prof. Li Wei", "Dr. Priya Sharma", "Alice Lin"],
     "publication_year": 2023, "venue": "Journal of Machine Learning Research"},

    {"title": "Topological Data Analysis for Persistent Homology in Neural Networks",
     "abstract": "Persistent homology reveals that network width correlates with 0th Betti number reduction speed, providing new geometric insights into neural network generalization and loss landscape topology.",
     "keywords": ["topological data analysis", "persistent homology", "neural networks", "deep learning theory"],
     "department": "Mathematics", "authors": ["Prof. Li Wei", "Alice Lin", "Prof. Robert Feynman Jr"],
     "publication_year": 2022, "venue": "ICML 2022"},

    {"title": "Stochastic Differential Equations for Score-Based Generative Models",
     "abstract": "A unified SDE framework for diffusion models proving equivalence between DDPM, DDIM, and continuous-time formulations. Motivates a novel variance-preserving SDE achieving FID 2.1 on CIFAR-10.",
     "keywords": ["diffusion models", "SDE", "generative models", "score matching", "DDPM"],
     "department": "Mathematics", "authors": ["Prof. Li Wei", "Dr. Priya Sharma"],
     "publication_year": 2024, "venue": "NeurIPS 2024"},

    {"title": "Stochastic Modeling of Renewable Energy Grids with Battery Storage",
     "abstract": "Stochastic MPC for real-time dispatch in hybrid renewable grids (solar + wind + battery). On a 100 MW microgrid model, reduces curtailment by 41% and improves renewable utilization to 94.7%.",
     "keywords": ["stochastic modeling", "renewable energy", "battery storage", "MPC", "microgrid"],
     "department": "Mathematics", "authors": ["Prof. Li Wei", "Dr. Fatima Hassan", "Dr. James Wilson"],
     "publication_year": 2023, "venue": "Applied Energy"},

    # ── Management Studies ────────────────────────────────────
    {"title": "AI Adoption in Healthcare: A Cross-National Survey Study",
     "abstract": "Survey of 1,847 healthcare professionals across India, Germany, and the US. Trust (68%), explainability (61%), and regulatory uncertainty (54%) are primary barriers. AI-trained clinicians show 3.2x higher adoption intent.",
     "keywords": ["AI adoption", "healthcare management", "digital transformation", "survey"],
     "department": "Management Studies", "authors": ["Prof. Helen Zhang", "Dr. Anita Gupta", "Diana Student"],
     "publication_year": 2023, "venue": "Journal of Medical Internet Research"},

    {"title": "Platform Economics and Network Effects in EdTech Startups",
     "abstract": "Panel data from 340 EdTech platforms (2015–2023) models network effect strength. Instructor-student ratio below 1:50 shows 2.8x stronger retention network effects, informing optimal scaling strategies.",
     "keywords": ["platform economics", "network effects", "EdTech", "two-sided markets"],
     "department": "Management Studies", "authors": ["Prof. Helen Zhang", "Diana Student"],
     "publication_year": 2022, "venue": "Strategic Management Journal"},

    {"title": "NLP-Driven Patent Landscape Analysis for R&D Strategy",
     "abstract": "BERT topic modeling and citation network analysis on 2.3M renewable energy patents identifies 14 emerging clusters, predicts citation impact with R²=0.79, and provides actionable R&D strategy recommendations.",
     "keywords": ["NLP", "patent analysis", "BERT", "topic modeling", "R&D strategy"],
     "department": "Management Studies", "authors": ["Prof. Helen Zhang", "Dr. Priya Sharma", "Dr. Alice Kim"],
     "publication_year": 2023, "venue": "Research Policy"},

    # ── Cross-Departmental ────────────────────────────────────
    {"title": "Privacy-Preserving Federated Learning for Medical Image Analysis",
     "abstract": "FedMed combines federated learning with homomorphic encryption for collaborative radiology AI training without sharing patient data. Deployed across 7 hospitals, achieves within 0.8% AUC of centralized training.",
     "keywords": ["federated learning", "homomorphic encryption", "medical imaging", "privacy"],
     "department": "Computer Science", "authors": ["Dr. Alice Kim", "Dr. Sarah Chen", "Dr. Raj Patel", "Dr. Bob Johnson"],
     "publication_year": 2023, "venue": "MICCAI 2023"},
]


def _generate_body_text(paper: dict) -> str:
    """Generate realistic paper body text for chunking and embedding."""
    kw_str = ", ".join(paper["keywords"][:3])
    return (
        f"1. INTRODUCTION\n{paper['abstract']}\n\n"
        f"2. RELATED WORK\nPrior work in {kw_str} has laid the foundation for this research. "
        f"This paper builds on recent advances in {paper['keywords'][0]} and extends them "
        f"to address limitations in {paper['keywords'][1] if len(paper['keywords']) > 1 else paper['keywords'][0]}.\n\n"
        f"3. METHODOLOGY\nThe proposed approach employs {paper['keywords'][0]} as its core technique. "
        f"Experiments were conducted on standard benchmarks with rigorous ablation studies "
        f"to validate each component of the proposed system.\n\n"
        f"4. EXPERIMENTS AND RESULTS\nExtensive evaluation demonstrates significant improvements. "
        f"Statistical significance confirmed with p < 0.05 across all primary metrics. "
        f"Ablation studies confirm the contribution of each proposed component.\n\n"
        f"5. CONCLUSION\nWe presented a novel approach to {paper['keywords'][0]}. "
        f"Results demonstrate state-of-the-art performance and open directions for future research "
        f"in {paper['keywords'][-1]}.\n\n"
        f"REFERENCES\n[1] Foundational work in {paper['keywords'][0]}. "
        f"[2] Recent advances in {paper['keywords'][1] if len(paper['keywords']) > 1 else paper['keywords'][0]}."
    )


def insert_sample_papers():
    print("\n" + "=" * 65)
    print("  ResearchIQ — Sample Dataset Generator  (50 papers)")
    print("=" * 65 + "\n")

    DatabaseManager.connect()
    col = get_papers_collection()
    users_col = get_users_collection()

    # Get an uploader user (prefer research_head, fallback to any)
    uploader = users_col.find_one({"role": "research_head"}) or users_col.find_one({})
    if not uploader:
        print("  ❌  No users found! Run 'python scripts/init_db.py' first.\n")
        sys.exit(1)

    uploader_id = uploader["user_id"]
    uploader_name = uploader["name"]

    from backend.ml.impact_predictor import ImpactPredictor
    predictor = ImpactPredictor()

    inserted = 0
    skipped = 0
    for paper in PAPERS:
        if col.find_one({"title": paper["title"]}):
            skipped += 1
            continue

        paper_id = str(uuid.uuid4())
        full_text = _generate_body_text(paper)
        # Chunk with 800-char chunks, 100-char overlap
        chunks = []
        i = 0
        while i < len(full_text):
            chunks.append(full_text[i:i + 800])
            i += 700

        doc = {
            "paper_id": paper_id,
            **paper,
            "uploaded_by": uploader_id,
            "uploaded_by_name": uploader_name,
            "file_path": f"./data/papers/sample_{paper_id}.pdf",
            "filename": f"sample_{paper_id}.pdf",
            "file_size_mb": round(random.uniform(0.5, 4.0), 2),
            "page_count": random.randint(8, 16),
            "upload_date": datetime.utcnow() - timedelta(days=random.randint(1, 730)),
            "processing_status": "completed",
            "extracted_text": full_text,
            "text_chunks": chunks,
            "sections": {"abstract": paper["abstract"]},
            "embedding_stored": False,
            "predicted_impact_score": None,
            "processing_error": None,
        }

        score = predictor.predict_single(doc)
        doc["predicted_impact_score"] = score

        col.insert_one(doc)
        inserted += 1
        dept = paper["department"][:15].ljust(15)
        print(f"  ✅  [{dept}] {paper['title'][:52]}")

    print(f"\n  📦  Inserted {inserted} new | Skipped {skipped} existing")
    print(f"  📊  Total in DB: {col.count_documents({})}\n")

    # ── Embed all un-embedded papers into ChromaDB ────────────
    print("  🔢  Generating embeddings (sentence-transformers)...")
    print("      First run downloads ~90 MB model — please wait...\n")

    try:
        from backend.vectordb.chroma_manager import ChromaManager
        chroma = ChromaManager()

        to_embed = list(col.find(
            {"embedding_stored": False, "processing_status": "completed"},
            {"paper_id": 1, "title": 1, "abstract": 1, "keywords": 1,
             "department": 1, "authors": 1, "publication_year": 1,
             "venue": 1, "text_chunks": 1}
        ))

        print(f"  📌  Embedding {len(to_embed)} papers into ChromaDB...\n")

        for i, paper_doc in enumerate(to_embed, 1):
            try:
                chunks = paper_doc.get("text_chunks") or [paper_doc.get("abstract", "")]
                chroma.add_paper(paper_doc, chunks)
                col.update_one(
                    {"paper_id": paper_doc["paper_id"]},
                    {"$set": {"embedding_stored": True}}
                )
                print(f"     [{i:2d}/{len(to_embed)}] ✅  {paper_doc['title'][:52]}")
            except Exception as e:
                print(f"     [{i:2d}/{len(to_embed)}] ⚠️   Failed: {e}")

        stats = chroma.get_collection_stats()
        print(f"\n  ✅  ChromaDB: {stats['total_embeddings']} total embeddings stored")

    except Exception as e:
        print(f"  ⚠️   ChromaDB embedding skipped: {e}")
        print("      RAG chatbot will not work until embeddings are generated.")

    # ── Final Summary ─────────────────────────────────────────
    total = col.count_documents({})
    depts = col.distinct("department")
    years = [y for y in col.distinct("publication_year") if y]
    authors = col.distinct("authors")

    print("\n" + "=" * 65)
    print("  🎉  Dataset generation complete!")
    print("=" * 65)
    print(f"\n  Total papers   : {total}")
    print(f"  Departments    : {len(depts)}")
    if years:
        print(f"  Year range     : {min(years)} – {max(years)}")
    print(f"  Unique authors : ~{len(authors)}")
    print("\n  ➡️   Start the server: python main.py")
    print("  🌐  Open in browser: http://localhost:8000\n")


if __name__ == "__main__":
    insert_sample_papers()
