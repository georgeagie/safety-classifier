# Safety Classifier

A research project exploring **data-driven safety classification** for autonomous agents.  
This model predicts whether an agent’s action is **nominal** or **safety-critical**, enabling real-time risk assessment in control and robotics applications.

## Overview
This work investigates how machine learning can identify safety-critical behaviors based on agent state, goal, and environment parameters.  
The classifier is trained on a labeled dataset of simulated trajectories and evaluated through standard performance metrics and visualizations.

## Objectives
- Develop a supervised classifier for safety-critical decision detection  
- Evaluate reliability using quantitative metrics and confusion analysis  
- Support downstream safety assurance and control barrier function research  

## Methodology
1. **Data Processing:** Load and preprocess trajectory data labeled as nominal or safety-critical  
2. **Model Training:** Train a neural network-based classifier in PyTorch  
3. **Evaluation:** Generate loss curves, confusion matrices, and performance reports  
4. **Inference:** Run local classification on unseen trajectory data  

