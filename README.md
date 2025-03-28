# Replication package of From Annotation to Detection: Identifying Architecture Technical Debt in Jira Issues using Active Learning

## Description of this study:
Architectural Technical Debt (ATD) accumulates when architectural compromises are made to advance short-term development, often resulting in long-term maintenance issues. Identifying ATD items in issue-tracking systems (such as Jira) is crucial for the sustainability of the system. This study aims to automate ATD detection by contributing an enriched dataset of Jira issues and evaluating classification models using Active Learning. We relabeled an initial dataset of 116 ATD-related Jira issues, refining it to 57 items agreed upon by all authors. From these, we extracted 15 representative keywords using TF-IDF, KeyBERT, and class-specific KeyBERT. These keywords facilitated the labeling of additional Jira issues across multiple projects, thereby constructing an extended ATD dataset for this study. Active learning techniques were then applied to optimize the annotation effort and improve classification accuracy. Our method produced a high-quality ATD dataset and demonstrated that active learning significantly enhances ATD classification performance when compared to traditional supervised learning. The proposed model showed improved accuracy, consistency, and generalizability across diverse projects. Our work shows how to minimize annotation costs, improve classification performance, and achieve more scalable and automated ATD detection and management.

## Contents

### Dataset
- `LATEST-ATD-DATASET.csv`\
    A CSV dataset derived from Jira issue trackers of ten Apache open-source projects contains issue reports labelled ATD, Weak-ATD, and Non-ATD.
- `LATEST-ATD-DATASET-NO-WEAK.csv`\
    This is a CSV dataset derived from Jira issue trackers of ten Apache open-source projects. It contains issue reports labeled as ATD and Non-ATD, with the exception of Weak-ATD.

### Source code

#### `data prep/` 
  This folder contains all source code used in the **data preparation phase**

#### `unsupervised/`
  This folder includes code for detecting ATD using **three keyword-based methods**

#### `supervised/`
  This folder contains code for both **supervised learning** and **active learning** approaches to ATD detection, including model training and active learning query strategies

### Results
  This folder contains extracted keywords from three keyword-based methods and performance results in terms of precision, recall, and f1-score from four different query strategies
