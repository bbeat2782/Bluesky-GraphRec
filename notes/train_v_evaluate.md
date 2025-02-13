train_link_prediction.py                evaluate_link_prediction.py
│                                      │
├── Purpose                            ├── Purpose
│   └── Train new models               │   └── Evaluate saved models
│                                      │
├── Data Loading                       ├── Data Loading
│   ├── Training data                  │   ├── Test data only
│   ├── Validation data               │   └── Uses eval-specific data loader
│   └── Test data                     │
│                                      │
├── Model Handling                     ├── Model Handling
│   ├── Creates new model             │   ├── Loads saved model
│   ├── Initializes weights           │   └── No training/updates
│   └── Saves checkpoints             │
│                                      │
├── Process Flow                       ├── Process Flow
│   ├── Training loop                 │   ├── Single evaluation pass
│   ├── Validation steps              │   ├── Compute metrics
│   ├── Early stopping                │   └── Save results
│   └── Model updates                 │
│                                      │
└── Output                            └── Output
    ├── Trained model                     ├── Evaluation metrics
    ├── Training curves                   └── JSON results file
    └── Validation metrics