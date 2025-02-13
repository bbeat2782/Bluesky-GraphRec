train_link_prediction.py
|
|--[Imports & Setup]
|   |--Standard libraries (logging, time, sys, etc.)
|   |--PyTorch
|   |--Custom modules (TGAT, CAWN, GraphRec, etc.)
|
|--[Main Training Loop]
    |
    |--[Data Preparation]
    |   |--Load data (train/val/test) get_link_prediction_data
            
    |   |--Initialize samplers 
    |   |--Create data loaders 
    |
    |--[Multiple Training Runs]
        |
        |--[Per Run]
            |
            |--[Setup]
            |   |--Initialize model
            |   |--Setup optimizer
            |   |--Setup early stopping
            |
            |--[Training Epochs]
                |
                |--[Per Epoch]
                    |
                    |--[Training Phase]
                    |   |--Sample batches
                    |   |--Compute embeddings
                    |   |--Calculate BPR loss
                    |   |--Backprop & optimize
                    |
                    |--[Validation Phase]
                    |   |--Evaluate on validation set
                    |   |--Evaluate on new node validation
                    |
                    |--[Testing Phase]
                    |   |--Periodic testing
                    |   |--Evaluate on test set
                    |   |--Evaluate on new node test
                    |
                    |--[Metrics & Logging]
                        |--Save training curves
                        |--Save model checkpoints
                        |--Log metrics