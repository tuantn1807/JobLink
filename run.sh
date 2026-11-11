#!/bin/bash

PYTHON_BIN=".venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

# Train GNN + cache HR KG if checkpoint does not exist yet
if [ ! -f "data/hr_kg_gnn.pt" ] || [ ! -f "data/hr_kg_cache.pkl" ]; then
    echo "Training HR Knowledge Graph GNN..."
    "$PYTHON_BIN" train_gnn.py
fi

# Run the Flask app
echo "Starting LLM-Enhanced HR KG + GNN Job Matching System..."
"$PYTHON_BIN" app.py
