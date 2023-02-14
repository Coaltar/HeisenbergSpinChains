.PHONY: start test test-sparse test-graph

start: 
	pdm run jupyter-notebook .

test:
	pdm run python "./src/lattice/LatticeGenerator.py"

test-sparse: 
	pdm run python "./src/lattice/lattice_sparse.py"

test-graph: 
	pdm run python "./src/lattice/graph.py"
