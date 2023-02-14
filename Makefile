.PHONY: test test-sparse test-graph


test:
	pdm run python "./src/lattice/LatticeGenerator.py"

test-sparse: 
	pdm run python "./src/lattice/lattice_sparse.py"

test-graph: 
	pdm run python "./src/lattice/graph.py"
