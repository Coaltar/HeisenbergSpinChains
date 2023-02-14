
## Overview

A simple program to produce and diagonalize the Hamiltonian for a spin-1/2 Heisenberg chain (isotropic).

Has two sets of functions to that end, contained in lattice.py and lattice_sparse.py.
The latter introduces a few minor optimizations using scipy sparse matrices.

Produces some basic graphs to demonstrate the runtime of these algorithms, outputs the base energy values to stdout.

Really a very simple set of scripts.

## Future? (Post Morterm?)

Future improvements to this might include: 
- Message passing to make use of multiple cpus
- Port it to C (yeah right!)
- Expand to work with 2 Dimensional or Anisotropic lattices 
