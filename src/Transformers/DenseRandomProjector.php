<?php

namespace Rubix\ML\Transformers;

/**
 * Dense Random Projector
 *
 * A Random Projector is a dimensionality reducer based on the
 * Johnson-Lindenstrauss lemma that uses a random matrix to project a feature
 * vector onto a user-specified number of dimensions. It is faster than most
 * non-randomized dimensionality reduction techniques and offers similar
 * performance. The difference between the Dense and Sparse Random Projectors
 * are that the Dense version uses a dense random guassian distribution and the
 * Sparse version uses a sparse matrix (mostly 0’s).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DenseRandomProjector extends SparseRandomProjector
{
    const BETA = 1;

    const DISTRIBUTION = [-1, 1];
}
