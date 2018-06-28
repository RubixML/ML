<?php

namespace Rubix\ML\Transformers;

class DenseRandomProjector extends SparseRandomProjector
{
    const BETA = 1;

    const DISTRIBUTION = [-1, 1];
}
