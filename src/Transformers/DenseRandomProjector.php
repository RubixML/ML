<?php

namespace Rubix\ML\Transformers;

class DenseRandomProjector extends SparseRandomProjector
{
    const MULTIPLIER = 1;

    const DISTRIBUTION = [-1, 1];
}
