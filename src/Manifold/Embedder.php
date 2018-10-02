<?php

namespace Rubix\ML\Manifold;

use Rubix\ML\Datasets\Dataset;

interface Embedder
{
    const EPSILON = 1e-8;

    /**
     * Embed a high dimensional sample matrix into a lower dimensional one.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function embed(Dataset $dataset) : array;
}
