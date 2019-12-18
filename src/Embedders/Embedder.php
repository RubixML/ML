<?php

namespace Rubix\ML\Embedders;

use Rubix\ML\Datasets\Dataset;

interface Embedder
{
    /**
     * Return the data types that this embedder is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array;

    /**
     * Embed a high dimensional dataset into a lower dimensional one.
     *
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     * @return array[]
     */
    public function embed(Dataset $dataset) : array;
}
