<?php

namespace Rubix\ML\Embedders;

use Rubix\ML\Model;
use Rubix\ML\Datasets\Dataset;

interface Embedder extends Model
{
    /**
     * Embed a high dimensional dataset into a lower dimensional one.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array[]
     */
    public function embed(Dataset $dataset) : array;
}
