<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;

interface Stateful extends Transformer
{
    /**
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void;
}
