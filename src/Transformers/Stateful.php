<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;

interface Stateful extends Transformer
{
    /**
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void;

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool;
}
