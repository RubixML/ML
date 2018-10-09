<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;

interface Elastic extends Stateful
{
    /**
     * Update the fitting of the transformer.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function update(Dataset $dataset) : void;
}
