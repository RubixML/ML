<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;

interface Transformer
{
    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void;

    /**
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void;
}
