<?php

namespace Rubix\ML\Embedders;

use Rubix\ML\Transformers\Transformer;

interface Embedder extends Transformer
{
    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array;
}
