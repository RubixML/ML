<?php

namespace Rubix\ML\Embedders;

use Rubix\ML\Datasets\Dataset;

interface Embedder
{
    /**
     * Return the data types that this embedder is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array;

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array;

    /**
     * Embed a high dimensional dataset into a lower dimensional one.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array[]
     */
    public function embed(Dataset $dataset) : array;
}
