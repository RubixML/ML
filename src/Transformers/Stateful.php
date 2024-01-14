<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;

/**
 * Stateful
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Stateful extends Transformer
{
    /**
     * Fit the transformer to a dataset.
     *
     * @param Dataset $dataset
     */
    public function fit(Dataset $dataset) : void;

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool;
}
