<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;

/**
 * Elastic
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Elastic extends Stateful
{
    /**
     * Update the fitting of the transformer.
     *
     * @param Dataset $dataset
     */
    public function update(Dataset $dataset) : void;
}
