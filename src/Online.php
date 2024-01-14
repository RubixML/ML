<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

/**
 * Online
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Online extends Learner
{
    /**
     * Perform a partial train on the learner.
     *
     * @param Dataset $dataset
     */
    public function partial(Dataset $dataset) : void;
}
