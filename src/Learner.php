<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

/**
 * Learner
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Learner
{
    /**
     * Train the learner with a dataset.
     *
     * @param Dataset $dataset
     */
    public function train(Dataset $dataset) : void;

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool;
}
