<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Trainable
{
    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void;

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool;
}
