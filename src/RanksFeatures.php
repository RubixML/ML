<?php

namespace Rubix\ML;

interface RanksFeatures extends Learner
{
    /**
     * Return the normalized importance scores of each feature column of the training set.
     *
     * @return float[]
     */
    public function featureImportances() : array;
}
