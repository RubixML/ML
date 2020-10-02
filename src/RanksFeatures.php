<?php

namespace Rubix\ML;

use Stringable;

interface RanksFeatures extends Stringable
{
    /**
     * Return the normalized importance scores of each feature column of the training set.
     *
     * @return float[]
     */
    public function featureImportances() : array;
}
