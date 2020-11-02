<?php

namespace Rubix\ML;

use Stringable;

/**
 * Ranks Features
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface RanksFeatures extends Trainable, Stringable
{
    /**
     * Return the normalized importance scores of each feature column of the training set.
     *
     * @return float[]
     */
    public function featureImportances() : array;
}
