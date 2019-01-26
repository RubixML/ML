<?php

namespace Rubix\ML\Other\Specifications;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

class DatasetIsCompatibleWithEstimator
{
    /**
     * Perform a check.
     * 
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @param  \Rubix\ML\Estimator  $estimator
     * @throws \InvalidArgumentException
     * @return void
     */
    public static function check(Dataset $dataset, Estimator $estimator) : void
    {
        $types = $dataset->uniqueTypes();

        $same = array_intersect($types, $estimator->compatibility());

        if (count($same) < count($types)) {
            throw new InvalidArgumentException('Estimator is not'
                . ' compatible with the data types given.');
        };
    }
}