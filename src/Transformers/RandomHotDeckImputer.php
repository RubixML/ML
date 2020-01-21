<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

use const Rubix\ML\PHI;

/**
 * Random Hot Deck Imputer
 *
 * A method of imputation similar to KNN Imputer but instead of computing a weighted average
 * of the neighbors' features, Random Hot Deck picks a value from the neighborhood randomly
 * but sampled by distance. This makes Random Hot Deck Imputer slightly more computationally
 * efficient while satisfying some balancing equations at the same time.
 *
 * **Note:** NaN safe distance kernels, such as Safe Euclidean, are required
 * for continuous features.
 *
 * References:
 * [1] C. Hasler et al. (2015). Balanced k-Nearest Neighbor Imputation.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RandomHotDeckImputer extends KNNImputer
{
    /**
     * Choose a value to impute from a given set of values.
     *
     * @param (string|int|float)[] $values
     * @param float[] $distances
     * @param \Rubix\ML\DataType $type
     * @return string|int|float
     */
    protected function impute(array $values, array $distances, DataType $type)
    {
        if ($this->weighted) {
            $weights = [];

            foreach ($distances as $distance) {
                $weights[] = 1.0 / (1.0 + $distance);
            }

            $value = $type->isContinuous() ? NAN : '?';

            $max = (int) round(array_sum($weights) * PHI);

            $delta = rand(0, $max) / PHI;

            foreach ($weights as $index => $weight) {
                $delta -= $weight;

                if ($delta <= 0.0) {
                    $value = $values[$index];
                        
                    break 1;
                }
            }
        } else {
            $value = $values[rand(0, $this->k - 1)];
        }

        return $value;
    }
}
