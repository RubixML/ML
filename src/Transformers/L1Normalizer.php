<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

use function array_map;
use function array_walk;
use function array_sum;

/**
 * L1 Normalizer
 *
 * Transform each sample vector in the sample matrix such that each feature is divided
 * by the L1 norm (or *magnitude*) of that vector. The resulting sample will have
 * continuous features between 0 and 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class L1Normalizer implements Transformer
{
    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Transform the dataset in place.
     *
     * @param array<mixed[]> $samples
     */
    public function transform(array &$samples) : void
    {
        array_walk($samples, [$this, 'normalize']);
    }

    /**
     * Normalize a sample by its L1 norm.
     *
     * @param list<int|float> $sample
     */
    protected function normalize(array &$sample) : void
    {
        $norm = array_sum(array_map('abs', $sample));

        if ($norm == 0) {
            return;
        }

        foreach ($sample as &$value) {
            $value /= $norm;
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'L1 Normalizer';
    }
}
