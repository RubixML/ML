<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataType;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;
use RuntimeException;

/**
 * Variance Threshold Filter
 *
 * A type of feature selector that selects feature columns that have a greater
 * variance than the user-specified threshold. As an extreme example, if a
 * feature column has a variance of 0 then that feature will all be valued equally.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class VarianceThresholdFilter implements Stateful
{
    /**
     * Feature columns with a variance greater than this threshold will be
     * selected.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The feature columns that have been selected.
     *
     * @var array|null
     */
    protected $selected;

    /**
     * @param  float  $threshold
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $threshold = 0.)
    {
        if ($threshold < 0.) {
            throw new InvalidArgumentException('Threshold must be 0 or greater'
                . ", $threshold given.");
        }

        $this->threshold = $threshold;
    }

    /**
     * Is the transformer fitted?
     * 
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->selected);
    }

    /**
     * Return the column indexes that have been selected during fitting.
     * 
     * @return array
     */
    public function selected() : array
    {
        return array_keys($this->selected ?: []);
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $columns = $dataset->columnsByType(DataType::CONTINUOUS);

        $this->selected = [];

        foreach ($columns as $column => $values) {
            if (Stats::variance($values) <= $this->threshold) {
                continue 1;
            }

            $this->selected[$column] = true;
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->selected)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $sample = array_intersect_key($sample, $this->selected);
        }
    }
}
