<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
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
class VarianceThresholdFilter implements Transformer, Stateful
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
            throw new InvalidArgumentException('Threshold must be a positive'
                . ' value.');
        }

        $this->threshold = $threshold;
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
        $this->selected = [];

        foreach ($dataset->types() as $column => $type) {
            if ($type === DataFrame::CONTINUOUS) {
                $values = $dataset->column($column);

                if (Stats::variance($values) <= $this->threshold) {
                    continue 1;
                }
            }

            $this->selected[$column] = true;
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @param  array|null  $labels
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples, ?array &$labels = null) : void
    {
        if (is_null($this->selected)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $sample = array_intersect_key($sample, $this->selected);
        }
    }
}
