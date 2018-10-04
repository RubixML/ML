<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;
use RuntimeException;

/**
 * Variance Threshold Filter
 *
 * A type of feature selector that removes all columns that have a lower
 * variance than the threshold. Variance is computed as the population variance
 * of all the values in the feature column.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class VarianceThresholdFilter implements Transformer
{
    /**
     * The minimum variance a feature column must have in order to be selected.
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
     * @return array
     */
    public function selected() : array
    {
        return array_keys($this->selected ?? []);
    }

    /**
     * Fit the transformer to the incoming data frame.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function fit(DataFrame $dataframe) : void
    {
        $this->selected = [];

        foreach ($dataframe->types() as $column => $type) {
            if ($type === DataFrame::CONTINUOUS) {
                list($mean, $variance) = Stats::meanVar($dataframe->column($column));

                if ($variance > $this->threshold) {
                    $this->selected[$column] = true;
                }
            } else {
                $this->selected[$column] = true;
            }
        }
    }

    /**
     * Apply the transformation to the samples in the data frame.
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
            $sample = array_values(array_intersect_key($sample, $this->selected));
        }
    }
}
