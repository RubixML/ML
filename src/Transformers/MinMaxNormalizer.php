<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;
use RuntimeException;

/**
 * Min Max Normalizer
 *
 * The Min Max Normalization scales the input features to a value between
 * a user-specified range (default 0 to 1).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MinMaxNormalizer implements Transformer, Online
{
    /**
     * The minimum value of the transformed features.
     * 
     * @var float
     */
    protected $min;

    /**
     * The maximum value of the transformed features.
     * 
     * @var float
     */
    protected $max;

    /**
     * The computed minimums of the fitted data.
     *
     * @var array|null
     */
    protected $minimums;

    /**
     * The computed maximums of the fitted data.
     *
     * @var array|null
     */
    protected $maximums;

    /**
     * The scale of each feature column.
     * 
     * @var array|null
     */
    protected $scales;

    /**
     * The scaled minimums of each feature column.
     * 
     * @var array|null
     */
    protected $mins;

    /**
     * @param  float  $min
     * @param  float  $max
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $min = 0., float $max = 1.)
    {
        if ($min > $max) {
            throw new InvalidArgumentException('Minimum cannot be greater'
                . ' than maximum.');
        }

        $this->min = $min;
        $this->max = $max;
    }

    /**
     * Return the minmums of each feature column.
     * 
     * @return array|null
     */
    public function minimums() : ?array
    {
        return $this->minimums;
    }

    /**
     * Return the maximums of each feature column.
     * 
     * @return array|null
     */
    public function maximums() : ?array
    {
        return $this->maximums;
    }

    /**
     * Fit the transformer to the incoming data frame.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function fit(DataFrame $dataframe) : void
    {
        $this->minimums = $this->maximums
            = $this->scales = $this->mins = [];

        foreach ($dataframe->types() as $column => $type) {
            if ($type === DataFrame::CONTINUOUS) {
                $this->minimums[$column] = INF;
                $this->maximums[$column] = -INF;
            }
        }

        $this->update($dataframe);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function update(DataFrame $dataframe) : void
    {
        if (is_null($this->minimums) or is_null($this->maximums)) {
            $this->fit($dataframe);
        }

        foreach ($dataframe->types() as $column => $type) {
            if ($type === DataFrame::CONTINUOUS) {
                $values = $dataframe->column($column);

                list($min, $max) = Stats::range($values);

                $min = min($min, $this->minimums[$column]);
                $max = max($max, $this->maximums[$column]);

                $scale = ($this->max - $this->min)
                    / ($max - $min) ?: self::EPSILON;

                $minHat = $this->min - $min * $scale;

                $this->minimums[$column] = $min;
                $this->maximums[$column] = $max;
                $this->scales[$column] = $scale;
                $this->mins[$column] = $minHat;
            }
        }
    }

    /**
     * Apply the transformation to the sample matrix.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->mins) or is_null($this->scales)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->scales as $column => $scale) {
                $sample[$column] = ($this->mins[$column] + $sample[$column])
                    * $scale;
            }
        }
    }
}
