<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;
use RuntimeException;

/**
 * Min Max Normalizer
 *
 * The *Min Max* Normalizer scales the input features to a value between
 * a user-specified range (default 0 to 1).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MinMaxNormalizer implements Elastic
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
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return $this->mins and $this->scales;
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
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->minimums = $this->maximums = $this->scales = $this->mins = [];

        foreach ($dataset->types() as $column => $type) {
            if ($type === DataType::CONTINUOUS) {
                $this->minimums[$column] = INF;
                $this->maximums[$column] = -INF;
            }
        }

        $this->update($dataset);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function update(Dataset $dataset) : void
    {
        if (is_null($this->minimums) or is_null($this->maximums)) {
            $this->fit($dataset);
            
            return;
        }

        $columns = $dataset->columnsByType(DataType::CONTINUOUS);

        foreach ($columns as $column => $values) {
            [$min, $max] = Stats::range($values);

            $min = min($min, $this->minimums[$column]);
            $max = max($max, $this->maximums[$column]);

            $scale = ($this->max - $this->min)
                / (($max - $min) ?: self::EPSILON);

            $minHat = $this->min - $min * $scale;

            $this->minimums[$column] = $min;
            $this->maximums[$column] = $max;
            $this->scales[$column] = $scale;
            $this->mins[$column] = $minHat;
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
        if (is_null($this->mins) or is_null($this->scales)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->scales as $column => $scale) {
                $sample[$column] *= $scale;
                $sample[$column] += $this->mins[$column];
            }
        }
    }
}
