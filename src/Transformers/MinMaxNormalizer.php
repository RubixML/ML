<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use const Rubix\ML\EPSILON;

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
class MinMaxNormalizer implements Transformer, Stateful, Elastic, Persistable
{
    use AutotrackRevisions;

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
     * @var (int|float)[]|null
     */
    protected $minimums;

    /**
     * The computed maximums of the fitted data.
     *
     * @var (int|float)[]|null
     */
    protected $maximums;

    /**
     * The scale of each feature column.
     *
     * @var (int|float)[]|null
     */
    protected $scales;

    /**
     * The scaled minimums of each feature column.
     *
     * @var (int|float)[]|null
     */
    protected $mins;

    /**
     * @param float $min
     * @param float $max
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $min = 0.0, float $max = 1.0)
    {
        if ($min > $max) {
            throw new InvalidArgumentException('Minimum cannot be greater'
                . ' than maximum.');
        }

        $this->min = $min;
        $this->max = $max;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
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
     * @return (int|float)[]|null
     */
    public function minimums() : ?array
    {
        return $this->minimums;
    }

    /**
     * Return the maximums of each feature column.
     *
     * @return (int|float)[]|null
     */
    public function maximums() : ?array
    {
        return $this->maximums;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->minimums = $this->maximums = $this->scales = $this->mins = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $this->minimums[$column] = INF;
                $this->maximums[$column] = -INF;
            }
        }

        $this->update($dataset);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function update(Dataset $dataset) : void
    {
        if ($this->minimums === null or $this->maximums === null) {
            $this->fit($dataset);

            return;
        }

        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        foreach ($dataset->columnTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $values = $dataset->column($column);

                $min = min($values);
                $max = max($values);

                $min = min($min, $this->minimums[$column]);
                $max = max($max, $this->maximums[$column]);

                $scale = ($this->max - $this->min)
                    / (($max - $min) ?: EPSILON);

                $minHat = $this->min - $min * $scale;

                $this->minimums[$column] = $min;
                $this->maximums[$column] = $max;
                $this->scales[$column] = $scale;
                $this->mins[$column] = $minHat;
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->mins === null or $this->scales === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->scales as $column => $scale) {
                $value = &$sample[$column];

                $value *= $scale;
                $value += $this->mins[$column];
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Min Max Normalizer (min: {$this->min}, max: {$this->max})";
    }
}
