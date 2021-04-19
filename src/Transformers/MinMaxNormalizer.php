<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function min;
use function max;

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
    protected float $min;

    /**
     * The maximum value of the transformed features.
     *
     * @var float
     */
    protected float $max;

    /**
     * The computed minimums of the fitted data.
     *
     * @var (int|float)[]|null
     */
    protected ?array $minimums = null;

    /**
     * The computed maximums of the fitted data.
     *
     * @var (int|float)[]|null
     */
    protected ?array $maximums = null;

    /**
     * The scale coefficients of each feature.
     *
     * @var float[]|null
     */
    protected ?array $scales = null;

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
        return $this->minimums and $this->maximums;
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

        $this->minimums = $this->maximums = [];

        foreach ($dataset->featureTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $values = $dataset->feature($column);

                $min = min($values);
                $max = max($values);

                $scale = ($this->max - $this->min) / ($max - $min);

                $this->minimums[$column] = $min;
                $this->maximums[$column] = $max;
                $this->scales[$column] = $scale;
            }
        }
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function update(Dataset $dataset) : void
    {
        if ($this->minimums === null or $this->maximums === null or $this->scales === null) {
            $this->fit($dataset);

            return;
        }

        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        foreach ($this->scales as $column => &$scale) {
            $values = $dataset->feature($column);

            $min = min($this->minimums[$column], ...$values);
            $max = max($this->maximums[$column], ...$values);

            $scale = ($this->max - $this->min) / ($max - $min);

            $this->minimums[$column] = $min;
            $this->maximums[$column] = $max;
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
        if ($this->scales === null or $this->minimums === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->scales as $column => $scale) {
                $value = &$sample[$column];

                $min = $this->minimums[$column];

                $value *= $scale;

                $value += $this->min - $min * $scale;
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
