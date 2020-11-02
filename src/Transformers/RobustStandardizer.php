<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use RuntimeException;
use Stringable;

use function is_null;

use const Rubix\ML\EPSILON;

/**
 * Robust Standardizer
 *
 * This Transformer standardizes continuous features by removing the median and
 * dividing over the median absolute deviation (MAD), a value referred to as
 * robust Z-Score. The use of robust statistics makes this standardizer more
 * immune to outliers than the Z Scale Standardizer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RobustStandardizer implements Transformer, Stateful, Stringable
{
    /**
     * Should we center the data at 0?
     *
     * @var bool
     */
    protected $center;

    /**
     * The computed medians of the fitted data indexed by column.
     *
     * @var (int|float)[]|null
     */
    protected $medians;

    /**
     * The computed median absolute deviations of the fitted data
     * indexed by column.
     *
     * @var (int|float)[]|null
     */
    protected $mads;

    /**
     * @param bool $center
     */
    public function __construct(bool $center = true)
    {
        $this->center = $center;
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
        return $this->medians and $this->mads;
    }

    /**
     * Return the medians calculated by fitting the training set.
     *
     * @return (int|float)[]|null
     */
    public function medians() : ?array
    {
        return $this->medians;
    }

    /**
     * Return the median absolute deviations calculated during fitting.
     *
     * @return (int|float)[]|null
     */
    public function mads() : ?array
    {
        return $this->mads;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->medians = $this->mads = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $values = $dataset->column($column);

                [$median, $mad] = Stats::medianMad($values);

                $this->medians[$column] = $median;
                $this->mads[$column] = $mad ?: EPSILON;
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->mads) or is_null($this->medians)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->mads as $column => $mad) {
                $value = &$sample[$column];

                if ($this->center) {
                    $value -= $this->medians[$column];
                }

                $value /= $mad;
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
        return 'Robust Standardizer {center: ' . Params::toString($this->center) . ')';
    }
}
