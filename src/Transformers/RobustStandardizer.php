<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\RuntimeException;

/**
 * Robust Standardizer
 *
 * This standardizer transforms continuous features by centering them around the median and scaling by the median absolute
 * deviation (MAD) referred to as a *robust*  or *modified* Z-Score. The use of robust statistics make this standardizer
 * more immune to outliers than Z Scale Standardizer.
 *
 * $$
 * {\displaystyle z^\prime = {x - \operatorname {median}(X) \over MAD }}
 * $$
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RobustStandardizer implements Transformer, Stateful, Reversible, Persistable
{
    use AutotrackRevisions;

    /**
     * Should we center the data at 0?
     *
     * @var bool
     */
    protected bool $center;

    /**
     * The computed medians of the fitted data indexed by column.
     *
     * @var (int|float)[]|null
     */
    protected ?array $medians = null;

    /**
     * The computed median absolute deviations of the fitted data
     * indexed by column.
     *
     * @var (int|float)[]|null
     */
    protected ?array $mads = null;

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
     * @param Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->medians = $this->mads = [];

        foreach ($dataset->featureTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $values = $dataset->feature($column);

                [$median, $mad] = Stats::medianMad($values);

                $this->medians[$column] = $median;
                $this->mads[$column] = $mad ?: 1.0;
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->mads === null or $this->medians === null) {
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
     * Perform the reverse transformation to the samples.
     *
     * @param list<list<mixed>> $samples
     * @throws RuntimeException
     */
    public function reverseTransform(array &$samples) : void
    {
        if ($this->mads === null or $this->medians === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->mads as $column => $mad) {
                $value = &$sample[$column];

                $value *= $mad;

                if ($this->center) {
                    $value += $this->medians[$column];
                }
            }
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
        return 'Robust Standardizer {center: ' . Params::toString($this->center) . ')';
    }
}
