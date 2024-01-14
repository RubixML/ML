<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\RuntimeException;

use const Rubix\ML\EPSILON;

/**
 * Max Absolute Scaler
 *
 * Scale the sample matrix by the maximum absolute value of each feature column
 * independently such that the feature value is between -1 and 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MaxAbsoluteScaler implements Transformer, Stateful, Elastic, Reversible, Persistable
{
    use AutotrackRevisions;

    /**
     * The maximum absolute values for each fitted feature column.
     *
     * @var (int|float)[]|null
     */
    protected ?array $maxabs = null;

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
        return isset($this->maxabs);
    }

    /**
     * Return the maximum absolute values for each feature column.
     *
     * @return (int|float)[]|null
     */
    public function maxabs() : ?array
    {
        return $this->maxabs;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->maxabs = [];

        foreach ($dataset->featureTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $this->maxabs[$column] = -INF;
            }
        }

        $this->update($dataset);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param Dataset $dataset
     */
    public function update(Dataset $dataset) : void
    {
        if ($this->maxabs === null) {
            $this->fit($dataset);

            return;
        }

        foreach ($this->maxabs as $column => $oldMax) {
            $values = $dataset->feature($column);

            $max = max(array_map('abs', $values));

            $max = max($oldMax, $max);

            $this->maxabs[$column] = $max ?: EPSILON;
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
        if ($this->maxabs === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->maxabs as $column => $maxabs) {
                $sample[$column] /= $maxabs;
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
        if ($this->maxabs === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->maxabs as $column => $maxabs) {
                $sample[$column] *= $maxabs;
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
        return 'Max Absolute Scaler';
    }
}
