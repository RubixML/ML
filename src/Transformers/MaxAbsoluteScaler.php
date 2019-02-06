<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataType;
use RuntimeException;

/**
 * Max Absolute Scaler
 * 
 * Scale the sample matrix by the maximum absolute value of each feature
 * column independently such that the feature will be between -1 and 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MaxAbsoluteScaler implements Elastic
{
    /**
     * The maximum absolute values for each fitted feature column.
     * 
     * @var array|null
     */
    protected $maxabs;

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
     * @return array|null
     */
    public function maxabs() : ?array
    {
        return $this->maxabs;
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->maxabs = [];

        foreach ($dataset->types() as $column => $type) {
            if ($type === DataType::CONTINUOUS) {
                $this->maxabs[$column] = -INF;
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
        if (is_null($this->maxabs)) {
            $this->fit($dataset);
            return;
        }

        foreach ($this->maxabs as $column => $oldMax) {
             $values = $dataset->column($column);

             $max = max(array_map('abs', $values));

             $max = max($oldMax, $max);

             $this->maxabs[$column] = $max ?: self::EPSILON;
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
        if (is_null($this->maxabs)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($sample as $column => &$feature) {
                $feature /= $this->maxabs[$column];
            }
        }
    }
}
