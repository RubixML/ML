<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;

use function is_null;

/**
 * Variance Threshold Filter
 *
 * A type of feature selector that selects features with the greatest variance.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class VarianceThresholdFilter implements Transformer, Stateful
{
    /**
     * A type of feature selector that selects the top *k* features with the greatest variance.
     *
     * @var int
     */
    protected $maxFeatures;

    /**
     * The variances of the selected feature columns.
     *
     * @var float[]|null
     */
    protected $variances;

    /**
     * @param int $maxFeatures
     * @throws \InvalidArgumentException
     */
    public function __construct(int $maxFeatures)
    {
        if ($maxFeatures < 1) {
            throw new InvalidArgumentException('Maximum features'
                . " must be greater than 0, $maxFeatures given.");
        }

        $this->maxFeatures = $maxFeatures;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return \Rubix\ML\DataType[]
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
        return isset($this->variances);
    }

    /**
     * Return the offsets of the columns that were selected during fitting.
     *
     * @return int[]
     */
    public function selected() : array
    {
        return array_keys($this->variances ?: []);
    }

    /**
     * Return the variances of the selected feature columns.
     *
     * @return float[]|null
     */
    public function variances() : ?array
    {
        return $this->variances;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::check($dataset, $this);

        $variances = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $variances[$column] = Stats::variance($dataset->column($column));
            }
        }

        arsort($variances);

        $this->variances = array_slice($variances, 0, $this->maxFeatures, true);
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->variances)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $sample = array_values(array_intersect_key($sample, $this->variances));
        }
    }
}
