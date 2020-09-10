<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;
use Stringable;

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
class VarianceThresholdFilter implements Transformer, Stateful, Stringable
{
    /**
     * The minimum number of features to select from the dataset.
     *
     * @var int
     */
    protected $minFeatures;

    /**
     * The variances of the dropped feature columns.
     *
     * @var float[]|null
     */
    protected $variances;

    /**
     * @param int $minFeatures
     * @throws \InvalidArgumentException
     */
    public function __construct(int $minFeatures)
    {
        if ($minFeatures < 1) {
            throw new InvalidArgumentException('Min features must be'
                . " greater than 0, $minFeatures given.");
        }

        $this->minFeatures = $minFeatures;
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
     * Return the variances of the dropped feature columns.
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
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $variances = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $variances[$column] = Stats::variance($dataset->column($column));
            }
        }

        asort($variances);

        $k = max(0, $dataset->numColumns() - $this->minFeatures);

        $this->variances = array_slice($variances, 0, $k, true);
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
            $sample = array_values(array_diff_key($sample, $this->variances));
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Variance Threshold Filter (max_features: {$this->minFeatures})";
    }
}
