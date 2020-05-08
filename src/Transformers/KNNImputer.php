<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\NaNSafe;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\SafeEuclidean;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\argmax;
use function in_array;
use function is_null;

/**
 * KNN Imputer
 *
 * An unsupervised imputer that replaces missing values in datasets with the distance-weighted
 * average of the samples' *k* nearest neighbors' values.
 *
 * **Note:** NaN safe distance kernels, such as Safe Euclidean, are required for continuous
 * features.
 *
 * References:
 * [1] O. Troyanskaya et al. (2001). Missing value estimation methods for DNA microarrays.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KNNImputer implements Transformer, Stateful
{
    /**
     * The number of neighbors to consider when imputing a value.
     *
     * @var int
     */
    protected $k;

    /**
     * Should we use the inverse distances as confidence scores when imputing
     * values.
     *
     * @var bool
     */
    protected $weighted;

    /**
     * The placeholder category that denotes missing values.
     *
     * @var string
     */
    protected $categoricalPlaceholder;

    /**
     * The spatial tree used to run nearest neighbor searches.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected $tree;

    /**
     * The data types of the fitted feature columns.
     *
     * @var \Rubix\ML\DataType[]|null
     */
    protected $types;

    /**
     * @param int $k
     * @param bool $weighted
     * @param string $categoricalPlaceholder
     * @param \Rubix\ML\Graph\Trees\Spatial|null $tree
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $k = 5,
        bool $weighted = true,
        string $categoricalPlaceholder = '?',
        ?Spatial $tree = null
    ) {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to impute a value, $k given.");
        }

        if ($tree and in_array(DataType::continuous(), $tree->kernel()->compatibility())) {
            $kernel = $tree->kernel();

            if (!$kernel instanceof NaNSafe) {
                throw new InvalidArgumentException('Continuous distance kernels'
                    . ' must implement the NaNSafe interface.');
            }
        }

        $this->k = $k;
        $this->weighted = $weighted;
        $this->categoricalPlaceholder = $categoricalPlaceholder;
        $this->tree = new BallTree(30, new SafeEuclidean());
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->tree->kernel()->compatibility();
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return !$this->tree->bare();
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::check($dataset, $this);
        
        $donors = [];

        foreach ($dataset->samples() as $sample) {
            foreach ($sample as $value) {
                if (is_float($value)) {
                    if (is_nan($value)) {
                        continue 2;
                    }
                } else {
                    if ($value === $this->categoricalPlaceholder) {
                        continue 2;
                    }
                }
            }

            $donors[] = $sample;
        }

        if (empty($donors)) {
            throw new RuntimeException('No complete donors found.');
        }

        $labels = array_fill(0, count($donors), 0);

        $dataset = Labeled::quick($donors, $labels);

        $this->tree->grow($dataset);

        $this->types = $dataset->columnTypes();
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->tree->bare() or is_null($this->types)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $neighbors = $distances = [];

            foreach ($sample as $column => &$value) {
                if (is_float($value) && is_nan($value) or $value === $this->categoricalPlaceholder) {
                    if (empty($neighbors)) {
                        [$neighbors, $labels, $distances] = $this->tree->nearest($sample, $this->k);
                    }

                    $values = array_column($neighbors, $column);

                    $type = $this->types[$column];

                    $value = $this->impute($values, $distances, $type);
                }
            }
        }
    }

    /**
     * Choose a value to impute from a given set of values.
     *
     * @param (string|int|float)[] $values
     * @param float[] $distances
     * @param \Rubix\ML\DataType $type
     * @return string|int|float
     */
    protected function impute(array $values, array $distances, DataType $type)
    {
        if ($type->isContinuous()) {
            if ($this->weighted) {
                $weights = [];

                foreach ($distances as $distance) {
                    $weights[] = 1.0 / (1.0 + $distance);
                }

                $value = Stats::weightedMean($values, $weights);
            } else {
                $value = Stats::mean($values);
            }
        } else {
            if ($this->weighted) {
                $weights = array_fill_keys($values, 0.0);

                foreach ($distances as $i => $distance) {
                    $weights[$values[$i]] += 1.0 / (1.0 + $distance);
                }
            } else {
                $weights = array_count_values($values);
            }

            $value = argmax($weights);
        }

        return $value;
    }
}
