<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\NaNSafe;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\SafeEuclidean;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\argmax;
use function is_float;
use function is_nan;
use function in_array;
use function array_column;
use function array_count_values;
use function array_fill_keys;
use function array_unique;

/**
 * KNN Imputer
 *
 * An unsupervised imputer that replaces missing values in datasets with the distance-weighted
 * average of the samples' *k* nearest neighbors' values. The average for a continuous feature
 * column is defined as the mean of the values of each donor sample while average is defined as
 * the most frequent for categorical features.
 *
 * **Note:** Requires NaN-safe distance kernels, such as Safe Euclidean, for continuous features.
 *
 * References:
 * [1] O. Troyanskaya et al. (2001). Missing value estimation methods for DNA microarrays.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KNNImputer implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

    /**
     * The number of donor samples to consider when imputing a value.
     *
     * @var int
     */
    protected int $k;

    /**
     * Should the imputed values be sampled from a distribution weighted by distance?
     *
     * @var bool
     */
    protected bool $weighted;

    /**
     * The placeholder category that denotes missing values.
     *
     * @var string
     */
    protected string $categoricalPlaceholder;

    /**
     * The spatial tree used to run nearest neighbor searches.
     *
     * @var Spatial
     */
    protected Spatial $tree;

    /**
     * The data types of the fitted feature columns.
     *
     * @var \Rubix\ML\DataType[]|null
     */
    protected ?array $types = null;

    /**
     * @param int $k
     * @param bool $weighted
     * @param string $categoricalPlaceholder
     * @param Spatial|null $tree
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $k = 5,
        bool $weighted = false,
        string $categoricalPlaceholder = '?',
        ?Spatial $tree = null
    ) {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 donor is required'
                . " to impute a value, $k given.");
        }

        if ($tree and in_array(DataType::continuous(), $tree->kernel()->compatibility())) {
            $kernel = $tree->kernel();

            if (!$kernel instanceof NaNSafe) {
                throw new InvalidArgumentException('Continuous distance kernels'
                    . ' must implement the NaNSafe interface.');
            }
        }

        if (empty($categoricalPlaceholder)) {
            throw new InvalidArgumentException('Categorical placeholder cannot be empty.');
        }

        $this->k = $k;
        $this->weighted = $weighted;
        $this->categoricalPlaceholder = $categoricalPlaceholder;
        $this->tree = $tree ?? new BallTree(30, new SafeEuclidean());
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
     * @param Dataset $dataset
     * @throws RuntimeException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

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
            throw new RuntimeException('No complete donors found in dataset.');
        }

        $labels = array_fill(0, count($donors), '');

        $this->tree->grow(Labeled::quick($donors, $labels));

        $this->types = $dataset->featureTypes();
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->tree->bare() or $this->types === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $donors = [];

            foreach ($sample as $column => &$value) {
                if (is_float($value) && is_nan($value) or $value === $this->categoricalPlaceholder) {
                    if (empty($donors)) {
                        [$donors, $labels, $distances] = $this->tree->nearest($sample, $this->k);

                        if ($this->weighted) {
                            $weights = [];

                            foreach ($distances as $distance) {
                                $weights[] = 1.0 / (1.0 + $distance);
                            }
                        }
                    }

                    $values = array_column($donors, $column);

                    $type = $this->types[$column];

                    switch ($type) {
                        case DataType::continuous():
                            if (isset($weights)) {
                                $value = Stats::weightedMean($values, $weights);
                            } else {
                                $value = Stats::mean($values);
                            }

                            break;

                        case DataType::categorical():
                        default:
                            if (isset($weights)) {
                                $scores = array_fill_keys(array_unique($values), 0.0);

                                foreach ($weights as $i => $weight) {
                                    $scores[$values[$i]] += $weight;
                                }
                            } else {
                                $scores = array_count_values($values);
                            }

                            $value = argmax($scores);

                            break;
                    }
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
        return "KNN Imputer (k: {$this->k}, weighted: {$this->weighted},"
            . " categorical placeholder: {$this->categoricalPlaceholder},"
            . " tree: {$this->tree})";
    }
}
