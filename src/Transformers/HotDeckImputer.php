<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
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

use function is_float;
use function is_nan;
use function in_array;
use function array_column;
use function array_sum;
use function array_rand;
use function round;
use function rand;

/**
 * Hot Deck Imputer
 *
 * A method of imputation that is similar to KNN Imputer but instead of computing a weighted average
 * of the neighbors' feature values, Random Hot Deck picks a value from the neighborhood randomly
 * but optionally weighted by distance. Compared to its KNN counterpart, Hot Deck Imputer is slightly
 * more computationally efficient while satisfying some balancing equations at the same time.
 *
 * **Note:** NaN safe distance kernels, such as Safe Euclidean, are required
 * for continuous features.
 *
 * References:
 * [1] C. Hasler et al. (2015). Balanced k-Nearest Neighbor Imputation.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class HotDeckImputer implements Transformer, Stateful, Persistable
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
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected \Rubix\ML\Graph\Trees\Spatial $tree;

    /**
     * @param int $k
     * @param bool $weighted
     * @param string $categoricalPlaceholder
     * @param \Rubix\ML\Graph\Trees\Spatial|null $tree
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
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
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->tree->bare()) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $randMax = getrandmax();

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

                            $total = array_sum($weights);

                            $phi = $randMax / $total;

                            $max = (int) round($total * $phi);
                        }
                    }

                    $values = array_column($donors, $column);

                    if (isset($weights, $max, $phi)) {
                        $delta = rand(0, $max) / $phi;

                        foreach ($weights as $offset => $weight) {
                            $delta -= $weight;

                            if ($delta <= 0.0) {
                                $value = $values[$offset];

                                break;
                            }
                        }
                    } else {
                        $offset = array_rand($values);

                        $value = $values[$offset];
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
        return "Hot Deck Imputer (k: {$this->k}, weighted: {$this->weighted},"
            . " categorical placeholder: {$this->categoricalPlaceholder},"
            . " tree: {$this->tree})";
    }
}
