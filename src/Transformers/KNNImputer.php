<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Kernels\Distance\NaNSafe;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\SafeEuclidean;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\argmax;
use function array_slice;

/**
 * KNN Imputer
 *
 * An unsupervised imputer that replaces missing values in datasets with the
 * weighted average according to the sample's k nearest neighbors.
 *
 * **Note:** NaN safe distance kernels, such as Safe Euclidean, are required
 * for continuous features.
 *
 * References:
 * [1] O. Troyanskaya et al. (2001). Missing value estimation methods for
 * DNA microarrays.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KNNImputer implements Transformer, Stateful, Elastic
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
     * The distance kernel to use when computing the distances.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The categorical placeholder variable denoting the category that
     * contains missing values.
     *
     * @var mixed
     */
    protected $placeholder;

    /**
     * The donor samples from the fitted training set.
     *
     * @var array[]
     */
    protected $samples = [
        //
    ];

    /**
     * @param int $k
     * @param bool $weighted
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param string $placeholder
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $k = 5,
        bool $weighted = true,
        ?Distance $kernel = null,
        string $placeholder = '?'
    ) {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to make a prediction, $k given.");
        }

        if ($kernel and in_array(DataType::CONTINUOUS, $kernel->compatibility())) {
            if (!$kernel instanceof NaNSafe) {
                throw new InvalidArgumentException('Continuous distance kernels'
                    . ' must be NaN safe.');
            }
        }

        $this->k = $k;
        $this->weighted = $weighted;
        $this->kernel = $kernel ?? new SafeEuclidean();
        $this->placeholder = $placeholder;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->kernel->compatibility();
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return !empty($this->samples);
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     * @throws \RuntimeException
     */
    public function fit(Dataset $dataset) : void
    {
        $this->samples = [];

        $this->update($dataset);

        if (empty($this->samples)) {
            throw new RuntimeException('No complete donor samples'
                . ' could be found.');
        }
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     * @throws \InvalidArgumentException
     */
    public function update(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::check($dataset, $this);
        
        $donors = [];

        foreach ($dataset->samples() as $sample) {
            foreach ($sample as $value) {
                if ((is_float($value) and is_nan($value)) or $value === $this->placeholder) {
                    continue 2;
                }
            }

            $donors[] = $sample;
        }

        $this->samples = array_merge($this->samples, $donors);
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (empty($this->samples)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $neighbors = $distances = [];

            foreach ($sample as $column => &$value) {
                if ((is_float($value) and is_nan($value)) or $value === $this->placeholder) {
                    if (empty($neighbors)) {
                        [$neighbors, $distances] = $this->nearest($sample);
                    }

                    $values = array_column($neighbors, $column);

                    $value = $this->impute($values, $distances);
                }
            }
        }
    }

    /**
     * Choose a value to impute from a given set of values.
     *
     * @param (string|int|float)[] $values
     * @param float[] $distances
     * @throws \RuntimeException
     * @return string|int|float
     */
    protected function impute(array $values, array $distances)
    {
        if (empty($values)) {
            throw new RuntimeException('Cannot impute because of 0 donors.');
        }

        if (is_string($values[0])) {
            if ($this->weighted) {
                $weights = array_fill_keys($values, 0.);

                foreach ($distances as $i => $distance) {
                    $weights[$values[$i]] += 1. / (1. + $distance);
                }
            } else {
                $weights = array_count_values($values);
            }

            $value = argmax($weights);
        } else {
            if ($this->weighted) {
                $weights = [];

                foreach ($distances as $distance) {
                    $weights[] = 1. / (1. + $distance);
                }

                $value = Stats::weightedMean(array_values($values), $weights);
            } else {
                $value = Stats::mean($values);
            }
        }

        return $value;
    }

    /**
     * Find the K nearest neighbors to the given sample vector using
     * the brute force method.
     *
     * @param mixed[] $sample
     * @return array[]
     */
    protected function nearest(array $sample) : array
    {
        $distances = [];

        foreach ($this->samples as $neighbor) {
            $distances[] = $this->kernel->compute($sample, $neighbor);
        }

        asort($distances);

        $distances = array_slice($distances, 0, $this->k, true);

        $neighbors = array_intersect_key($this->samples, $distances);

        return [$neighbors, $distances];
    }
}
