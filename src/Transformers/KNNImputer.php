<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Kernels\Distance\NaNSafe;
use Rubix\ML\Kernels\Distance\SafeEuclidean;
use InvalidArgumentException;
use RuntimeException;

/**
 * KNN Imputer
 *
 * An unsupervised *hot deck* imputer that replaces NaN values in numerical
 * datasets with the weighted average of the sample's k nearest neighbors.
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
     * The NaN safe distance kernel to use when computing the distances.
     *
     * @var \Rubix\ML\Kernels\Distance\NaNSafe
     */
    protected $kernel;

    /**
     * The donor samples from the fitted training set.
     *
     * @var array
     */
    protected $samples = [
        //
    ];

    /**
     * @param int $k
     * @param bool $weighted
     * @param \Rubix\ML\Kernels\Distance\NaNSafe|null $kernel
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 5, bool $weighted = true, ?NaNSafe $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to make a prediction, $k given.");
        }

        $this->k = $k;
        $this->weighted = $weighted;
        $this->kernel = $kernel ?? new SafeEuclidean();
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        $this->samples = [];

        $this->update($dataset);
    }

    /**
     * Update the fitting of the transformer.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function update(Dataset $dataset) : void
    {
        if (!$dataset->homogeneous() or $dataset->columnType(0) !== DataType::CONTINUOUS) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }

        $donors = [];

        foreach ($dataset->samples() as $sample) {
            foreach ($sample as $value) {
                if (is_float($value) and is_nan($value)) {
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
     * @param array $samples
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
                if (is_float($value) and is_nan($value)) {
                    if (empty($neighbors)) {
                        [$neighbors, $distances] = $this->nearest($sample);
                    }

                    $values = array_column($neighbors, $column);

                    if ($this->weighted) {
                        $weights = [];
        
                        foreach ($distances as $i => $distance) {
                            $weights[] = 1. / (1. + $distance);
                        }
        
                        $value = Stats::weightedMean($values, $weights);
                    } else {
                        $value = Stats::mean($values);
                    }
                }
            }
        }
    }

    /**
     * Find the K nearest neighbors to the given sample vector using
     * the brute force method.
     *
     * @param array $sample
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

        $neighbors = array_values(array_intersect_key($this->samples, $distances));

        return [$neighbors, $distances];
    }
}
