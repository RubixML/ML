<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * KNN Regressor
 *
 * A version of K Nearest Neighbors that uses the mean outcome of K
 * nearest data points to make continuous valued predictions suitable
 * for regression problems.
 *
 * > **Note**: K Nearest Neighbors is considered a *lazy* learning
 * estimator because it does the majority of its computation at
 * prediction time.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KNNRegressor implements Online, Persistable
{
    /**
     * The number of neighbors to consider when making a prediction.
     *
     * @var int
     */
    protected $k;

    /**
     * The distance kernel to use when computing the distances.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * Should we use the inverse distances as confidence scores when
     * making predictions?
     *
     * @var bool
     */
    protected $weighted;

    /**
     * The training samples.
     *
     * @var array
     */
    protected $samples = [
        //
    ];

    /**
     * The training labels.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * @param int $k
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param bool $weighted
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 3, ?Distance $kernel = null, bool $weighted = true)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to make a prediction, $k given.");
        }

        $this->k = $k;
        $this->kernel = $kernel ?? new Euclidean();
        $this->weighted = $weighted;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::REGRESSOR;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->kernel->compatibility();
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->samples and $this->labels;
    }

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        $this->samples = $this->labels = [];

        $this->partial($dataset);
    }

    /**
     * Store the sample and outcome arrays. No other work to be done as this is
     * a lazy learning algorithm.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->samples = array_merge($this->samples, $dataset->samples());
        $this->labels = array_merge($this->labels, $dataset->labels());
    }

    /**
     * Make a prediction based on the nearest neighbors.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->samples) or empty($this->labels)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset as $sample) {
            [$distances, $labels] = $this->nearest($sample);

            if ($this->weighted) {
                $weights = [];

                foreach ($distances as $i => $distance) {
                    $weights[] = 1. / (1. + $distance);
                }

                $outcome = Stats::weightedMean($labels, $weights);
            } else {
                $outcome = Stats::mean($labels);
            }

            $predictions[] = $outcome;
        }

        return $predictions;
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

        $labels = array_values(array_intersect_key($this->labels, $distances));

        return [$distances, $labels];
    }
}
