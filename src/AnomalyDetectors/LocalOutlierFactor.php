<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Local Outlier Factor
 *
 * Local Outlier Factor (LOF) measures the local deviation of density of a given sample
 * with respect to its k nearest neighbors. As such, LOF only considers the local region
 * of a sample thus enabling it to detect anomalies within individual clusters of data.
 *
 * References:
 * [1] M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local
 * Outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LocalOutlierFactor implements Learner, Online, Persistable
{
    const THRESHOLD = 1.5;

    /**
     * The number of nearest neighbors to consider a local region.
     *
     * @var int
     */
    protected $k;

    /**
     * The percentage of outliers that are assumed to be present in the
     * training set.
     *
     * @var float
     */
    protected $contamination;

    /**
     * The distance kernel to use when computing the distances between two
     * data points.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The training samples.
     *
     * @var array[]
     */
    protected $samples = [
        //
    ];

    /**
     * The precomputed k distances between each training sample and its kth
     * nearest neighbor.
     *
     * @var float[]
     */
    protected $kdistances = [
        //
    ];
 
    /**
     * The precomputed local reachability densities of the training set.
     *
     * @var float[]
     */
    protected $lrds = [
        //
    ];

    /**
     * The local outlier factor offset used by the decision function.
     *
     * @var float|null
     */
    protected $offset;

    /**
     * @param  int  $k
     * @param  float  $contamination
     * @param  \Rubix\ML\Kernels\Distance\Distance  $kernel
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 20, float $contamination = 0.1, ?Distance $kernel = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to form a local region, $k given.");
        }

        if ($contamination < 0.) {
            throw new InvalidArgumentException('Contamination cannot be less'
                . " than 0, $contamination given.");
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->k = $k;
        $this->contamination = $contamination;
        $this->kernel = $kernel;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::DETECTOR;
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
        return $this->offset and $this->samples;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        $this->samples = $this->kdistances = $this->lrds = [];

        $this->partial($dataset);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function partial(Dataset $dataset) : void
    {
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->samples = array_merge($this->samples, $dataset->samples());

        $distances = [];

        foreach ($this->samples as $i => $sample) {
            $distances[] = $d = $this->localRegion($sample);

            $this->kdistances[$i] = end($d);
        }

        foreach ($distances as $i => $row) {
            $this->lrds[$i] = $this->localReachabilityDensity($row);
        }

        $lofs = [];

        foreach ($this->samples as $sample) {
            $lofs[] = $this->localOutlierFactor($sample);
        }

        $shift = Stats::percentile($lofs, 100. * $this->contamination);
        
        $this->offset = self::THRESHOLD + $shift;
    }

    /**
     * Make predictions from a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->samples)) {
            throw new RuntimeException('The learner has not'
                . ' not been trained.');
        };
        
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset as $sample) {
            $lof = $this->localOutlierFactor($sample);

            $predictions[] = $lof > $this->offset ? 1 : 0;
        }

        return $predictions;
    }

    /**
     * Calculate the local outlier factor of a given sample.
     *
     * @param  array  $sample
     * @throws \RuntimeException
     * @return float
     */
    protected function localOutlierFactor(array $sample) : float
    {
        if (empty($this->lrds)) {
            throw new RuntimeException('Local reachability distances have'
                . ' not been computed, must train estimator first.');
        }

        $distances = $this->localRegion($sample);

        $lrd = $this->localReachabilityDensity($distances);

        $lrds = array_intersect_key($this->lrds, $distances);

        $ratios = [];

        foreach ($lrds as $lHat) {
            $ratios[] = $lHat / $lrd;
        }

        return Stats::mean($ratios);
    }

    /**
     * Calculate the local reachability density of a sample given its
     * distances to its k nearest neighbors.
     *
     * @param  array  $distances
     * @throws \RuntimeException
     * @return float
     */
    protected function localReachabilityDensity(array $distances) : float
    {
        if (empty($this->kdistances)) {
            throw new RuntimeException('K distances have not been computed,'
                . ' must train estimator first.');
        }

        $kdistances = array_intersect_key($this->kdistances, $distances);

        $rds = array_map('max', $distances, $kdistances);

        $mean = Stats::mean($rds);

        return 1. / ($mean ?: self::EPSILON);
    }

    /**
     * Find the K nearest neighbors to the given sample vector using the
     * brute force method.
     *
     * @param  array  $sample
     * @throws \RuntimeException
     * @return array
     */
    protected function localRegion(array $sample) : array
    {
        $distances = [];

        foreach ($this->samples as $neighbor) {
            $distances[] = $this->kernel->compute($sample, $neighbor);
        }

        asort($distances);

        return array_slice($distances, 0, $this->k, true);
    }
}
