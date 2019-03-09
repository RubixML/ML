<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Ranking;
use Rubix\ML\Persistable;
use Rubix\ML\Graph\KDTree;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * K-d LOF
 *
 * A k-d tree accelerated version of Local Outlier Factor which benefits
 * from fast nearest neighbors search.
 *
 * References:
 * [1] M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local
 * Outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDLOF extends KDTree implements Learner, Ranking, Persistable
{
    protected const THRESHOLD = 1.5;

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
     * @param int $k
     * @param float $contamination
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param int $maxLeafSize
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $k = 20,
        float $contamination = 0.1,
        ?Distance $kernel = null,
        int $maxLeafSize = 30
    ) {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to form a local region, $k given.");
        }

        if ($contamination < 0.) {
            throw new InvalidArgumentException('Contamination cannot be less'
                . " than 0, $contamination given.");
        }

        $this->k = $k;
        $this->contamination = $contamination;

        parent::__construct($maxLeafSize, $kernel);
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::ANOMALY_DETECTOR;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !$this->bare() and $this->lrds;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $samples = $dataset->samples();

        $dataset = Labeled::quick($samples, array_keys($samples));

        $this->grow($dataset);

        $this->kdistances = $this->lrds = [];

        $indices = $distances = [];

        foreach ($dataset as $sample) {
            [$iHat, $dHat] = $this->nearest($sample, $this->k);

            $distances[] = $dHat;
            $indices[] = $iHat;
            $this->kdistances[] = end($dHat);
        }

        foreach ($distances as $i => $row) {
            $this->lrds[] = $this->localReachabilityDensity($indices[$i], $row);
        }

        $lofs = [];

        foreach ($dataset as $sample) {
            $lofs[] = $this->localOutlierFactor($sample);
        }

        $shift = Stats::percentile($lofs, 100. * $this->contamination);
        
        $this->offset = self::THRESHOLD + $shift;
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([self::class, 'decide'], $this->rank($dataset));
    }

    /**
     * Apply an arbitrary unnormalized scoring function over the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function rank(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('The learner has not'
                . ' been trained.');
        }
        
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        return array_map([self::class, 'localOutlierFactor'], $dataset->samples());
    }

    /**
     * Calculate the local outlier factor of a given sample.
     *
     * @param array $sample
     * @throws \RuntimeException
     * @return float
     */
    protected function localOutlierFactor(array $sample) : float
    {
        if (empty($this->lrds)) {
            throw new RuntimeException('Local reachability distances have'
                . ' not been computed, must train estimator first.');
        }

        [$indices, $distances] = $this->nearest($sample, $this->k);

        $lrd = $this->localReachabilityDensity($indices, $distances);

        $lrds = [];
        
        foreach ($indices as $index) {
            $lrds[] = $this->lrds[$index];
        }

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
     * @param array $indices
     * @param array $distances
     * @throws \RuntimeException
     * @return float
     */
    protected function localReachabilityDensity(array $indices, array $distances) : float
    {
        if (empty($this->kdistances)) {
            throw new RuntimeException('K distances have not been computed,'
                . ' must train estimator first.');
        }

        $kdistances = [];

        foreach ($indices as $index) {
            $kdistances[] = $this->kdistances[$index];
        }

        $rds = array_map('max', $distances, $kdistances);

        $mean = Stats::mean($rds);

        return 1. / ($mean ?: self::EPSILON);
    }

    /**
     * The decision function.
     *
     * @param float $score
     * @return int
     */
    protected function decide(float $score) : int
    {
        return $score > $this->offset ? 1 : 0;
    }
}
