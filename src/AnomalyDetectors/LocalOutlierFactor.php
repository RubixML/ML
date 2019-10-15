<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Ranking;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\EPSILON;

/**
 * Local Outlier Factor
 *
 * Local Outlier Factor (LOF) measures the local deviation of density of a given
 * sample with respect to its *k* nearest neighbors. As such, LOF only considers
 * the local region (or *neighborhood*) of an unknown sample which enables it to
 * detect anomalies within individual clusters of data.
 *
 * References:
 * [1] M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local
 * Outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LocalOutlierFactor implements Estimator, Learner, Ranking, Persistable
{
    use PredictsSingle;
    
    protected const DEFAULT_THRESHOLD = 1.5;

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
     * @var float|null
     */
    protected $contamination;

    /**
     * The k-d tree used for nearest neighbor queries.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected $tree;

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
     * The local outlier factor threshold used by the decision function.
     *
     * @var float|null
     */
    protected $threshold;

    /**
     * @param int $k
     * @param float|null $contamination
     * @param \Rubix\ML\Graph\Trees\Spatial|null $tree
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k = 20, ?float $contamination = null, ?Spatial $tree = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . " to form a local region, $k given.");
        }

        if (isset($contamination) and ($contamination < 0. or $contamination > 0.5)) {
            throw new InvalidArgumentException('Contamination must be'
                . " between 0 and 0.5, $contamination given.");
        }

        $this->k = $k;
        $this->contamination = $contamination;
        $this->tree = $tree ?? new KDTree();
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
        return !$this->tree->bare()
            and $this->kdistances
            and $this->lrds;
    }

    /**
     * Return the base k-d tree instance.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    public function tree() : Spatial
    {
        return $this->tree;
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

        $labels = range(0, $dataset->numRows() - 1);

        $dataset = Labeled::quick($dataset->samples(), $labels);

        $this->tree->grow($dataset);

        $this->kdistances = $this->lrds = [];

        $iHat = $dHat = [];

        foreach ($dataset as $sample) {
            [$samples, $indices, $distances] = $this->tree->nearest($sample, $this->k);

            $iHat[] = $indices;
            $dHat[] = $distances;
            
            $this->kdistances[] = end($distances);
        }

        $this->lrds = array_map([self::class, 'localReachabilityDensity'], $iHat, $dHat);
        
        if (isset($this->contamination)) {
            $lofs = array_map([self::class, 'localOutlierFactor'], $dataset->samples());

            $threshold = Stats::percentile($lofs, 100. - (100. * $this->contamination));
        }

        $this->threshold = $threshold ?? self::DEFAULT_THRESHOLD;
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
        if ($this->tree->bare() or empty($this->lrds)) {
            throw new RuntimeException('Estimator has not been trained.');
        }
        
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        return array_map([self::class, 'localOutlierFactor'], $dataset->samples());
    }

    /**
     * Calculate the local outlier factor of a given sample given its k
     * nearest neighbors.
     *
     * @param array $sample
     * @throws \RuntimeException
     * @return float
     */
    protected function localOutlierFactor(array $sample) : float
    {
        [$samples, $indices, $distances] = $this->tree->nearest($sample, $this->k);

        $lrd = $this->localReachabilityDensity($indices, $distances);

        $ratios = [];
        
        foreach ($indices as $index) {
            $ratios[] = $this->lrds[$index] / $lrd;
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
        $kdistances = [];

        foreach ($indices as $index) {
            $kdistances[] = $this->kdistances[$index];
        }

        $rds = array_map('max', $distances, $kdistances);

        return 1. / (Stats::mean($rds) ?: EPSILON);
    }

    /**
     * The decision function.
     *
     * @param float $score
     * @return int
     */
    protected function decide(float $score) : int
    {
        return $score > $this->threshold ? 1 : 0;
    }
}
