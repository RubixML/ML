<?php

namespace Rubix\ML\AnomalyDetectors;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Ranking;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\RanksSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\warn_deprecated;

use const Rubix\ML\LOG_EPSILON;

/**
 * Loda
 *
 * *Lightweight Online Detector of Anomalies* uses a series of sparse random
 * projection vectors to produce scalar inputs to an ensemble of unique
 * one-dimensional equi-width histograms. The histograms are then used to estimate
 * the probability density of an unknown sample during inference.
 *
 * References:
 * [1] T. Pevný. (2015). Loda: Lightweight on-line detector of anomalies.
 * [2] L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Loda implements Estimator, Learner, Online, Scoring, Ranking, Persistable
{
    use AutotrackRevisions, PredictsSingle, RanksSingle;

    /**
     * The minimum number of histogram bins.
     *
     * @var int
     */
    protected const MIN_BINS = 3;

    /**
     * The minimum dimensionality required to produce sparse projections.
     *
     * @var int
     */
    protected const MIN_SPARSE_DIMENSIONS = 3;

    /**
     * The number of projection/histogram pairs in the ensemble.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The number of bins in each equi-width histogram.
     *
     * @var int|null
     */
    protected $bins;

    /**
     * Should we calculate the equi-width bin count on the fly?
     *
     * @var bool
     */
    protected $fitBins;

    /**
     * The proportion of outliers that are assumed to be present in the
     * training set.
     *
     * @var float
     */
    protected $contamination;

    /**
     * The sparse random projection matrix.
     *
     * @var \Tensor\Matrix|null
     */
    protected $r;

    /**
     * The edges and bin counts of each histogram.
     *
     * @var array[]
     */
    protected $histograms = [
        //
    ];

    /**
     * The minimum negative log likelihood score necessary to flag an anomaly.
     *
     * @var float|null
     */
    protected $threshold;

    /**
     * The number of samples that have been learned so far.
     *
     * @var int
     */
    protected $n = 0;

    /**
     * Estimate the number of bins from the number of samples in a dataset.
     *
     * @param int $n
     * @return int
     */
    public static function estimateBins(int $n) : int
    {
        return (int) round(log($n, 2)) + 1;
    }

    /**
     * @param int $estimators
     * @param int|null $bins
     * @param float $contamination
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $estimators = 100, ?int $bins = null, float $contamination = 0.1)
    {
        if ($estimators < 1) {
            throw new InvalidArgumentException('Number of estimators'
                . " must be greater than 0, $estimators given.");
        }

        if (isset($bins) and $bins < self::MIN_BINS) {
            throw new InvalidArgumentException('Bins must be greater'
                . ' than ' . self::MIN_BINS . ", $bins given.");
        }

        if ($contamination < 0.0 or $contamination > 0.5) {
            throw new InvalidArgumentException('Contamination must be'
                . " between 0 and 0.5, $contamination given.");
        }

        $this->estimators = $estimators;
        $this->bins = $bins;
        $this->fitBins = is_null($bins);
        $this->contamination = $contamination;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @internal
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::anomalyDetector();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'estimators' => $this->estimators,
            'bins' => $this->bins,
            'contamination' => $this->contamination,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->r and $this->histograms and $this->threshold;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        [$m, $n] = $dataset->shape();

        if ($this->fitBins) {
            $this->bins = max(self::estimateBins($m), self::MIN_BINS);
        }

        $this->r = Matrix::gaussian($n, $this->estimators);

        if ($n >= self::MIN_SPARSE_DIMENSIONS) {
            $mask = Matrix::rand($n, $this->estimators)
                ->less(sqrt($n) / $n);

            $this->r = $this->r->multiply($mask);
        }

        $projections = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        foreach ($projections->asArray() as $values) {
            $edges = Vector::linspace(min($values), max($values), $this->bins - 1)->asArray();

            $counts = array_fill(0, count($edges), 0);

            foreach ($values as $value) {
                foreach ($edges as $k => $edge) {
                    if ($value <= $edge) {
                        ++$counts[$k];

                        continue 2;
                    }
                }
            }

            $this->histograms[] = [$edges, $counts];
        }

        $this->n = $m;

        $densities = $this->densities($projections);

        $this->threshold = Stats::quantile($densities, 1.0 - $this->contamination);
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$this->r or !$this->histograms or !$this->threshold) {
            $this->train($dataset);

            return;
        }

        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new DatasetHasDimensionality($dataset, $this->r->m()),
        ])->check();

        $projections = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        foreach ($projections->asArray() as $i => $values) {
            [$edges, $counts] = $this->histograms[$i];

            foreach ($values as $value) {
                foreach ($edges as $k => $edge) {
                    if ($value <= $edge) {
                        ++$counts[$k];

                        continue 2;
                    }
                }
            }

            $this->histograms[$i] = [$edges, $counts];
        }

        $n = $dataset->numRows();

        $this->n += $n;

        $densities = $this->densities($projections);

        $threshold = Stats::quantile($densities, 1.0 - $this->contamination);

        $weight = $n / $this->n;

        $this->threshold = (1.0 - $weight) * $this->threshold + $weight * $threshold;
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([$this, 'decide'], $this->score($dataset));
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<float>
     */
    public function score(Dataset $dataset) : array
    {
        if (!$this->r or !$this->histograms or !$this->threshold) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->r->m())->check();

        $projections = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        return $this->densities($projections);
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @deprecated
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return list<float>
     */
    public function rank(Dataset $dataset) : array
    {
        warn_deprecated('Rank() is deprecated, use score() instead.');

        return $this->score($dataset);
    }

    /**
     * Estimate the probability density function of each 1-dimensional projection
     * using the histograms generated during training.
     *
     * @param \Tensor\Matrix $projections
     * @return list<float>
     */
    protected function densities(Matrix $projections) : array
    {
        $densities = array_fill(0, $projections->n(), 0.0);

        foreach ($projections->asArray() as $i => $values) {
            [$edges, $counts] = $this->histograms[$i];

            foreach ($values as $j => $value) {
                foreach ($edges as $k => $edge) {
                    if ($value <= $edge) {
                        $count = $counts[$k];

                        $densities[$j] += $count > 0
                            ? -log($count / $this->n)
                            : -LOG_EPSILON;

                        break;
                    }
                }
            }
        }

        foreach ($densities as &$density) {
            $density /= $this->estimators;
        }

        return $densities;
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

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Loda (' . Params::stringify($this->params()) . ')';
    }
}
