<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Ranking;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\Tensor\Matrix;
use Rubix\Tensor\Vector;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Traits\RankSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\EPSILON;
use const Rubix\ML\LOG_EPSILON;

/**
 * Loda
 *
 * *Lightweight Online Detector of Anomalies* uses a sparse random projection matrix
 * to produce input to an ensemble of unique one dimensional equi-width histograms
 * able to estimate the probability density of a potential anomaly. The anomaly score
 * is defined as the negative log likelihood of a sample being an outlier. Thus,
 * samples with low probability density will be given a high anomaly score.
 *
 * References:
 * [1] T. Pevný. (2015). Loda: Lightweight on-line detector of anomalies.
 * [2] L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Loda implements Estimator, Learner, Online, Ranking, Persistable
{
    use PredictsSingle, RankSingle;
    
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
     * The minimum negative log likelihood score necessary to flag an anomaly.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The sparse random projection matrix.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $r;

    /**
     * The edges, and bin counts of each histogram.
     *
     * @var array[]|null
     */
    protected $histograms;

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
     * @param float $threshold
     * @throws \InvalidArgumentException
     */
    public function __construct(int $estimators = 100, ?int $bins = null, float $threshold = 10.)
    {
        if ($estimators < 1) {
            throw new InvalidArgumentException('At least 1 histogram is'
                . " requied to make a prediction, $estimators given.");
        }

        if (isset($bins) and $bins < 1) {
            throw new InvalidArgumentException('The number of bins cannot'
                . " be less than 1, $bins given.");
        }

        if ($threshold < 0.) {
            throw new InvalidArgumentException('Threshold must be'
                . " greater than 0, $threshold given.");
        }

        $this->estimators = $estimators;
        $this->bins = $bins;
        $this->fitBins = is_null($bins);
        $this->threshold = $threshold;
    }

    /**
     * Return the integer encoded estimator type.
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
        return $this->r and $this->histograms;
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

        [$m, $n] = $dataset->shape();

        $this->r = Matrix::gaussian($n, $this->estimators);

        if ($n >= self::MIN_SPARSE_DIMENSIONS) {
            $mask = Matrix::rand($n, $this->estimators)
                ->less(sqrt($n) / $n);

            $this->r = $this->r->multiply($mask);
        }

        $projections = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        if ($this->fitBins) {
            $this->bins = self::estimateBins($m);
        }

        foreach ($projections as $values) {
            $start = min($values) - EPSILON;
            $end = max($values) + EPSILON;

            $edges = Vector::linspace($start, $end, $this->bins + 1)->asArray();

            $edges[] = INF;

            $counts = array_fill(0, $this->bins + 2, 0);

            $interior = array_slice($edges, 1, $this->bins, true);

            foreach ($values as $value) {
                foreach ($interior as $k => $edge) {
                    if ($value < $edge) {
                        $counts[$k]++;

                        continue 2;
                    }
                }

                $counts[$this->bins]++;
            }

            $this->histograms[] = [$edges, $counts];
        }

        $this->n = $m;
    }

    /**
     * Perform a partial train on the learner.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$this->r or !$this->histograms) {
            $this->train($dataset);

            return;
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $projections = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        foreach ($projections as $i => $values) {
            [$edges, $counts] = $this->histograms[$i];

            $interior = array_slice($edges, 1, $this->bins, true);

            foreach ($values as $value) {
                foreach ($interior as $k => $edge) {
                    if ($value < $edge) {
                        $counts[$k]++;

                        continue 2;
                    }
                }

                $counts[$this->bins]++;
            }

            $this->histograms[$i] = [$edges, $counts];
        }

        $this->n += $dataset->numRows();
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
        return array_map([$this, 'decide'], $this->rank($dataset));
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
        if (!$this->r or !$this->histograms) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $projections = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        $densities = array_fill(0, $projections->n(), 0.);
    
        foreach ($projections as $i => $values) {
            [$edges, $counts] = $this->histograms[$i];

            foreach ($values as $j => $value) {
                foreach ($edges as $k => $edge) {
                    if ($value < $edge) {
                        $count = $counts[$k];

                        $densities[$j] += $count > 0
                            ? -log($counts[$k] / $this->n)
                            : -LOG_EPSILON;

                        break 1;
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
}
