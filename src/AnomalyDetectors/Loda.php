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
use Rubix\ML\Other\Traits\RankSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\array_transpose;
use function array_slice;

use const Rubix\ML\EPSILON;
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
class Loda implements Estimator, Learner, Online, Ranking, Persistable
{
    use PredictsSingle, RankSingle;
    
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
     * The edges, and bin counts of each histogram.
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
     * @throws \InvalidArgumentException
     */
    public function __construct(int $estimators = 100, ?int $bins = null, float $contamination = 0.1)
    {
        if ($estimators < 1) {
            throw new InvalidArgumentException('At least 1 histogram is'
                . " requied to make a prediction, $estimators given.");
        }

        if (isset($bins) and $bins < 1) {
            throw new InvalidArgumentException('The number of bins cannot'
                . " be less than 1, $bins given.");
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
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::anomalyDetector();
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
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
        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        [$m, $n] = $dataset->shape();

        if ($this->fitBins) {
            $this->bins = self::estimateBins($m);
        }

        $this->r = Matrix::gaussian($n, $this->estimators);

        if ($n >= self::MIN_SPARSE_DIMENSIONS) {
            $mask = Matrix::rand($n, $this->estimators)
                ->less(sqrt($n) / $n);

            $this->r = $this->r->multiply($mask);
        }

        $projections = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->asArray();

        foreach (array_transpose($projections) as $values) {
            $start = min($values) - EPSILON;
            $end = max($values) + EPSILON;

            $edges = Vector::linspace($start, $end, $this->bins + 1)->asArray();

            $edges[] = INF;

            $counts = array_fill(0, $this->bins + 2, 0);

            $interior = array_slice($edges, 1, $this->bins, true);

            foreach ($values as $value) {
                foreach ($interior as $k => $edge) {
                    if ($value < $edge) {
                        ++$counts[$k];

                        continue 2;
                    }
                }

                ++$counts[$this->bins];
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

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        $projections = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->asArray();

        foreach (array_transpose($projections) as $i => $values) {
            [$edges, $counts] = $this->histograms[$i];

            $interior = array_slice($edges, 1, $this->bins, true);

            foreach ($values as $value) {
                foreach ($interior as $k => $edge) {
                    if ($value < $edge) {
                        ++$counts[$k];

                        continue 2;
                    }
                }

                ++$counts[$this->bins];
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
     * @return int[]
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([$this, 'decide'], $this->rank($dataset));
    }

    /**
     * Apply an arbitrary unnormalized scoring function over the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return float[]
     */
    public function rank(Dataset $dataset) : array
    {
        if (!$this->r or !$this->histograms or !$this->threshold) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $projections = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->asArray();

        return $this->densities($projections);
    }

    /**
     * Estimate the probability density function of each 1-dimensional projection
     * using the histograms generated during training.
     *
     * @param array[] $projections
     * @return float[]
     */
    protected function densities(array $projections) : array
    {
        $densities = array_fill(0, count($projections), 0.0);
    
        foreach ($projections as $i => $values) {
            foreach ($values as $j => $value) {
                [$edges, $counts] = $this->histograms[$j];

                foreach ($edges as $k => $edge) {
                    if ($value < $edge) {
                        $count = $counts[$k];

                        $densities[$i] += $count > 0
                            ? -log($count / $this->n)
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
