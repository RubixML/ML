<?php

namespace Rubix\ML\AnomalyDetectors;

use Tensor\Matrix;
use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\linspace;
use function count;
use function is_null;
use function array_slice;
use function array_fill;
use function round;
use function min;
use function max;
use function log;
use function sqrt;

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
class Loda implements Estimator, Learner, Online, Scoring, Persistable
{
    use AutotrackRevisions;

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
     * The proportion of outliers that are assumed to be present in the training set.
     *
     * @var float
     */
    protected float $contamination;

    /**
     * The number of projection/histogram pairs in the ensemble.
     *
     * @var positive-int
     */
    protected int $estimators;

    /**
     * The number of bins in each equi-width histogram.
     *
     * @var int|null
     */
    protected ?int $bins = null;

    /**
     * Should we calculate the equi-width bin count on the fly?
     *
     * @var bool
     */
    protected bool $fitBins;

    /**
     * The sparse random projection matrix.
     *
     * @var \Tensor\Matrix|null
     */
    protected ?\Tensor\Matrix $r = null;

    /**
     * The edges and bin counts of each histogram.
     *
     * @var array{list<float>,list<int<0,max>>}|mixed[]
     */
    protected array $histograms = [
        //
    ];

    /**
     * The minimum negative log likelihood score necessary to flag an anomaly.
     *
     * @var float|null
     */
    protected ?float $threshold;

    /**
     * The number of samples that have been learned so far.
     *
     * @var int
     */
    protected int $n = 0;

    /**
     * @param float $contamination
     * @param int $estimators
     * @param int|null $bins
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $contamination = 0.1, int $estimators = 100, ?int $bins = null)
    {
        if ($contamination < 0.0 or $contamination > 0.5) {
            throw new InvalidArgumentException('Contamination must be'
                . " between 0 and 0.5, $contamination given.");
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Number of estimators'
                . " must be greater than 0, $estimators given.");
        }

        if (isset($bins) and $bins < self::MIN_BINS) {
            throw new InvalidArgumentException('Bins must be greater'
                . ' than ' . self::MIN_BINS . ", $bins given.");
        }

        $this->contamination = $contamination;
        $this->estimators = $estimators;
        $this->bins = $bins;
        $this->fitBins = is_null($bins);
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
            'contamination' => $this->contamination,
            'estimators' => $this->estimators,
            'bins' => $this->bins,
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
            $this->bins = max(self::MIN_BINS, (int) round(log($m, 2.0)) + 1);
        }

        $this->r = Matrix::gaussian($n, $this->estimators);

        if ($n >= self::MIN_SPARSE_DIMENSIONS) {
            $mask = Matrix::rand($n, $this->estimators)
                ->less(sqrt($n) / $n);

            $this->r = $this->r->multiply($mask);
        }

        $projections = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose()
            ->asArray();

        foreach ($projections as $values) {
            $min = (float) min($values);
            $max = (float) max($values);

            $edges = linspace($min, $max, $this->bins + 1);

            $edges = array_slice($edges, 1, -1);

            $edges[] = INF;

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
            ->transpose()
            ->asArray();

        foreach ($projections as $i => $values) {
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

        $n = $dataset->numSamples();

        $this->n += $n;

        $densities = $this->densities($projections);

        $threshold = Stats::quantile($densities, 1.0 - $this->contamination);

        $beta = $n / $this->n;

        $this->threshold = (1.0 - $beta) * $this->threshold + $beta * $threshold;
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
            ->transpose()
            ->asArray();

        return $this->densities($projections);
    }

    /**
     * Estimate the probability density function of each 1-dimensional projection using the histograms
     * created during training.
     *
     * @param list<list<float>> $projections
     * @return list<float>
     */
    protected function densities(array $projections) : array
    {
        $n = count(current($projections) ?: []);

        $densities = array_fill(0, $n, 0.0);

        foreach ($projections as $i => $values) {
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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Loda (' . Params::stringify($this->params()) . ')';
    }
}
