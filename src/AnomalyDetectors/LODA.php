<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\Tensor\Matrix;
use Rubix\Tensor\Vector;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\EPSILON;
use const Rubix\ML\LOG_EPSILON;

/**
 * LODA
 *
 * Lightweight Online Detector of Anomalies uses sparse random projection vectors
 * to generate an ensemble of unique one dimensional equi-width histograms able
 * to estimate the probability density of an unknown sample. The anomaly score is
 * given by the negative log likelihood whose upper threshold can be set by the
 * user through the *contamination* hyper-parameter.
 *
 * References:
 * [1] T. Pevný. (2015). Loda: Lightweight on-line detector of anamalies.
 * [2] L. Birg´e et al. (2005). How Many Bins Should Be Put In A Regular Histogram.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LODA implements Estimator, Learner, Online, Ranking, Persistable
{
    protected const MIN_SPARSE_DIMENSIONS = 3;

    /**
     * The number of bins in each equi-width histogram.
     *
     * @var int
     */
    protected $bins;

    /**
     * The number of projection/histogram pairs in the ensemble.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The threshold anomaly score to be flagged as an outlier.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The random projection matrix.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $r;

    /**
     * The edges, counts, and densities of the histograms.
     *
     * @var array[]|null
     */
    protected $histograms;

    /**
     * The number of samples that have been learned so far.
     *
     * @var int|null
     */
    protected $n;

    /**
     * Estimate the number of bins from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return int
     */
    public static function estimateBins(Dataset $dataset) : int
    {
        return (int) round(log($dataset->numRows(), 2)) + 1;
    }

    /**
     * @param int $bins
     * @param int $estimators
     * @param float $threshold
     * @throws \InvalidArgumentException
     */
    public function __construct(int $bins = 5, int $estimators = 100, float $threshold = 5.5)
    {
        if ($bins < 1) {
            throw new InvalidArgumentException('The number of bins cannot'
                . " be less than 1, $bins given.");
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('At least 1 histogram is'
                . " requied to make a prediction, $estimators given.");
        }

        if ($threshold < 0.) {
            throw new InvalidArgumentException('Threshold must be'
                . " greater than 0, $threshold given.");
        }

        $this->bins = $bins;
        $this->estimators = $estimators;
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

        $z = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        foreach ($z as $values) {
            $min = min($values) - EPSILON;
            $max = max($values) + EPSILON;

            $edges = Vector::linspace($min, $max, $this->bins + 1)->asArray();

            $edges[] = INF;

            $counts = array_fill(0, $this->bins + 2, 0);

            $interior = array_slice($edges, 1, $this->bins);

            foreach ($values as $value) {
                foreach ($interior as $k => $edge) {
                    if ($value < $edge) {
                        $counts[$k]++;

                        continue 2;
                    }
                }

                $counts[$this->bins]++;
            }

            $densities = [];

            foreach ($counts as $count) {
                $densities[] = $count > 0
                    ? -log($count / $m)
                    : -LOG_EPSILON;
            }

            $this->histograms[] = [$edges, $counts, $densities];
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
        if (!$this->r or !$this->histograms or !$this->n) {
            $this->train($dataset);

            return;
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->n += $dataset->numRows();

        $z = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        foreach ($z as $i => $values) {
            [$edges, $counts, $densities] = $this->histograms[$i];

            $interior = array_slice($edges, 1, $this->bins);

            foreach ($values as $value) {
                foreach ($interior as $k => $edge) {
                    if ($value < $edge) {
                        $counts[$k]++;

                        continue 2;
                    }
                }

                $counts[$this->bins]++;
            }

            foreach ($counts as $j => $count) {
                $densities[$j] = $count > 0
                    ? -log($count / $this->n)
                    : -LOG_EPSILON;
            }

            $this->histograms[$i] = [$edges, $counts, $densities];
        }
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
            throw new RuntimeException('The learner has not'
                . ' been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $z = Matrix::quick($dataset->samples())
            ->matmul($this->r)
            ->transpose();

        return $this->logLikelihood($z);
    }

    /**
     * Return the negative log likelihoods of each projection.
     *
     * @param \Rubix\Tensor\Matrix $z
     * @return array
     */
    protected function logLikelihood(Matrix $z) : array
    {
        $likelihoods = array_fill(0, $z->n(), 0.);
        
        foreach ($z as $i => $values) {
            [$edges, $counts, $densities] = $this->histograms[$i];

            foreach ($values as $j => $value) {
                foreach ($edges as $k => $edge) {
                    if ($value < $edge) {
                        $likelihoods[$j] += $densities[$k];

                        break 1;
                    }
                }
            }
        }

        foreach ($likelihoods as &$likelihood) {
            $likelihood /= $this->estimators;
        }

        return $likelihoods;
    }

    /**
     * The decision function.
     *
     * @param float $likelihood
     * @return int
     */
    protected function decide(float $likelihood) : int
    {
        return $likelihood > $this->threshold ? 1 : 0;
    }
}
