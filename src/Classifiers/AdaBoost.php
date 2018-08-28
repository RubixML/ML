<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Ensemble;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;
use RuntimeException;

/**
 * AdaBoost
 *
 * Short for Adaptive Boosting, this ensemble classifier can improve the
 * performance of an otherwise weak classifier by focusing more attention on
 * samples that are harder to classify.
 *
 * References:
 * [1] Y. Freund et al. (1996). A Decision-theoretic Generalization of On-line
 * Learning and an Application to Boosting.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AdaBoost implements Estimator, Ensemble, Persistable
{
    /**
     * The class name of the base classifier.
     *
     * @var string
     */
    protected $base;

    /**
     * The constructor arguments of the base classifier.
     *
     * @var array
     */
    protected $params = [
        //
    ];

    /**
     * The number of estimators to train. Note that the algorithm will
     * terminate early if it can train a classifier that exceeds the threshold
     * hyperparameter.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The ratio of samples to train each classifier on.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The amount of accuracy to tolerate before early stopping.
     *
     * @var float
     */
    protected $tolerance;

    /**
     * The unique binary class labels of the trainign set.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * The memoized inverse of the classes associative array for optimization.
     *
     * @var array
     */
    protected $beta = [
        //
    ];

    /**
     * The ensemble of "weak" classifiers.
     *
     * @var array
     */
    protected $ensemble = [
        //
    ];

    /**
     * The weight of each training sample in the dataset.
     *
     * @var array
     */
    protected $weights = [
        //
    ];

    /**
     * The amount of influence a particular classifier has. i.e. the
     * classifier's ability to make accurate predictions.
     *
     * @var array
     */
    protected $influence = [
        //
    ];

    /**
     * @param  string  $base
     * @param  array  $params
     * @param  int  $estimators
     * @param  float  $ratio
     * @param  float  $tolerance
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $base, array $params = [], int $estimators = 100,
                                float $ratio = 0.1, float $tolerance = 1e-3)
    {
        $proxy = new $base(...$params);

        if ($proxy instanceof MetaEstimator) {
            throw new InvalidArgumentException('Base class cannot be a meta'
                . ' estimator.');
        }

        if ($proxy->type() !== self::CLASSIFIER) {
            throw new InvalidArgumentException('Base estimator must be a'
                . ' classifier.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Ensemble must contain at least'
                . ' 1 classifier.');
        }

        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException('Sample ratio must be between'
                . ' 0.01 and 1.0.');
        }

        if ($tolerance < 0. or $tolerance > 1.) {
            throw new InvalidArgumentException('Tolerance must be between'
                . ' 0 and 1.');
        }

        $this->base = $base;
        $this->params = $params;
        $this->estimators = $estimators;
        $this->ratio = $ratio;
        $this->tolerance = $tolerance;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
    }

    /**
     * Return the ensemble of estimators.
     *
     * @return array
     */
    public function estimators() : array
    {
        return $this->ensemble;
    }

    /**
     * Return the weights associated with each training sample.
     *
     * @return array
     */
    public function weights() : array
    {
        return $this->weights;
    }

    /**
     * Return the list of influence values for each classifier in the ensemble.
     *
     * @return array
     */
    public function influence() : array
    {
        return $this->influence;
    }

    /**
     * Train a boosted enemble of binary classifiers assigning an influence value
     * to each one and re-weighting the training data accordingly to reflect how
     * difficult a particular sample is to classify.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        $n = $dataset->numRows();

        $classes = $dataset->possibleOutcomes();

        if (count($classes) !== 2) {
            throw new InvalidArgumentException('The number of unique outcomes'
                . ' must be exactly 2, ' . (string) count($classes) . ' found.');
        }

        $k = (int) round($this->ratio * $n);

        $this->classes = [1 => $classes[0], -1 => $classes[1]];
        $this->beta = array_flip($this->classes);

        $this->weights = array_fill(0, count($dataset), 1 / count($dataset));

        $this->ensemble = $this->influence = [];

        for ($epoch = 0; $epoch < $this->estimators; $epoch++) {
            $estimator = new $this->base(...$this->params);

            $subset = $dataset->randomWeightedSubsetWithReplacement($k, $this->weights);

            $estimator->train($subset);

            $predictions = $estimator->predict($dataset);

            $error = 0.;

            foreach ($predictions as $i => $prediction) {
                if ($prediction !== $dataset->label($i)) {
                    $error += $this->weights[$i];
                }
            }

            $total = array_sum($this->weights);
            $error = ($error / $total) + self::EPSILON;
            $influence = 0.5 * log((1. - $error) / $error);

            foreach ($predictions as $i => $prediction) {
                $x = $this->beta[$prediction];
                $y = $this->beta[$dataset->label($i)];

                $this->weights[$i] *= M_E ** (-$influence * $x * $y) / $total;
            }

            $this->influence[] = $influence;
            $this->ensemble[] = $estimator;

            if ($error < $this->tolerance) {
                break 1;
            }
        }
    }

    /**
     * Make a prediction by consulting the ensemble of experts and chosing the class
     * label closest to the value of the weighted sum of each expert's prediction.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->ensemble)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $scores = array_fill(0, $dataset->numRows(), 0.);

        foreach ($this->ensemble as $i => $estimator) {
            foreach ($estimator->predict($dataset) as $j => $prediction) {
                $output = $prediction === $this->classes[1] ? 1 : -1;

                $scores[$j] += $output * $this->influence[$i];
            }
        }

        $predictions = [];

        foreach ($scores as $score) {
            $predictions[] = $this->classes[$score > 0 ? 1 : -1];
        }

        return $predictions;
    }
}
