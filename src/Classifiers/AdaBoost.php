<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Ensemble;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Functions\Argmax;
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
 * [2] J. Zhu et al. (2006). Multi-class AdaBoost.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AdaBoost implements Estimator, Ensemble, Persistable
{
    /**
     * The base classifier to be boosted.
     *
     * @var \Rubix\ML\Estimator
     */
    protected $base;

    /**
     * The number of estimators to train in the ensemble.
     *
     * @var int
     */
    protected $estimators;

    /**
     * The learning rate i.e the step size.
     * 
     * @var float
     */
    protected $rate;

    /**
     * The ratio of samples to train each weak learner on.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The amount of error to tolerate before early stopping.
     *
     * @var float
     */
    protected $tolerance;
    
    /**
     * The class labels of the training set.
     *
     * @var array
     */
    protected $classes = [
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
     * The average error of a training sample at each epoch.
     *
     * @var array
     */
    protected $steps = [
        //
    ];

    /**
     * @param  \Rubix\ML\Estimator  $base
     * @param  int  $estimators
     * @param  float  $rate
     * @param  float  $ratio
     * @param  float  $tolerance
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(Estimator $base = null, int $estimators = 100, float $rate = 1.,
                                float $ratio = 0.8, float $tolerance = 1e-4)
    {
        if (is_null($base)) {
            $base = new ClassificationTree(1);
        }

        if ($base instanceof MetaEstimator) {
            throw new InvalidArgumentException('Base estimator cannot be a meta'
                . ' estimator.');
        }

        if ($base->type() !== self::CLASSIFIER) {
            throw new InvalidArgumentException('Base estimator must be a'
                . ' classifier.');
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Ensemble must contain at least'
                . ' 1 estimator.');
        }

        if ($rate < 0.) {
            throw new InvalidArgumentException('Learning rate must be greater'
                . ' than 0.');
        }

        if ($ratio < 0.01 or $ratio > 1.) {
            throw new InvalidArgumentException('Sample ratio must be between'
                . ' 0.01 and 1.0.');
        }

        if ($tolerance < 0. or $tolerance > 1.) {
            throw new InvalidArgumentException('Error tolerance must be between'
                . ' 0 and 1.');
        }

        $this->base = $base;
        $this->estimators = $estimators;
        $this->rate = $rate;
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
     * Return the average error at every epoch.
     *
     * @return array
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * Train a boosted enemble of *weak* classifiers assigning an influence value
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

        $this->classes = $dataset->possibleOutcomes();

        $labels = $dataset->labels();
        
        $n = $dataset->numRows();
        $p = (int) round($this->ratio * $n);
        $k = count($this->classes);

        $this->weights = array_fill(0, $n, 1 / $n);

        $this->ensemble = $this->influence = $this->steps = [];

        for ($epoch = 0; $epoch < $this->estimators; $epoch++) {
            $estimator = clone $this->base;

            $subset = $dataset->randomWeightedSubsetWithReplacement($p, $this->weights);

            $estimator->train($subset);

            $predictions = $estimator->predict($dataset);

            $loss = 0.;

            foreach ($predictions as $i => $prediction) {
                if ($prediction !== $labels[$i]) {
                    $loss += $this->weights[$i];
                }
            }

            $total = array_sum($this->weights);

            $loss /= $total;

            $influence = $this->rate
                * (log((1. - $loss) / ($loss ?: self::EPSILON))
                + log($k - 1));

            $this->ensemble[] = $estimator;
            $this->steps[] = $loss;
            $this->influence[] = $influence;

            if ($loss < $this->tolerance or $total < 0.) {
                break 1;
            }

            foreach ($predictions as $i => $prediction) {
                if ($prediction !== $labels[$i]) {
                    $this->weights[$i] *= exp($influence);
                }
            }

            $total = array_sum($this->weights);

            foreach ($this->weights as &$weight) {
                $weight /= $total;
            }
        }
    }

    /**
     * Make a prediction by consulting the ensemble of experts and choosing the class
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

        $scores = array_fill(0, $dataset->numRows(),
            array_fill_keys($this->classes, 0.));

        foreach ($this->ensemble as $i => $estimator) {
            $influence = $this->influence[$i];

            foreach ($estimator->predict($dataset) as $j => $prediction) {
                $scores[$j][$prediction] += $influence;
            }
        }

        $predictions = [];

        foreach ($scores as $dist) {
            $predictions[] = Argmax::compute($dist);
        }

        return $predictions;
    }
}
