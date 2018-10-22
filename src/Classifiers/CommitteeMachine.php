<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Ensemble;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\MetaEstimator;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Functions\Argmax;
use InvalidArgumentException;
use RuntimeException;

/**
 * Committee Machine
 *
 * A voting ensemble that aggregates the predictions of a committee of heterogeneous
 * probabilistic classifiers (called experts). The committee uses a user-specified
 * influence-based scheme to make final predictions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CommitteeMachine implements Estimator, Learner, Ensemble, Probabilistic, Persistable
{
    /**
     * The committee of experts. i.e. the ensemble of estimators.
     *
     * @var array
     */
    protected $experts;

    /**
     * The influence values of each expert in the committee.
     *
     * @var array
     */
    protected $influence;

    /**
     * The unique class labels.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * @param  array  $experts
     * @param  array|null  $influence
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $experts, ?array $influence = null)
    {
        $n = count($experts);

        if ($n < 1) {
            throw new InvalidArgumentException('Ensemble must contain at least'
                . ' 1 estimator, 0 given.');
        }

        foreach ($experts as $expert) {
            if ($expert instanceof MetaEstimator) {
                throw new InvalidArgumentException('Base estimator cannot be a'
                    . ' meta estimator, ' . gettype($expert) . ' given.');
            }

            if ($expert->type() !== self::CLASSIFIER) {
                throw new InvalidArgumentException('Base estimator must be a'
                    . ' classifier, ' . gettype($expert) . ' given.');
            }

            if (!$expert instanceof Probabilistic) {
                throw new InvalidArgumentException('Base estimator must'
                    . ' implement the probabilistic interface.');
            }
        }

        if (is_array($influence)) {
            if (count($influence) !== $n) {
                throw new InvalidArgumentException("The number of influence"
                    . " values must equal the number of experts, $n needed"
                    . " but " . count($influence) . "given.");
            }

            $total = array_sum($influence);

            if ($total == 0) {
                throw new InvalidArgumentException('Total influence for the'
                    . ' committee cannot be 0.');
            }

            $influence = array_map(function ($value) use ($total) {
                return $value / $total;
            }, $influence);
        } else {
            $influence = array_fill(0, $n, 1 / $n);
        }

        $this->experts = $experts;
        $this->influence = $influence;
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
     * Return the ensemble of eclassifiers.
     *
     * @return array
     */
    public function estimators() : array
    {
        return $this->experts;
    }

    /**
     * Return the normalized influence values for each expert in the committee.
     *
     * @return array
     */
    public function influence() : array
    {
        return $this->influence;
    }

    /**
     * Train all the experts with the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        $this->classes = $dataset->possibleOutcomes();

        foreach ($this->experts as $expert) {
            $expert->train($dataset);
        }
    }

    /**
     * Make a prediction based on the class probabilities.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($this->proba($dataset) as $probabilities) {
            $predictions[] = Argmax::compute($probabilities);
        }

        return $predictions;
    }

    /**
     * Output a vector of probabilities estimates.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (empty($this->experts)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $probabilities = array_fill(0, $dataset->numRows(),
            array_fill_keys($this->classes, 0.));

        foreach ($this->experts as $i => $expert) {
            $influence = $this->influence[$i];

            foreach ($expert->proba($dataset) as $j => $joint) {
                foreach ($joint as $class => $probability) {
                    $probabilities[$j][$class] += $influence
                        * $probability;
                }
            }
        }

        return $probabilities;
    }
}
