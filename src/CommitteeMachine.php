<?php

namespace Rubix\ML;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use Rubix\ML\Regressors\Regressor;
use Rubix\ML\Clusterers\Clusterer;
use Rubix\ML\Classifiers\Classifier;
use Rubix\ML\AnomalyDetectors\Detector;
use InvalidArgumentException;

/**
 * Committee Machine
 *
 * A voting Ensemble that aggregates the predictions of a committee of
 * user-specified, heterogeneous estimators (called experts) of a single type
 * (i.e all Classifiers, Regressors, etc). The committee uses a hard-voting
 * scheme to make final predictions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CommitteeMachine implements MetaEstimator, Ensemble, Persistable
{
    /**
     * The committee of experts. i.e. the ensemble of estimators.
     *
     * @var array
     */
    protected $experts = [
        //
    ];

    /**
     * @param  array  $experts
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $experts)
    {
        if (count($experts) < 1) {
            throw new InvalidArgumentException('Ensemble must contain at least'
                . ' 1 estimator.');
        }

        foreach ($experts as $expert) {
            if (!$expert instanceof Estimator) {
                throw new InvalidArgumentException('Base class must implement'
                    . ' the estimator interface.');
            }

            if ($expert instanceof Clusterer) {
                throw new InvalidArgumentException('This meta estimator does'
                    . ' not work on clusterers.');
            }

            if ($expert instanceof MetaEstimator) {
                throw new InvalidArgumentException('Base estimator cannot be a'
                    . ' meta estimator.');
            }

            if ($experts[0] instanceof Classifier) {
                if (!$expert instanceof Classifier) {
                    throw new InvalidArgumentException('Cannot mix estimator'
                        . ' types, ' . gettype($expert) . ' found,'
                        . ' Classifier expected.');
                }
            } else if ($experts[0] instanceof Regressor) {
                if (!$expert instanceof Regressor) {
                    throw new InvalidArgumentException('Cannot mix estimator'
                        . ' types, ' . gettype($expert) . ' found,'
                        . ' Regressor expected.');
                }
            } else if ($experts[0] instanceof Detector) {
                if (!$expert instanceof Detector) {
                    throw new InvalidArgumentException('Cannot mix estimator'
                        . ' types, ' . gettype($expert) . ' found,'
                        . ' anomaly Detector expected.');
                }
            }
        }

        $this->experts = $experts;
    }

    /**
     * Return the ensemble of estimators.
     *
     * @return array
     */
    public function estimators() : array
    {
        return $this->experts;
    }

    /**
     * Train all the experts with the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        foreach ($this->experts as $expert) {
            $expert->train($dataset);
        }
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [[]];

        foreach ($this->experts as $expert) {
            foreach ($expert->predict($dataset) as $i => $prediction) {
                $predictions[$i][] = $prediction;
            }
        }

        return $this->tally($predictions);
    }

    /**
     * Tally each estimators prediction and weight it according to its
     * associated normalized influence value.
     *
     * @param  array  $predictions
     * @return array
     */
    protected function tally(array $predictions) : array
    {
        if ($this->experts[0] instanceof Classifier) {
            return array_map(function ($votes) {
                $counts = array_count_values($votes);

                return array_search(max($counts), $counts);
            }, $predictions);
        } else if ($this->experts[0] instanceof Detector) {
            return array_map(function ($votes) {
                return Average::mean($votes) >= 0.5 ? 1 : 0;
            }, $predictions);
        } else {
            return array_map(function ($votes) {
                return Average::mean($votes);
            }, $predictions);
        }
    }
}
