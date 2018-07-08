<?php

namespace Rubix\ML;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Regressors\Regressor;
use Rubix\ML\Clusterers\Clusterer;
use Rubix\ML\Classifiers\Classifier;
use Rubix\ML\AnomalyDetectors\Detector;
use InvalidArgumentException;

class CommitteeMachine implements MetaEstimator, Ensemble, Persistable
{
    /**
     * The user-specified influence that each classifier has in the committee.
     *
     * @var array
     */
    protected $influence = [
        //
    ];

    /**
     * The committee of experts. i.e. the ensemble of estimators.
     *
     * @var array
     */
    protected $ensemble = [
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
            if ($experts[0][1] instanceof Classifier) {
                if (!$expert[1] instanceof Classifier) {
                    throw new InvalidArgumentException('Cannot mix estimator'
                        . ' types. ' . gettype($expert) . ' found,'
                        . ' classifier expected.');
                }
            } else if (!$experts[0][1] instanceof Regressor) {
                if (!$expert[1] instanceof Regressor) {
                    throw new InvalidArgumentException('Cannot mix estimator'
                        . ' types. ' . gettype($expert) . ' found,'
                        . ' regressor expected.');
                }
            } else if (!$experts[0][1] instanceof Detector) {
                if (!$expert[1] instanceof Detector) {
                    throw new InvalidArgumentException('Cannot mix estimator'
                        . ' types. ' . gettype($expert) . ' found,'
                        . ' anomaly detector expected.');
                }
            }
        }

        $total = 0.0;

        foreach ($experts as &$expert) {
            if (!is_array($expert)) {
                $expert = [1, $expert];
            }

            if (count($expert) !== 2) {
                throw new InvalidArgumentException('Exactly 2 arguments are'
                    . ' required for estimator configuration.');
            }

            if (!is_int($expert[0]) and !is_float($expert[0])) {
                throw new InvalidArgumentException('Influence parameter must be'
                    . ' an integer or floating point number.');
            }

            if ($expert[0] < 0) {
                throw new InvalidArgumentException('Influence cannot be less'
                    . ' than 0.');
            }

            if ($expert[1] instanceof Clusterer) {
                throw new InvalidArgumentException('This meta estimator does'
                    . ' not work with clusterers.');
            }

            $total += $expert[0];
        }

        unset($expert);

        foreach ($experts as $expert) {
            $this->influence[] = $expert[0] / $total;
            $this->ensemble[] = $expert[1];
        }
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
     * Return the influence of each estimator.
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
        foreach ($this->ensemble as $expert) {
            $expert->train(clone $dataset);
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
        $votes = [[]];

        foreach ($this->ensemble as $expert) {
            foreach ($expert->predict($dataset) as $i => $vote) {
                $votes[$i][] = $vote;
            }
        }

        return $this->tally($votes);
    }

    /**
     * Tally each estimators prediction and weight it according to its
     * associated normalized influence value.
     *
     * @param  array  $votes
     * @return array
     */
    protected function tally(array $votes) : array
    {
        if ($this->ensemble[0] instanceof Classifier) {
            return array_map(function ($predictions) {
                $counts = array_fill_keys(array_unique($predictions), 0);

                foreach ($predictions as $i => $prediction) {
                    $counts[$prediction] += $this->influence[$i];
                }

                return array_search(max($counts), $counts);
            }, $votes);
        } else if ($this->ensemble[0] instanceof Detector) {
            return array_map(function ($predictions) {
                $score = 0.0;

                foreach ($predictions as $i => $prediction) {
                    $score += $this->influence[$i] * $prediction;
                }

                return $score >= 0.5 ? 1 : 0;
            }, $votes);
        } else {
            return array_map(function ($predictions) {
                $value = 0.0;

                foreach ($predictions as $i => $prediction) {
                    $value += $this->influence[$i] * $prediction;
                }

                return $value;
            }, $votes);
        }
    }
}
