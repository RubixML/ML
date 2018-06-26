<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

class CommitteeMachine implements Multiclass, Probabilistic, Persistable
{
    /**
     * The committee of experts. i.e. the ensemble of probabilistic classifiers.
     *
     * @var array
     */
    protected $experts = [
        //
    ];

    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * @param  array  $experts
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $experts)
    {
        if (count($experts) === 0) {
            throw new InvalidArgumentException('Must have at least 1 expert in'
                . ' the committee.');
        }

        foreach ($experts as $estimator) {
            $this->addExpert($estimator);
        }
    }

    /**
     * Return the underlying estimator instances.
     *
     * @return array
     */
    public function experts() : array
    {
        return $this->experts;
    }

    /**
     * Train all the experts with the dataset.
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

        foreach ($this->experts as $estimator) {
            $estimator->train(clone $dataset);
        }
    }

    /**
     * Make a prediction based on the class that recieved the highest
     * probability score from the committee of experts.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($this->proba($samples) as $distribution) {
            $best = ['probability' => -INF, 'outcome' => null];

            foreach ($distribution as $class => $probability) {
                if ($probability > $best['probability']) {
                    $best['probability'] = $probability;
                    $best['outcome'] = $class;
                }
            }

            $predictions[] = $best['outcome'];
        }

        return $predictions;
    }

    /**
     * Combine the probablistic predictions of the committee.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $n = count($this->experts) + self::EPSILON;

        $probabilities = array_fill(0, $dataset->numRows(),
            array_fill_keys($this->classes, 0.0));

        foreach ($this->experts as $expert) {
            foreach ($expert->proba($dataset) as $i => $distribution) {
                foreach ($distribution as $class => $probability) {
                    $probabilities[$i][$class] += $probability / $n;
                }
            }
        }

        return $probabilities;
    }

    /**
     * Add an expert to the committee.
     *
     * @param  \Rubix\ML\Classifiers\Classifier  $estimator
     * @throws \InvalidArgumentException
     * @return void
     */
    protected function addExpert(Classifier $estimator) : void
    {
        if (!$estimator instanceof Probabilistic) {
            throw new InvalidArgumentException('Estimator must be a'
                . ' probabilistic classifier.');
        }

        $this->experts[] = $estimator;
    }
}
