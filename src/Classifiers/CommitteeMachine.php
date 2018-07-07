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
     * The committee of experts. i.e. the ensemble of
     * probabilistic classifiers.
     *
     * @var array
     */
    protected $experts = [
        //
    ];

    /**
     * The user-specified influence that each classifier has in the committee.
     *
     * @var array
     */
    protected $influence = [
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

        $total = 0.0;

        foreach ($experts as &$expert) {
            if (!is_array($expert)) {
                $expert = [1, $expert];
            }

            if (count($expert) !== 2) {
                throw new InvalidArgumentException('Exactly 2 arguments are'
                    . ' required for expert configuration.');
            }

            if (!is_int($expert[0]) and !is_float($expert[0])) {
                throw new InvalidArgumentException('Influence parameter must be'
                    . ' an integer or floating point number.');
            }

            if ($expert[0] < 0) {
                throw new InvalidArgumentException('Influence cannot be less'
                    . ' than 0.');
            }

            if (!$expert[1] instanceof Classifier) {
                throw new InvalidArgumentException('Estimator must be a'
                    . ' classifier, ' . gettype($expert[0]) . ' found.');
            }

            $total += $expert[0];
        }

        $this->experts = $this->influence = [];

        foreach ($experts as &$expert) {
            $this->influence[] = $expert[0] / $total;
            $this->experts[] = $expert[1];
        }
    }

    /**
     * Return the underlying classifier instances.
     *
     * @return array
     */
    public function experts() : array
    {
        return $this->experts;
    }

    /**
     * Return the iinfluence of each classifier.
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
        $probabilities = array_fill(0, $dataset->numRows(),
            array_fill_keys($this->classes, 0.0));

        foreach ($this->experts as $i => $expert) {
            foreach ($expert->predict($dataset) as $j => $prediction) {
                $probabilities[$j][$prediction] += $this->influence[$i];
            }
        }

        return $probabilities;
    }
}
