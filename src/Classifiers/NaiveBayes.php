<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use InvalidArgumentException;

class NaiveBayes implements Multiclass, Online, Probabilistic, Persistable
{
    /**
     * The amount of additive (Laplace) smoothing to apply to the probabilities.
     *
     * @var float
     */
    protected $smoothing;

    /**
     * The weight of each class as a proportion of the entire training set.
     *
     * @var array
     */
    protected $weights = [
        //
    ];

    /**
     * The precomputed prior probabilities of each label given by their weight.
     *
     * @var array
     */
    protected $priors = [
        //
    ];

    /**
     * The count of each feature from the training set.
     *
     * @var array
     */
    protected $counts = [
        //
    ];

    /**
     * The precomputed probabilities of each feature given each class label.
     *
     * @var array
     */
    protected $probs = [
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
     * @param  float  $smoothing
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $smoothing = 1.0)
    {
        if ($smoothing < 0.0) {
            throw new InvalidArgumentException('Smoothing parameter cannot be'
                . ' less than 0.');
        }

        $this->smoothing = $smoothing;
    }

    /**
     * Return the class prior probabilities based on their weight over all
     * training samples.
     *
     * @return array
     */
    public function priors() : array
    {
        return $this->priors;
    }

    /**
     * Return the probabilities of each feature given each class label.
     *
     * @return array
     */
    public function probabilities() : array
    {
        return $this->probs;
    }

    /**
     * Compute the probabilities of each feature in the training set.
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

        $this->weights = array_fill_keys($this->classes, 0);

        $this->priors = array_fill_keys($this->classes, 0.0);

        $this->counts = $this->probs = array_fill_keys($this->classes,
            array_fill(0, $dataset->numColumns(), []));

        $this->partial($dataset);
    }

    /**
     * Compute the rolling counts and probabilities.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        if (in_array(self::CONTINUOUS, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' categorical features.');
        }

        if (empty($this->weights)) {
            $this->train($dataset);
        }

        foreach ($dataset->stratify() as $class => $samples) {
            foreach (array_map(null, ...$samples) as $column => $values) {
                $counts = array_count_values((array) $values);

                foreach ($counts as $category => $count) {
                    if (!isset($this->counts[$class][$column][$category])) {
                        $this->counts[$class][$column][$category] = $count;
                    } else {
                        $this->counts[$class][$column][$category] += $count;
                    }
                }

                $total = (2.0 * $this->smoothing) + array_sum($counts);

                foreach ($this->counts[$class][$column] as $category => $count) {
                    $probability = ($count + $this->smoothing) / $total;

                    $this->probs[$class][$column][$category] = $probability;
                }
            }

            $this->weights[$class] += count($samples);
        }

        $total = (count($this->weights) * $this->smoothing)
            + array_sum($this->weights);

        foreach ($this->weights as $class => $weight) {
            $this->priors[$class] = ($weight + $this->smoothing) / $total;
        }
    }

    /**
    * Calculate the probabilities of the sample being a member of a class and
    * chose the class with the highest probability as the prediction.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($this->proba($dataset) as $probabilities) {
            $best = ['probability' => -INF, 'outcome' => null];

            foreach ($probabilities as $class => $probability) {
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
     * Calculate the probabilities of the sample being a member of each class.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $probabilities = [];

        foreach ($dataset as $i => $sample) {
            $scores = [];

            foreach ($this->classes as $class) {
                $score = $this->priors[$class];

                foreach ($sample as $column => $feature) {
                    $score *= $this->probs[$class][$column][$feature] ?? 0.0;
                }

                $scores[$class] = $score;
            }

            $total = array_sum($scores);

            foreach ($scores as $class => $score) {
                $probabilities[$i][$class] = $score / $total;
            }
        }

        return $probabilities;
    }
}
