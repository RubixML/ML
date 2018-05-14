<?php

namespace Rubix\Engine\Estimators;

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\Datasets\WeightedSupervised;
use Rubix\Engine\Estimators\Predictions\Prediction;
use InvalidArgumentException;
use RuntimeException;
use ReflectionClass;

class AdaBoost implements BinaryClassifier, Persistable
{
    /**
     * The reflector instance of the base classifier.
     *
     * @param \ReflectionClass
     */
    protected $reflector;

    /**
     * The constructor arguments of the base classifier.
     *
     * @var array
     */
    protected $params = [
        //
    ];

    /**
     * The number of experts to train. Note that the algorithm will terminate early
     * if it can train a classifier that exceeds the threshold hyperparameter.
     *
     * @var int
     */
    protected $experts;

    /**
     * The ratio of samples to train each classifier on.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The threshold accuracy of a single classifier before the algorithm stops early.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The actual labels of the binary class outcomes.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * The ensemble of experts that specialize in classifying certain aspects of
     * the training set.
     *
     * @var array
     */
    protected $ensemble = [
        //
    ];

    /**
     * The amount of influence an expert has. i.e. the classifier's ability to
     * make accurate predictions.
     *
     * @var array
     */
    protected $influence = [
        //
    ];

    /**
     * @param  string  $base
     * @param  array  $params
     * @param  int  $experts
     * @param  float  $ratio
     * @param  float  $threshold
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $base, array $params = [], int $experts = 100, float $ratio = 0.1, float $threshold = 0.999)
    {
        $this->reflector = new ReflectionClass($base);

        if (!in_array(Classifier::class, $this->reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('Base class must be a classifier.');
        }

        if ($experts < 1) {
            throw new InvalidArgumentException('The number of experts cannot be less than 1.');
        }

        if ($ratio < 0.01 || $ratio > 1) {
            throw new InvalidArgumentException('Sample ratio must be a float value between 0.01 and 1.0.');
        }

        if ($threshold < 0 || $threshold > 1) {
            throw new InvalidArgumentException('Threshold value must be a float between 0 and 1.');
        }

        $this->params = $params;
        $this->experts = $experts;
        $this->ratio = $ratio;
        $this->threshold = $threshold;
    }

    /**
     * Return the array of trained classifiers that comprise the ensemble.
     *
     * @return array
     */
    public function ensemble() : array
    {
        return $this->ensemble;
    }

    /**
     * Return the list of influence values for each estimator in the ensemble.
     *
     * @return array
     */
    public function influence() : array
    {
        return $this->influence;
    }

    /**
     * Train a boosted enemble of binary classifiers assigning an influence value
     * to each one and re-weighting the training data according to reflect how
     * difficult a particular sample is to classify.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        $labels = $dataset->labels();

        if (count($labels) !== 2) {
            throw new InvalidArgumentException('The number of unique outcomes must be exactly 2, ' . (string) count($labels) . ' found.');
        }

        if (!$dataset instanceof WeightedSupervised) {
            $dataset = new WeightedSupervised($dataset->samples(), $dataset->outcomes());
        }

        $this->labels = [1 => $labels[0], -1 => $labels[1]];
        $this->ensemble = $this->influence = [];

        for ($round = 1; $round <= $this->experts; $round++) {
            $estimator = $this->reflector->newInstanceArgs($this->params);
            $predictions = [];
            $error = 0;

            $estimator->train($dataset->generateRandomSubsetWithReplacement($this->ratio));

            foreach ($dataset as $sample) {
                $predictions[] = $estimator->predict($sample)->outcome();
            }

            foreach ($dataset->outcomes() as $row => $outcome) {
                if ($predictions[$row] !== $outcome) {
                    $error += $dataset->weight($row);
                }
            }

            $total = $dataset->totalWeight();
            $error /= $total;
            $influence = 0.5 * log((1 - $error) / ($error ? $error : self::EPSILON));

            foreach ($dataset->outcomes() as $row => $outcome) {
                $x = $predictions[$row] === $this->labels[1] ? 1 : -1;
                $y = $outcome === $this->labels[1] ? 1 : -1;

                $dataset->setWeight($row,  $dataset->weight($row) * exp(-$influence * $x * $y) / $total);
            }

            $this->influence[] = $influence;
            $this->ensemble[] = $estimator;

            if ((1 - $error) > $this->threshold) {
                break 1;
            }
        }
    }

    /**
     * Make a prediction by consulting the ensemble of experts and chosing the class
     * label closest to the value of the weighted sum of each expert's prediction.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Estimaotors\Predictions\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $total = 0.0;

        foreach ($this->ensemble as $i => $estimator) {
            $prediction = $estimator->predict($sample);

            $output = $prediction->outcome() === $this->labels[1] ? 1 : -1;

            if ($prediction instanceof Probabalistic) {
                $output *= $prediction->probability();
            }

            $total += $this->influence[$i] * $output;
        }

        return new Prediction($this->labels[$total > 0 ? 1 : -1]);
    }
}
