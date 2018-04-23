<?php

namespace Rubix\Engine;

use InvalidArgumentException;
use RuntimeException;
use ReflectionClass;

class AdaBoost implements Classifier
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
    protected $args = [
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
     * The ratio of samples to consider during each training round.
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
     * The ensemble of experts that each specialize in classifying certain aspects of
     * the training set.
     *
     * @var array
     */
    protected $ensemble = [
        //
    ];

    /**
     * The amount of influence an expert has. i.e. the classifier's ability to
     * make accurate predictions historically.
     *
     * @var array
     */
    protected $influence = [
        //
    ];

    /**
     * @param  string  $base
     * @param  array  $args
     * @param  int  $experts
     * @param  float  $ratio
     * @param  float  $threshold
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $base, array $args = [], int $experts = 50, float $ratio = 0.1, float $threshold = 0.999)
    {
        $this->reflector = new ReflectionClass($base);

        if (!in_array(Classifier::class, $this->reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('Base estimator must be a classifier.');
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

        $this->args = $args;
        $this->experts = $experts;
        $this->ratio = $ratio;
        $this->threshold = $threshold;
    }

    /**
     * Train a boosted enemble of binary classifiers assigning an influence value
     * to each one and re-weighting the training data according to reflect how
     * difficult a particular sample is to classify.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $data) : void
    {
        if (!$data instanceof SupervisedDataset) {
            throw new InvalidArgumentException('This estimator requires a supervised dataset.');
        }

        $labels = $data->labels();

        if (count($labels) !== 2) {
            throw new InvalidArgumentException('The number of unique outcomes must be exactly 2, ' . (string) count($labels) . ' given.');
        }

        if (!$data instanceof WeightedDataset) {
            $data = $data->toWeightedDataset();
        }

        $this->labels = [1 => $labels[0], -1 => $labels[1]];
        $this->ensemble = $this->influence = [];

        $outcomes = $data->outcomes();

        for ($round = 1; $round <= $this->experts; $round++) {
            $expert = $this->reflector->newInstanceArgs($this->args);
            $error = 0;

            $expert->train($data->generateRandomWeightedSubsetWithReplacement($this->ratio));

            foreach ($data as $i => $sample) {
                if ($expert->predict($sample)->outcome() !== $outcomes[$i]) {
                    $error += $data->weight($i);
                }
            }

            $sum = $data->totalWeight();
            $error /= $sum;
            $influence = 0.5 * log((1 - $error) / ($error ? $error : self::EPSILON));

            foreach ($data as $i => $sample) {
                $prediction = $expert->predict($sample)->outcome() === $this->labels[1] ? 1 : -1;
                $outcome = $outcomes[$i] === $this->labels[1] ? 1 : -1;

                $data->setWeight($i,  $data->weight($i) * exp(-$influence * $outcome * $prediction) / $sum);
            }

            $this->influence[] = $influence;
            $this->ensemble[] = $expert;

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
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $sum = 0;

        foreach ($this->ensemble as $i => $expert) {
            $prediction = $expert->predict($sample)->outcome() === $this->labels[1] ? 1 : -1;

            $sum += $this->influence[$i] * $prediction;
        }

        return new Prediction($this->labels[$sum > 0 ? 1 : -1]);
    }
}
