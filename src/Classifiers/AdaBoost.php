<?php

namespace Rubix\Engine\Classifiers;

use Rubix\Engine\Supervised;
use Rubix\Engine\Persistable;
use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Labeled;
use InvalidArgumentException;
use RuntimeException;
use ReflectionClass;

class AdaBoost implements Supervised, BinaryClassifier, Persistable
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
    protected $classes = [
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
     * The weight of each training sample in the dataset.
     *
     * @var array
     */
    protected $weights = [
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
    public function __construct(string $base, array $params = [], int $experts = 100,
                                float $ratio = 0.1, float $threshold = 0.999)
    {
        $this->reflector = new ReflectionClass($base);

        if (!in_array(Classifier::class, $this->reflector->getInterfaceNames())) {
            throw new InvalidArgumentException('Base class must implement the'
                . ' classifier interface.');
        }

        if ($experts < 1) {
            throw new InvalidArgumentException('Must have at least 1 expert in'
                . ' the ensemble.');
        }

        if ($ratio < 0.01 or $ratio > 1) {
            throw new InvalidArgumentException('Sample ratio must be between'
                . ' 0.01 and 1.0.');
        }

        if ($threshold < 0 or $threshold > 1) {
            throw new InvalidArgumentException('Threshold must be between'
                . ' 0 and 1.');
        }

        $this->params = $params;
        $this->experts = $experts;
        $this->ratio = $ratio;
        $this->threshold = $threshold;
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
     * @param  \Rubix\Engine\Datasets\Labeled  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Labeled $dataset) : void
    {
        $classes = $dataset->possibleOutcomes();

        if (count($classes) !== 2) {
            throw new InvalidArgumentException('The number of unique outcomes'
                . ' must be exactly 2, ' . (string) count($classes) . ' found.');
        }

        $this->classes = [1 => $classes[0], -1 => $classes[1]];
        $this->weights = array_fill(0, count($dataset), 1 / count($dataset));

        $this->ensemble = $this->influence = [];

        for ($epoch = 1; $epoch <= $this->experts; $epoch++) {
            $estimator = $this->reflector->newInstanceArgs($this->params);

            $estimator->train($this->generateRandomWeightedSubset($dataset));

            $predictions = $estimator->predict($dataset);

            $error = 0.0;

            foreach ($dataset->labels() as $index => $outcome) {
                if ($predictions[$index] !== $outcome) {
                    $error += $this->weights[$index];
                }
            }

            $total = array_sum($this->weights);
            $error /= $total;
            $influence = 0.5 * log((1 - $error) / ($error + self::EPSILON));

            foreach ($dataset->labels() as $index => $outcome) {
                $x = $predictions[$index] === $this->classes[1] ? 1 : -1;
                $y = $outcome === $this->classes[1] ? 1 : -1;

                $this->weights[$index] *= exp(-$influence * $x * $y) / $total;
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
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $totals = array_fill(0, $samples->numRows(), 0.0);

        foreach ($this->ensemble as $i => $estimator) {
            foreach ($estimator->predict($samples) as $j => $prediction) {
                $output = $prediction === $this->classes[1] ? 1 : -1;

                $totals[$j] += $this->influence[$i] * $output;
            }
        }

        $predictions = [];

        foreach ($totals as $total) {
            $predictions[] = $this->classes[$total > 0 ? 1 : -1];
        }

        return $predictions;
    }

    /**
     * Generate a random weighted subset with replacement.
     *
     * @param  \Rubix\Engine\Datasets\Labeled  $dataset
     * @throws \InvalidArgumentException
     * @return self
     */
    public function generateRandomWeightedSubset(Labeled $dataset) : Labeled
    {
        $n = round($this->ratio * $dataset->numRows());
        $total = array_sum($this->weights);
        $scale = pow(10, 8);

        list($samples, $labels) = $dataset->all();

        $subset = [];

        for ($i = 0; $i < $n; $i++) {
            $random = random_int(0, $total * $scale) / $scale;

            for ($index = 0; $index < $dataset->numRows(); $index++) {
                $random -= $this->weights[$index];

                if ($random < 0) {
                    break 1;
                }
            }

            $subset[0][] = $samples[$index];
            $subset[1][] = $labels[$index];
        }

        return new Labeled(...$subset);
    }
}
