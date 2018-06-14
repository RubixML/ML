<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Metrics\Distance\Distance;
use Rubix\ML\Metrics\Distance\Euclidean;
use InvalidArgumentException;
use SplPriorityQueue;

class KNearestNeighbors implements Multiclass, Online, Probabilistic
{
    /**
     * The number of neighbors to consider when making a prediction.
     *
     * @var int
     */
    protected $k;

    /**
     * The distance function to use when computing the distances.
     *
     * @var \Rubix\ML\Metrics\Distance\Distance
     */
    protected $distanceFunction;

    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * The memoized coordinate vectors of the training data.
     *
     * @var array
     */
    protected $samples = [
        //
    ];

    /**
     * The memoized labels.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * @param  int  $k
     * @param  \Rubix\ML\Metrics\Distance\Distance  $distanceFunction
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k = 5, Distance $distanceFunction = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('At least 1 neighbor is required'
                . ' to make a prediction.');
        }

        if (!isset($distanceFunction)) {
            $distanceFunction = new Euclidean();
        }

        $this->k = $k;
        $this->distanceFunction = $distanceFunction;
    }

    /**
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

        $this->classes = $this->samples = $this->labels = [];

        $this->partial($dataset);
    }

    /**
     * Store the sample and outcome arrays. No other work to be done as this is
     * a lazy learning algorithm.
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

        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $this->classes = array_merge($this->classes, $dataset->possibleOutcomes());
        $this->samples = array_merge($this->samples, $dataset->samples());
        $this->labels = array_merge($this->labels, $dataset->labels());
    }

    /**
     * Make a prediction based on the class probabilities.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($this->proba($samples) as $probabilities) {
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
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function proba(Dataset $samples) : array
    {
        $probabilities = array_fill(0, $samples->numRows(),
            array_fill_keys($this->classes, 0.0));

        foreach ($samples as $i => $sample) {
            $neighbors = $this->findNearestNeighbors($sample);

            $n = count($neighbors);

            foreach (array_count_values($neighbors) as $class => $count) {
                $probabilities[$i][$class] = $count / ($n + self::EPSILON);
            }
        }

        return $probabilities;
    }

    /**
     * Find the K nearest neighbors to the given sample vector.
     *
     * @param  array  $sample
     * @return array
     */
    protected function findNearestNeighbors(array $sample) : array
    {
        $computed = new SplPriorityQueue();
        $neighbors = [];

        foreach ($this->samples as $index => $neighbor) {
            $distance = $this->distanceFunction->compute($sample, $neighbor);

            $computed->insert($this->labels[$index], -$distance);
        }

        $n = (count($this->samples) >= $this->k
            ? $this->k : count($this->samples));

        for ($i = 0; $i < $n; $i++) {
            $neighbors[] = $computed->extract();
        }

        return $neighbors;
    }
}
