<?php

namespace Rubix\Engine;

use Rubix\Engine\Graph\DistanceFunctions\Euclidean;
use Rubix\Engine\Graph\DistanceFunctions\DistanceFunction;
use InvalidArgumentException;

class KMeans implements Classifier, Clusterer
{
    /**
     * The target number of clusters.
     *
     * @var int
     */
    protected $k;

    /**
     * The distance function to use when computing the distances.
     *
     * @var \Rubix\Engine\Graph\DistanceFunctions\DistanceFunction
     */
    protected $distanceFunction;

    /**
     * The maximum number of iterations to run until the clusterer terminates.
     *
     * @var int
     */
    protected $maxIterations;

    /**
     * The k centroid vectors.
     *
     * @var array
     */
    protected $centroids = [
        //
    ];

    /**
     * @param  int  $k
     * @param  \Rubix\Engine\Graph\DistanceFunctions\DistanceFunction  $distanceFunction
     * @param  int  $maxIterations
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k, DistanceFunction $distanceFunction = null, int $maxIterations = PHP_INT_MAX)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be an integer greater than 1.');
        }

        if ($maxIterations < 1) {
            throw new InvalidArgumentException('Max interations must be greater than 1.');
        }

        if (!isset($distanceFunction)) {
            $distanceFunction = new Euclidean();
        }

        $this->k = $k;
        $this->distanceFunction = $distanceFunction;
        $this->maxIterations = $maxIterations;
    }

    /**
     * Return the learned coordinate vectors of the k centroids.
     *
     * @return array
     */
    public function centroids() : array
    {
        return $this->centroids;
    }

    /**
     * Alias of centroids().
     *
     * @return array
     */
    public function means() : array
    {
        return $this->centroids();
    }

    /**
     * Compute the coordinates of k centroids by clustering the training data based
     * on each sample's distance from one of the k centroids, then recompute the
     * centeroid coordinate as the mean of the new cluster's population.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function train(Dataset $data) : void
    {
        if (in_array(self::CATEGORICAL, $data->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with continuous samples.');
        }

        $labels = array_fill(0, $data->rows(), null);
        $sizes = array_fill(0, $this->k, 0);
        $points = $data->samples();
        $changed = true;

        shuffle($points);

        $this->centroids = array_splice($points, 0, $this->k);

        for ($i = 0; $i < $this->maxIterations; $i++) {
            foreach ($points as $id => $sample) {
                $label = $this->label($sample)['label'];

                if ($label !== $labels[$id]) {
                    $labels[$id] = $label;

                    $sizes[$label]++;
                } else {
                    $changed = false;
                }

                $n = $sizes[$label];

                foreach ($this->centroids[$label] as $column => &$mean) {
                    $mean = ($mean * ($n - 1) + $sample[$column]) / $n;
                }
            }

            if ($changed === false) {
                break 1;
            }
        }
    }

    /**
     * Cluster a given dataset based on the computed means of the centroids.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @return array
     */
    public function cluster(Dataset $data) : array
    {
        $this->train($data);

        $clusters = [];

        foreach ($data as $sample) {
            $label = $this->label($sample)['label'];

            $clusters[$label][] = $sample;
        }

        return $clusters;
    }

    /**
     * Classify the sample based on the distance from one of the k trained centroids.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $result = $this->label($sample);

        return new Prediction($result['label'], [
            'distance' => $result['distance'],
        ]);
    }

    /**
     * Label a given sample based on its distance from the k centroid means.
     *
     * @param  array  $sample
     * @return array
     */
    protected function label(array $sample) : array
    {
        $best = ['distance' => INF, 'label' => null];

        foreach ($this->centroids as $label => $centroid) {
            $distance = $this->distanceFunction->compute($sample, $centroid);

            if ($distance < $best['distance']) {
                $best = [
                    'distance' => $distance,
                    'label' => $label,
                ];
            }
        }

        return $best;
    }
}
