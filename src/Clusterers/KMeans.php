<?php

namespace Rubix\Engine\Clusterers;

use Rubix\Engine\Persistable;
use Rubix\Engine\Unsupervised;
use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Metrics\Distance\Euclidean;
use Rubix\Engine\Metrics\Distance\Distance;
use InvalidArgumentException;
use RuntimeException;

class KMeans implements Unsupervised, Clusterer, Persistable
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
     * @var \Rubix\Engine\Metrics\Distance\Distance
     */
    protected $distanceFunction;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The computed centroid vectors of the training data.
     *
     * @var int
     */
    protected $centroids = [
        //
    ];

    /**
     * @param  int  $k
     * @param  \Rubix\Engine\Contracts\Distance  $distanceFunction
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k, Distance $distanceFunction = null, int $epochs = PHP_INT_MAX)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Target clusters must be'
                . ' greater than 1.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Max epochs must be greater than'
                . ' 1.');
        }

        if (!isset($distanceFunction)) {
            $distanceFunction = new Euclidean();
        }

        $this->k = $k;
        $this->distanceFunction = $distanceFunction;
        $this->epochs = $epochs;
    }

    /**
     * Return the computed cluster centroids of the training data.
     *
     * @return array
     */
    public function centroids() : array
    {
        return $this->centroids;
    }

    /**
     * Pick K random samples and assign them as centroids. Compute the coordinates
     * of the centroids by clustering the points based on each sample's distance
     * from one of the k centroids, then recompute the centroid coordinate as the
     * mean of the new cluster.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return array
     */
    public function train(Dataset $dataset) : void
    {
        if ($dataset->numRows() < $this->k) {
            throw new RuntimeException('The number of samples cannot be less'
                . ' than the parameter K.');
        }

        $this->centroids = $dataset->randomize()->tail($this->k)->samples();

        $labels = array_fill(0, $dataset->numRows(), null);
        $sizes = array_fill(0, $this->k, 0);
        $changed = true;

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($dataset as $i => $sample) {
                $label = $this->label($sample);

                if ($label !== $labels[$i]) {
                    $labels[$i] = $label;
                    $sizes[$label]++;
                } else {
                    $changed = false;
                }

                foreach ($this->centroids[$label] as $column => &$mean) {
                    $mean = ($mean * ($sizes[$label] - 1) + $sample[$column])
                        / ($sizes[$label] + self::EPSILON);
                }
            }

            if ($changed === false) {
                break 1;
            }
        }
    }

    /**
     * Cluster the dataset by assigning a label to each sample.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($samples as $sample) {
            $predictions[] = $this->label($sample);
        }

        return $predictions;
    }

    /**
     * Label a given sample based on its distance from a particular centroid.
     *
     * @param  array  $sample
     * @return int
     */
    protected function label(array $sample) : int
    {
        $best = ['distance' => INF, 'label' => null];

        foreach ($this->centroids as $label => $centroid) {
            $distance = $this->distanceFunction->compute($sample, $centroid);

            if ($distance < $best['distance']) {
                $best = ['distance' => $distance, 'label' => $label];
            }
        }

        return $best['label'];
    }
}
