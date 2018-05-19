<?php

namespace Rubix\Engine\Clusterers;

use Rubix\Engine\Datasets\Unsupervised;
use Rubix\Engine\Metrics\Distance\Euclidean;
use Rubix\Engine\Metrics\Distance\Distance;
use InvalidArgumentException;
use RuntimeException;

class KMeans implements Clusterer
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
     * @var \Rubix\Engine\Contracts\Distance
     */
    protected $distanceFunction;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

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
            throw new InvalidArgumentException('Parameter K must be an integer larger than 1.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Max epochs must be larger than 1.');
        }

        if (!isset($distanceFunction)) {
            $distanceFunction = new Euclidean();
        }

        $this->k = $k;
        $this->distanceFunction = $distanceFunction;
        $this->epochs = $epochs;
    }

    /**
     * Cluster a given dataset based on the computed means of the centroids.
     *
     * @param  \Rubix\Engine\Datasets\Unsupervised  $dataset
     * @return array
     */
    public function cluster(Unsupervised $dataset) : array
    {
        $centroids = $this->findCentroids($dataset->samples());

        $clusters = [];

        foreach ($dataset as $sample) {
            $label = $this->label($sample, $centroids);

            $clusters[$label][] = $sample;
        }

        return $clusters;
    }

    /**
     * Pick K random samples and assign them as centroids. Compute the coordinates
     * of the centroids by clustering the points based on each sample's distance
     * from one of the k centroids, then recompute the centroid coordinate as the
     * mean of the new cluster.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return array
     */
    protected function findCentroids(array $samples) : array
    {
        if (count($samples) < $this->k) {
            throw new RuntimeException('The number of samples cannot be less than the parameter K.');
        }

        $labels = array_fill(0, count($samples), null);
        $sizes = array_fill(0, $this->k, 0);
        $changed = true;

        shuffle($samples);

        $centroids = array_splice($samples, 0, $this->k);

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($samples as $i => $sample) {
                $label = $this->label($sample, $centroids);

                if ($label !== $labels[$i]) {
                    $labels[$i] = $label;
                    $sizes[$label]++;
                } else {
                    $changed = false;
                }

                foreach ($centroids[$label] as $column => &$mean) {
                    $mean = ($mean * ($sizes[$label] - 1) + $sample[$column]) / $sizes[$label];
                }
            }

            if ($changed === false) {
                break 1;
            }
        }

        return $centroids;
    }

    /**
     * Label a given sample based on its distance from a particular centroid.
     *
     * @param  array  $sample
     * @param  array  $centroids
     * @return int
     */
    protected function label(array $sample, array $centroids) : int
    {
        $best = ['distance' => INF, 'label' => null];

        foreach ($centroids as $label => $centroid) {
            $distance = $this->distanceFunction->compute($sample, $centroid);

            if ($distance < $best['distance']) {
                $best = [
                    'distance' => $distance,
                    'label' => $label,
                ];
            }
        }

        return $best['label'];
    }
}
