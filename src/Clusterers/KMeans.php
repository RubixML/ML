<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;
use RuntimeException;

class KMeans implements Clusterer, Online, Persistable
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
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

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
     * @param  \Rubix\ML\Contracts\Distance  $kernel
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k, Distance $kernel = null, int $epochs = PHP_INT_MAX)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Must target at least one'
                . ' cluster.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if (!isset($kernel)) {
            $kernel = new Euclidean();
        }

        $this->k = $k;
        $this->kernel = $kernel;
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
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return array
     */
    public function train(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if ($dataset->numRows() < $this->k) {
            throw new RuntimeException('The number of samples cannot be less'
                . ' than the parameter K.');
        }

        $this->centroids = $dataset->randomize()->tail($this->k)->samples();

        $this->partial($dataset);
    }

    /**
     * Pick K random samples and assign them as centroids. Compute the coordinates
     * of the centroids by clustering the points based on each sample's distance
     * from one of the k centroids, then recompute the centroid coordinate as the
     * mean of the new cluster.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return array
     */
    public function partial(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if (empty($this->centroids)) {
            $this->train($dataset);
        }

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
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($dataset as $sample) {
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
            $distance = $this->kernel->compute($sample, $centroid);

            if ($distance < $best['distance']) {
                $best = ['distance' => $distance, 'label' => $label];
            }
        }

        return $best['label'];
    }
}
