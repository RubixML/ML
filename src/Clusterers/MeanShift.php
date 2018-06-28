<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;
use RuntimeException;

class MeanShift implements Clusterer, Persistable
{
    /**
     * The bandwidth of the radial basis function kernel. i.e. The maximum
     * distance between two points to be considered neighbors.
     *
     * @var float
     */
    protected $radius;

    /**
     * The distance kernel to use when computing the distances.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The sensitivity threshold. i.e. the minimum change in the centroid means
     * necessary for the algorithm to continue learning.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The computed centroid vectors of the training data.
     *
     * @var array
     */
    protected $centroids = [
        //
    ];

    /**
     * @param  float  $radius
     * @param  \Rubix\ML\Kernels\Distance\Distance  $kernel
     * @param  float  $threshold
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $radius, Distance $kernel = null,
                                float $threshold = 1e-8, int $epochs = PHP_INT_MAX)
    {
        if ($radius <= 0) {
            throw new InvalidArgumentException('Radius must be greater than'
                . ' 0');
        }

        if ($threshold < 0) {
            throw new InvalidArgumentException('Threshold cannot be set to less'
                . ' than 0.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if (!isset($kernel)) {
            $kernel = new Euclidean();
        }

        $this->radius = $radius;
        $this->kernel = $kernel;
        $this->threshold = $threshold;
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
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $this->centroids = $dataset->samples();

        $n = $dataset->numColumns();

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            $previous = $this->centroids;

            foreach ($this->centroids as $i => &$centroid) {
                $weighted = array_fill(0, $n, 0.0);
                $total = array_fill(0, $n, 0.0);

                foreach ($dataset as $sample) {
                    $distance = $this->kernel->compute($sample, $centroid);

                    if ($distance <= $this->radius) {
                        foreach ($sample as $column => $feature) {
                            $weight = exp(-(($distance ** 2)
                                / (2 * $this->radius ** 2)));

                            $weighted[$column] += $weight * $feature;
                            $total[$column] += $weight;
                        }
                    }
                }

                foreach ($centroid as $column => &$mean) {
                    $mean = $weighted[$column] / $total[$column];
                }

                foreach ($this->centroids as $j => $target) {
                    if ($i !== $j) {
                        $distance = $this->kernel->compute($centroid, $target);

                        if ($distance < $this->radius) {
                            unset($this->centroids[$j]);
                        }
                    }
                }
            }

            $this->centroids = array_map('unserialize',
                array_unique(array_map('serialize', $this->centroids)));

            $change = 0.0;

            foreach ($this->centroids as $i => $centroid) {
                foreach ($centroid as $j => $mean) {
                    $change += abs($previous[$i][$j] - $mean);
                }
            }

            if ($change < $this->threshold) {
                break 1;
            }
        }

        $this->centroids = array_values($this->centroids);
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
        $best = ['distance' => INF, 'label' => -1];

        foreach ($this->centroids as $label => $centroid) {
            $distance = $this->kernel->compute($sample, $centroid);

            if ($distance < $best['distance']) {
                $best['distance'] = $distance;
                $best['label'] = (int) $label;
            }
        }

        return $best['label'];
    }
}
