<?php

namespace Rubix\ML\Clusterers\Seeders;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;

use function count;

/**
 * Plus Plus
 *
 * This seeder attempts to maximize the chances of seeding distant clusters while still
 * remaining random. It does so by sequentially selecting random samples weighted by their
 * distance from the previous seed.
 *
 * References:
 * [1] D. Arthur et al. (2006). k-means++: The Advantages of Careful Seeding.
 * [2] A. Stetco et al. (2015). Fuzzy C-means++: Fuzzy C-means with effective
 * seeding initialization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PlusPlus implements Seeder
{
    /**
     * The distance kernel used to compute the distance between samples.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     */
    public function __construct(?Distance $kernel = null)
    {
        $this->kernel = $kernel ?? new Euclidean();
    }

    /**
     * Seed k cluster centroids from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $k
     * @return array[]
     */
    public function seed(Dataset $dataset, int $k) : array
    {
        $centroids = $dataset->randomSubsetWithReplacement(1)->samples();

        $samples = $dataset->samples();

        while (count($centroids) < $k) {
            $weights = [];

            foreach ($samples as $sample) {
                $bestDistance = INF;

                foreach ($centroids as $centroid) {
                    $distance = $this->kernel->compute($sample, $centroid);

                    if ($distance < $bestDistance) {
                        $bestDistance = $distance;
                    }
                }

                $weights[] = $bestDistance ** 2;
            }

            $subset = $dataset->randomWeightedSubsetWithReplacement(1, $weights);

            $centroids[] = $subset->sample(0);
        }

        return $centroids;
    }
}
