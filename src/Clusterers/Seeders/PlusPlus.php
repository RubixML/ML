<?php

namespace Rubix\ML\Clusterers\Seeders;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Specifications\DatasetIsNotEmpty;

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
    protected \Rubix\ML\Kernels\Distance\Distance $kernel;

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
     * @internal
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $k
     * @return list<list<string|int|float>>
     */
    public function seed(Dataset $dataset, int $k) : array
    {
        DatasetIsNotEmpty::with($dataset)->check();

        $centroids = $dataset->randomSubset(1)->samples();

        while (count($centroids) < $k) {
            $weights = [];

            foreach ($dataset->samples() as $sample) {
                $bestDistance = INF;

                foreach ($centroids as $centroid) {
                    $distance = $this->kernel->compute($sample, $centroid);

                    if ($distance < $bestDistance) {
                        $bestDistance = $distance;
                    }
                }

                $weights[] = $bestDistance ** 2;
            }

            $centroids[] = $dataset->randomWeightedSubsetWithReplacement(1, $weights)->sample(0);
        }

        return $centroids;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Plus Plus (kernel: {$this->kernel})";
    }
}
