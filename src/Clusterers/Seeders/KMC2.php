<?php

namespace Rubix\ML\Clusterers\Seeders;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;

use function count;

use const Rubix\ML\EPSILON;

/**
 * K-MC2
 *
 * A fast Plus Plus approximator that replaces the brute force method with a substantially
 * faster Markov Chain Monte Carlo (MCMC) sampling procedure with comparable results.
 *
 * References:
 * [1] O. Bachem et al. (2016). Approximate K-Means++ in Sublinear Time.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KMC2 implements Seeder
{
    /**
     * The number of candidate nodes in the Markov Chain.
     *
     * @var int
     */
    protected $m;

    /**
     * The distance kernel used to compute the distance between samples.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * @param int $m
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \InvalidArgumentException
     */
    public function __construct(int $m = 50, ?Distance $kernel = null)
    {
        if ($m < 1) {
            throw new InvalidArgumentException('The number of candidates'
                . " m must be greater than 1, $m given.");
        }

        $this->m = $m;
        $this->kernel = $kernel ?? new Euclidean();
    }

    /**
     * Seed k cluster centroids from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $k
     * @throws \RuntimeException
     * @return array
     */
    public function seed(Dataset $dataset, int $k) : array
    {
        $centroids = $dataset->randomSubsetWithReplacement(1)->samples();

        while (count($centroids) < $k) {
            $candidates = $dataset->randomSubsetWithReplacement($this->m)->samples();

            $x = array_pop($candidates) ?? [];

            $target = end($centroids) ?: [];

            $xDistance = $this->kernel->compute($x, $target) ?: EPSILON;

            foreach ($candidates as $y) {
                $yDistance = $this->kernel->compute($y, $target);

                $probability = min(1., $yDistance / $xDistance);

                if ($probability === 1.) {
                    $xDistance = $yDistance;
                    $x = $y;

                    continue 1;
                }

                $threshold = rand(0, PHP_INT_MAX) / PHP_INT_MAX;

                if ($probability > $threshold) {
                    $xDistance = $yDistance;
                    $x = $y;
                }
            }

            $centroids[] = $x;
        }

        return $centroids;
    }
}
