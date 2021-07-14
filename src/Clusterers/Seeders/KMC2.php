<?php

namespace Rubix\ML\Clusterers\Seeders;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Exceptions\InvalidArgumentException;

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
    protected int $m;

    /**
     * The distance kernel used to compute the distance between samples.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected \Rubix\ML\Kernels\Distance\Distance $kernel;

    /**
     * @param int $m
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $m = 50, ?Distance $kernel = null)
    {
        if ($m < 1) {
            throw new InvalidArgumentException('M must be greater'
                . " than 0, $m given.");
        }

        $this->m = $m;
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

        $max = getrandmax();

        while (count($centroids) < $k) {
            $candidates = $dataset->randomSubsetWithReplacement($this->m)->samples();

            $x = array_pop($candidates) ?? [];

            $target = end($centroids) ?: [];

            $xDistance = $this->kernel->compute($x, $target) ?: EPSILON;

            foreach ($candidates as $candidate) {
                $yDistance = $this->kernel->compute($candidate, $target);

                $density = min(1.0, $yDistance / $xDistance);

                $threshold = rand() / $max;

                if ($density > $threshold) {
                    $xDistance = $yDistance;

                    $x = $candidate;
                }
            }

            $centroids[] = $x;
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
        return "KMC2 (m: {$this->m}, kernel: {$this->kernel})";
    }
}
