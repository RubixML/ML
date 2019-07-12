<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Ball;
use Rubix\ML\Graph\Nodes\Cluster;
use Rubix\ML\Graph\Nodes\Hypersphere;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;
use SplObjectStorage;

/**
 * Ball Tree
 *
 * A binary spatial tree with *ball* nodes whose boundary is defined by a
 * centroid and radius.
 *
 * References:
 * [1] S. M. Omohundro. (1989). Five Balltree Construction Algorithms.
 * [2] M. Dolatshah et al. (2015). Ball*-tree: Efficient spatial indexing for
 * constrained nearest-neighbor search in metric spaces.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BallTree implements BinaryTree, Spatial
{
    /**
     * The maximum number of samples that each leaf node can contain.
     *
     * @var int
     */
    protected $maxLeafSize;

    /**
     * The distance function to use when computing the distances.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The root node of the tree.
     *
     * @var \Rubix\ML\Graph\Nodes\Hypersphere|null
     */
    protected $root;

    /**
     * @param int $maxLeafSize
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \InvalidArgumentException
     */
    public function __construct(int $maxLeafSize = 30, ?Distance $kernel = null)
    {
        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException('At least one sample is required'
                . " to form a leaf node, $maxLeafSize given.");
        }

        $this->kernel = $kernel ?? new Euclidean();
        $this->maxLeafSize = $maxLeafSize;
    }

    /**
     * @return \Rubix\ML\Graph\Nodes\Hypersphere|null
     */
    public function root() : ?Hypersphere
    {
        return $this->root;
    }

    /**
     * Return the height of the tree.
     *
     * @return int
     */
    public function height() : int
    {
        return $this->root ? $this->root->height() : 0;
    }

    /**
     * Return the balance of the tree.
     *
     * @return int
     */
    public function balance() : int
    {
        return $this->root ? $this->root->balance() : 0;
    }

    /**
     * Is the tree bare?
     *
     * @return bool
     */
    public function bare() : bool
    {
        return !$this->root;
    }

    /**
     * Return the distance kernel used to compute distances.
     *
     * @return \Rubix\ML\Kernels\Distance\Distance
     */
    public function kernel() : Distance
    {
        return $this->kernel;
    }

    /**
     * Insert a root node and recursively split the dataset a terminating
     * condition is met.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function grow(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Tree requires a labeled dataset.');
        }

        $this->root = Hypersphere::split($dataset, $this->kernel);

        $stack = [$this->root];

        while ($stack) {
            $current = array_pop($stack);

            if (!$current instanceof Hypersphere) {
                continue 1;
            }

            [$left, $right] = $current->groups();

            $current->cleanup();

            if ($left->numRows() > $this->maxLeafSize) {
                $stack[] = $node = Hypersphere::split($left, $this->kernel);
    
                $current->attachLeft($node);
            } else {
                $current->attachLeft(Cluster::terminate($left, $this->kernel));
            }
    
            if ($right->numRows() > $this->maxLeafSize) {
                $stack[] = $node = Hypersphere::split($right, $this->kernel);
    
                $current->attachRight($node);
            } else {
                $current->attachRight(Cluster::terminate($right, $this->kernel));
            }
        }
    }

    /**
     * Run a k nearest neighbors search and return the samples, labels, and
     * distances in a 3-tuple.
     *
     * @param array $sample
     * @param int $k
     * @throws \InvalidArgumentException
     * @return array[]
     */
    public function nearest(array $sample, int $k = 1) : array
    {
        if ($k < 1) {
            throw new InvalidArgumentException('The number of nearest'
                . " neighbors must be greater than 0, $k given.");
        }

        $visited = new SplObjectStorage();

        $samples = $labels = $distances = [];

        $stack = $this->tracePath($sample);

        while ($stack) {
            $current = array_pop($stack);

            if ($current instanceof Hypersphere) {
                $radius = $distances[$k - 1] ?? INF;

                foreach ($current->children() as $child) {
                    if (!$visited->contains($child)) {
                        if ($child instanceof Ball) {
                            $distance = $this->kernel->compute($sample, $child->center());
    
                            if ($distance < ($child->radius() + $radius)) {
                                $stack[] = $child;

                                continue 1;
                            }
                        }

                        $visited->attach($child);
                    }
                }

                $visited->attach($current);

                continue 1;
            }

            if ($current instanceof Cluster) {
                foreach ($current->samples() as $neighbor) {
                    $distances[] = $this->kernel->compute($sample, $neighbor);
                }

                $samples = array_merge($samples, $current->samples());
                $labels = array_merge($labels, $current->labels());

                array_multisort($distances, $samples, $labels);

                $visited->attach($current);
            }
        }

        return [
            array_slice($samples, 0, $k),
            array_slice($labels, 0, $k),
            array_slice($distances, 0, $k),
        ];
    }

    /**
     * Return all samples, labels, and distances within a given radius of a
     * sample.
     *
     * @param array $sample
     * @param float $radius
     * @throws \InvalidArgumentException
     * @return array[]
     */
    public function range(array $sample, float $radius) : array
    {
        if ($radius <= 0.) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        $samples = $labels = $distances = [];

        $stack = [$this->root];

        while ($stack) {
            $current = array_pop($stack);

            if ($current instanceof Hypersphere) {
                foreach ($current->children() as $child) {
                    if ($child instanceof Ball) {
                        $distance = $this->kernel->compute($sample, $child->center());

                        if ($distance <= ($child->radius() + $radius)) {
                            $stack[] = $child;
                        }
                    }
                }

                continue 1;
            }

            if ($current instanceof Cluster) {
                $lHat = $current->labels();

                foreach ($current->samples() as $i => $neighbor) {
                    $distance = $this->kernel->compute($sample, $neighbor);

                    if ($distance <= $radius) {
                        $samples[] = $neighbor;
                        $labels[] = $lHat[$i];
                        $distances[] = $distance;
                    }
                }
            }
        }

        return [$samples, $labels, $distances];
    }

    /**
     * Return the path taken from the root to a leaf node.
     *
     * @param array $sample
     * @return array
     */
    protected function tracePath(array $sample) : array
    {
        $current = $this->root;

        $path = [];

        while ($current) {
            $path[] = $current;

            if ($current instanceof Hypersphere) {
                $left = $current->left();
                $right = $current->right();

                if ($left instanceof Ball and $right instanceof Ball) {
                    $lHat = $this->kernel->compute($sample, $left->center());
                    $rHat = $this->kernel->compute($sample, $right->center());

                    if ($lHat < $rHat) {
                        $current = $current->left();
                    } else {
                        $current = $current->right();
                    }
                }

                continue 1;
            }

            if ($current instanceof Cluster) {
                break 1;
            }
        }

        return $path;
    }

    /**
     * Destroy the tree.
     */
    public function destroy() : void
    {
        unset($this->root);
    }
}
