<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Ball;
use Rubix\ML\Graph\Nodes\Clique;
use Rubix\ML\Graph\Nodes\Hypersphere;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Exceptions\InvalidArgumentException;
use SplObjectStorage;

use function count;
use function array_slice;

/**
 * Ball Tree
 *
 * A binary spatial tree that partitions the dataset into successively smaller
 * and tighter *ball* nodes whose boundary are defined by a centroid and radius.
 * Ball Trees work well in higher dimensions since the partitioning schema does
 * not rely on a finite number of 1-dimensional axis aligned splits as with k-d
 * trees.
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
     * The maximum number of unique samples that each leaf node can contain.
     *
     * @var int
     */
    protected int $maxLeafSize;

    /**
     * The distance function to use when computing the distances.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected \Rubix\ML\Kernels\Distance\Distance $kernel;

    /**
     * The root node of the tree.
     *
     * @var \Rubix\ML\Graph\Nodes\Ball|null
     */
    protected ?\Rubix\ML\Graph\Nodes\Ball $root = null;

    /**
     * @param int $maxLeafSize
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $maxLeafSize = 30, ?Distance $kernel = null)
    {
        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException('At least one sample is required'
                . " to form a leaf node, $maxLeafSize given.");
        }

        $this->maxLeafSize = $maxLeafSize;
        $this->kernel = $kernel ?? new Euclidean();
    }

    /**
     * Return the height of the tree i.e. the number of levels.
     *
     * @internal
     *
     * @return int
     */
    public function height() : int
    {
        return $this->root ? $this->root->height() : 0;
    }

    /**
     * Return the balance factor of the tree. A balanced tree will have
     * a factor of 0 whereas an imbalanced tree will either be positive
     * or negative indicating the direction and degree of the imbalance.
     *
     * @internal
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
     * @internal
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
     * @internal
     *
     * @return \Rubix\ML\Kernels\Distance\Distance
     */
    public function kernel() : Distance
    {
        return $this->kernel;
    }

    /**
     * Insert a root node and recursively split the dataset until a terminating
     * condition is met.
     *
     * @internal
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function grow(Labeled $dataset) : void
    {
        $this->root = Ball::split($dataset, $this->kernel);

        $stack = [$this->root];

        while ($current = array_pop($stack)) {
            [$left, $right] = $current->subsets();

            $current->cleanup();

            if ($left->numSamples() > $this->maxLeafSize) {
                $node = Ball::split($left, $this->kernel);

                $current->attachLeft($node);

                $stack[] = $node;
            } elseif (!$left->empty()) {
                $current->attachLeft(Clique::terminate($left, $this->kernel));
            }

            if ($right->numSamples() > $this->maxLeafSize) {
                $node = Ball::split($right, $this->kernel);

                if ($node->isPoint()) {
                    $current->attachRight(Clique::terminate($right, $this->kernel));
                } else {
                    $current->attachRight($node);

                    $stack[] = $node;
                }
            } elseif (!$right->empty()) {
                $current->attachRight(Clique::terminate($right, $this->kernel));
            }
        }
    }

    /**
     * Run a k nearest neighbors search and return the samples, labels, and distances in a 3-tuple.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @param int $k
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return array{list<list<mixed>>,list<mixed>,list<float>}
     */
    public function nearest(array $sample, int $k = 1) : array
    {
        $visited = new SplObjectStorage();

        $stack = $this->path($sample);

        $samples = $labels = $distances = [];

        while ($current = array_pop($stack)) {
            if ($current instanceof Ball) {
                $radius = $distances[$k - 1] ?? INF;

                foreach ($current->children() as $child) {
                    if (!$visited->contains($child)) {
                        if ($child instanceof Hypersphere) {
                            $distance = $this->kernel->compute($sample, $child->center());

                            if ($distance - $child->radius() < $radius) {
                                $stack[] = $child;

                                continue;
                            }
                        }

                        $visited->attach($child);
                    }
                }

                $visited->attach($current);

                continue;
            }

            if ($current instanceof Clique) {
                $dataset = $current->dataset();

                foreach ($dataset->samples() as $neighbor) {
                    $distances[] = $this->kernel->compute($sample, $neighbor);
                }

                $samples = array_merge($samples, $dataset->samples());
                $labels = array_merge($labels, $dataset->labels());

                array_multisort($distances, $samples, $labels);

                if (count($samples) > $k) {
                    $samples = array_slice($samples, 0, $k);
                    $labels = array_slice($labels, 0, $k);
                    $distances = array_slice($distances, 0, $k);
                }

                $visited->attach($current);
            }
        }

        return [$samples, $labels, $distances];
    }

    /**
     * Return all samples, labels, and distances within a given radius of a sample.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @param float $radius
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return array{list<list<mixed>>,list<mixed>,list<float>}
     */
    public function range(array $sample, float $radius) : array
    {
        /** @var list<Ball|Clique> */
        $stack = [$this->root];

        $samples = $labels = $distances = [];

        while ($current = array_pop($stack)) {
            if ($current instanceof Ball) {
                foreach ($current->children() as $child) {
                    if ($child instanceof Hypersphere) {
                        $distance = $this->kernel->compute($sample, $child->center());

                        if ($distance - $child->radius() < $radius) {
                            $stack[] = $child;
                        }
                    }
                }

                continue;
            }

            if ($current instanceof Clique) {
                $dataset = $current->dataset();

                foreach ($dataset->samples() as $i => $neighbor) {
                    $distance = $this->kernel->compute($sample, $neighbor);

                    if ($distance <= $radius) {
                        $samples[] = $neighbor;
                        $labels[] = $dataset->label($i);
                        $distances[] = $distance;
                    }
                }
            }
        }

        return [$samples, $labels, $distances];
    }

    /**
     * Destroy the tree.
     *
     * @internal
     */
    public function destroy() : void
    {
        $this->root = null;
    }

    /**
     * Return the path of a sample taken from the root node to a leaf node in an array.
     *
     * @param list<string|int|float> $sample
     * @return list<\Rubix\ML\Graph\Nodes\Hypersphere>
     */
    protected function path(array $sample) : array
    {
        $current = $this->root;

        $path = [];

        while ($current) {
            $path[] = $current;

            if ($current instanceof Ball) {
                $left = $current->left();
                $right = $current->right();

                if ($left instanceof Hypersphere and $right instanceof Hypersphere) {
                    $lDistance = $this->kernel->compute($sample, $left->center());
                    $rDistance = $this->kernel->compute($sample, $right->center());

                    if ($lDistance < $rDistance) {
                        $current = $left;
                    } else {
                        $current = $right;
                    }

                    continue;
                }

                if ($left instanceof Hypersphere) {
                    $current = $left;

                    continue;
                }

                if ($right instanceof Hypersphere) {
                    $current = $right;

                    continue;
                }
            }

            break;
        }

        return $path;
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
        return "Ball Tree (max leaf size: {$this->maxLeafSize}, kernel: {$this->kernel})";
    }
}
