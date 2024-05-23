<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Clique;
use Rubix\ML\Graph\Nodes\Hypersphere;
use Rubix\ML\Graph\Nodes\VantagePoint;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Exceptions\InvalidArgumentException;
use SplObjectStorage;

use function count;
use function array_slice;
use function array_pop;
use function array_multisort;
use function array_merge;

/**
 * Vantage Tree
 *
 * A Vantage Point Tree is a binary spatial tree that divides samples by their distance from the center of
 * a cluster called the *vantage point*. Samples that are closer to the vantage point will be put into one
 * branch of the tree while samples that are farther away will be put into the other branch.
 *
 * References:
 * [1] P. N. Yianilos. (1993). Data Structures and Algorithms for Nearest Neighbor Search in General Metric
 * Spaces.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class VantageTree implements BinaryTree, Spatial
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
     * @var Distance
     */
    protected $kernel;

    /**
     * The root node of the tree.
     *
     * @var VantagePoint|null
     */
    protected $root;

    /**
     * @param int $maxLeafSize
     * @param Distance|null $kernel
     * @throws \InvalidArgumentException
     */
    public function __construct(int $maxLeafSize = 30, ?Distance $kernel = null)
    {
        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException('Max leaf size must be'
                . " greater than 0, $maxLeafSize given.");
        }

        $this->maxLeafSize = $maxLeafSize;
        $this->kernel = $kernel ?? new Euclidean();
    }

    /**
     * Return the height of the tree i.e. the number of levels.
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
     * @return Distance
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
     * @param Labeled $dataset
     * @throws InvalidArgumentException
     */
    public function grow(Labeled $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Tree requires a labeled dataset.');
        }

        $this->root = VantagePoint::split($dataset, $this->kernel);

        $stack = [$this->root];

        while ($current = array_pop($stack)) {
            [$left, $right] = $current->subsets();

            $current->cleanup();

            if ($left->numSamples() > $this->maxLeafSize) {
                $node = VantagePoint::split($left, $this->kernel);

                if ($node->isPoint()) {
                    $current->attachLeft(Clique::terminate($left, $this->kernel));
                } else {
                    $current->attachLeft($node);

                    $stack[] = $node;
                }
            } elseif (!$left->empty()) {
                $current->attachLeft(Clique::terminate($left, $this->kernel));
            }

            if ($right->numSamples() > $this->maxLeafSize) {
                $node = VantagePoint::split($right, $this->kernel);

                $current->attachRight($node);

                $stack[] = $node;
            } elseif (!$right->empty()) {
                $current->attachRight(Clique::terminate($right, $this->kernel));
            }
        }
    }

    /**
     * Run a k nearest neighbors search and return the samples, labels, and
     * distances in a 3-tuple.
     *
     * @param (string|int|float)[] $sample
     * @param int $k
     * @throws InvalidArgumentException
     * @return array<array<mixed>>
     */
    public function nearest(array $sample, int $k = 1) : array
    {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be'
                . " greater than 0, $k given.");
        }

        $visited = new SplObjectStorage();

        $stack = $this->path($sample);

        $samples = $labels = $distances = [];

        while ($current = array_pop($stack)) {
            if ($current instanceof VantagePoint) {
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
     * @param (string|int|float)[] $sample
     * @param float $radius
     * @throws InvalidArgumentException
     * @return array<array<mixed>>
     */
    public function range(array $sample, float $radius) : array
    {
        if ($radius <= 0.0) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        $samples = $labels = $distances = [];

        $stack = [$this->root];

        while ($current = array_pop($stack)) {
            if ($current instanceof VantagePoint) {
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
     */
    public function destroy() : void
    {
        unset($this->root);
    }

    /**
     * Return the path of a sample taken from the root node to a leaf node
     * in an array.
     *
     * @param (string|int|float)[] $sample
     * @return mixed[]
     */
    protected function path(array $sample) : array
    {
        $current = $this->root;

        $path = [];

        while ($current) {
            $path[] = $current;

            if ($current instanceof VantagePoint) {
                $left = $current->left();
                $right = $current->right();

                if ($left instanceof Hypersphere) {
                    $distance = $this->kernel->compute($sample, $left->center());

                    if ($distance <= $left->radius()) {
                        $current = $left;
                    } else {
                        $current = $right;
                    }
                }

                continue;
            }

            break;
        }

        return $path;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Vantage Tree (max_leaf_size: {$this->maxLeafSize}, kernel: {$this->kernel})";
    }
}
