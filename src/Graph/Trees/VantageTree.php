<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Clique;
use Rubix\ML\Graph\Nodes\Hypersphere;
use Rubix\ML\Graph\Nodes\VantagePoint;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Subadditive;
use Rubix\ML\Exceptions\InvalidArgumentException;
use SplMaxHeap;
use SplObjectStorage;

use function array_pop;
use function is_nan;

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
    protected int $maxLeafSize;

    /**
     * The distance function to use when computing the distances.
     *
     * @var Distance
     */
    protected Distance $kernel;

    /**
     * The root node of the tree.
     *
     * @var VantagePoint|null
     */
    protected ?VantagePoint $root = null;

    /**
     * @param int $maxLeafSize
     * @param Distance|null $kernel
     * @throws InvalidArgumentException
     */
    public function __construct(int $maxLeafSize = 30, ?Distance $kernel = null)
    {
        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException('Max leaf size must be'
                . " greater than 0, $maxLeafSize given.");
        }

        if ($kernel and !$kernel instanceof Subadditive) {
            throw new InvalidArgumentException('Distance kernel must implement'
                . ' the Subadditive interface.');
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
        $this->root = VantagePoint::split($dataset, $this->kernel);

        $stack = [$this->root];

        while ($current = array_pop($stack)) {
            [$left, $right] = $current->subsets();

            $current->cleanup();

            if ($left->numSamples() > $this->maxLeafSize) {
                $node = VantagePoint::split($left, $this->kernel);

                // Left branch has potential to collapse into a point.
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
     * @param list<string|int|float> $sample
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

        $heap = new SplMaxHeap();

        $stack = $this->path($sample);

        while ($current = array_pop($stack)) {
            if ($current instanceof VantagePoint) {
                $radius = $heap->count() === $k ? $heap->top()[0] : INF;

                foreach ($current->children() as $child) {
                    if (!$visited->offsetExists($child)) {
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
                $labels = $current->dataset()->labels();

                foreach ($current->dataset()->samples() as $i => $neighbor) {
                    $distance = $this->kernel->compute($sample, $neighbor);

                    if (is_nan($distance)) {
                        continue;
                    }

                    if ($heap->count() < $k) {
                        $heap->insert([$distance, $neighbor, $labels[$i]]);

                        continue;
                    }

                    if ($distance >= $heap->top()[0]) {
                        continue;
                    }

                    $heap->extract();

                    $heap->insert([$distance, $neighbor, $labels[$i]]);
                }

                $visited->attach($current);
            }
        }

        $samples = $labels = $distances = [];

        foreach ($heap as [$distance, $neighbor, $label]) {
            $samples[] = $neighbor;
            $labels[] = $label;
            $distances[] = $distance;
        }

        return [$samples, $labels, $distances];
    }

    /**
     * Return all samples, labels, and distances within a given radius of a sample.
     *
     * @param list<string|int|float> $sample
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
        $this->root = null;
    }

    /**
     * Return the path of a sample taken from the root node to a leaf node
     * in an array.
     *
     * @param list<string|int|float> $sample
     * @return list<Hypersphere>
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

                if ($left instanceof Hypersphere and $right instanceof Hypersphere) {
                    $distance = $this->kernel->compute($sample, $left->center());

                    if ($distance <= $left->radius()) {
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
     * @return string
     */
    public function __toString() : string
    {
        return "Vantage Tree (max_leaf_size: {$this->maxLeafSize}, kernel: {$this->kernel})";
    }
}
