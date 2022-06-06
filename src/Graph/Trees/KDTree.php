<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\DataType;
use Rubix\ML\Graph\Nodes\Box;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Hypercube;
use Rubix\ML\Graph\Nodes\Neighborhood;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Exceptions\InvalidArgumentException;
use SplObjectStorage;

use function count;
use function array_slice;
use function in_array;

/**
 * K-d Tree
 *
 * A multi-dimensional binary search tree for fast nearest neighbor queries.
 * The K-d tree construction algorithm separates data points into bounded
 * hypercubes or *bounding boxes* that are used to prune off nodes during
 * nearest neighbor and range searches.
 *
 * [1] J. L. Bentley. (1975). Multidimensional Binary Search Trees Used for
 * Associative Searching.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDTree implements BinaryTree, Spatial
{
    /**
     * The maximum number of samples that each neighborhood node can contain.
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
     * @var \Rubix\ML\Graph\Nodes\Box|null
     */
    protected ?\Rubix\ML\Graph\Nodes\Box $root = null;

    /**
     * @param int $maxLeafSize
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $maxLeafSize = 30, ?Distance $kernel = null)
    {
        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException('At least one sample is required'
                . " to form a neighborhood, $maxLeafSize given.");
        }

        if ($kernel and !in_array(DataType::continuous(), $kernel->compatibility())) {
            throw new InvalidArgumentException('Distance kernel must be'
                . ' compatible with continuous features.');
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
     * Return the balance factor of the tree. A balanced tree will have  a factor of 0 whereas
     * an imbalanced tree will either be positive or negative indicating the direction and
     * degree of the imbalance.
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
     * Insert a root node and recursively split the dataset until a terminating condition is met.
     *
     * @internal
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function grow(Labeled $dataset) : void
    {
        if ($dataset->featureType(0) != DataType::continuous() or !$dataset->homogeneous()) {
            throw new InvalidArgumentException('KD Tree only works with continuous features.');
        }

        $this->root = Box::split($dataset);

        $stack = [$this->root];

        while ($current = array_pop($stack)) {
            [$left, $right] = $current->subsets();

            $current->cleanup();

            if ($left->numSamples() > $this->maxLeafSize) {
                $node = Box::split($left);

                if ($node->isPoint()) {
                    $current->attachLeft(Neighborhood::terminate($left));
                } else {
                    $current->attachLeft($node);

                    $stack[] = $node;
                }
            } elseif (!$left->empty()) {
                $current->attachLeft(Neighborhood::terminate($left));
            }

            if ($right->numSamples() > $this->maxLeafSize) {
                $node = Box::split($right);

                $current->attachRight($node);

                $stack[] = $node;
            } elseif (!$right->empty()) {
                $current->attachRight(Neighborhood::terminate($right));
            }
        }
    }

    /**
     * Run a k nearest neighbors search and return the samples, labels, and distances in a 3-tuple.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @param int $k
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return array{list<list<mixed>>,list<mixed>,list<float>}
     */
    public function nearest(array $sample, int $k = 1) : array
    {
        $visited = new SplObjectStorage();

        $samples = $labels = $distances = [];

        $stack = $this->path($sample);

        while ($current = array_pop($stack)) {
            if ($current instanceof Box) {
                $radius = $distances[$k - 1] ?? INF;

                foreach ($current->children() as $child) {
                    if (!$visited->contains($child)) {
                        if ($child instanceof Hypercube) {
                            foreach ($child->sides() as $side) {
                                $distance = $this->kernel->compute($sample, $side);

                                if ($distance < $radius) {
                                    $stack[] = $child;

                                    continue 2;
                                }
                            }
                        }

                        $visited->attach($child);
                    }
                }

                $visited->attach($current);

                continue;
            }

            if ($current instanceof Neighborhood) {
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
     * Run a range search over every cluster within radius and return the samples, labels and distances in a 3-tuple.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @param float $radius
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return array{list<list<mixed>>,list<mixed>,list<float>}
     */
    public function range(array $sample, float $radius) : array
    {
        $samples = $labels = $distances = [];

        /** @var list<Box|Neighborhood> */
        $stack = [$this->root];

        while ($current = array_pop($stack)) {
            if ($current instanceof Box) {
                foreach ($current->children() as $child) {
                    if ($child instanceof Hypercube) {
                        foreach ($child->sides() as $side) {
                            $distance = $this->kernel->compute($sample, $side);

                            if ($distance <= $radius) {
                                $stack[] = $child;

                                continue 2;
                            }
                        }
                    }
                }

                continue;
            }

            if ($current instanceof Neighborhood) {
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
     * @param list<int|float> $sample
     * @return list<\Rubix\ML\Graph\Nodes\Node|null>
     */
    protected function path(array $sample) : array
    {
        $current = $this->root;

        $path = [$current];

        while ($current instanceof Box) {
            if ($sample[$current->column()] < $current->value()) {
                $current = $current->left();
            } else {
                $current = $current->right();
            }

            if ($current) {
                $path[] = $current;
            }
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
        return "K-d Tree (max leaf size: {$this->maxLeafSize}, kernel: {$this->kernel})";
    }
}
