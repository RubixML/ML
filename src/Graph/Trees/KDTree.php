<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Graph\Nodes\Box;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Hypercube;
use Rubix\ML\Graph\Nodes\Neighborhood;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;
use SplObjectStorage;

/**
 * K-d Tree
 *
 * A multi-dimensional binary search tree for fast nearest neighbor queries.
 * Each node maintains its own *bounding box* that is used to prune off leaf
 * nodes during search.
 *
 * [1] J. L. Bentley. (1975). Multidimensional Binary Seach Trees Used for
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
     * @var \Rubix\ML\Graph\Nodes\Hypercube|null
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
                . " to form a neighborhood, $maxLeafSize given.");
        }

        $this->kernel = $kernel ?? new Euclidean();
        $this->maxLeafSize = $maxLeafSize;
    }

    /**
     * @return \Rubix\ML\Graph\Nodes\Hypercube|null
     */
    public function root() : ?Hypercube
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

        $this->root = Hypercube::split($dataset);

        $stack = [$this->root];

        while ($stack) {
            $current = array_pop($stack);

            if (!$current instanceof Hypercube) {
                continue 1;
            }

            [$left, $right] = $current->groups();

            $current->cleanup();

            if ($left->numRows() > $this->maxLeafSize) {
                $stack[] = $node = Hypercube::split($left);
    
                $current->attachLeft($node);
            } else {
                $current->attachLeft(Neighborhood::terminate($left));
            }
    
            if ($right->numRows() > $this->maxLeafSize) {
                $stack[] = $node = Hypercube::split($right);
    
                $current->attachRight($node);
            } else {
                $current->attachRight(Neighborhood::terminate($right));
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

            if ($current instanceof Hypercube) {
                $radius = $distances[$k - 1] ?? INF;

                foreach ($current->children() as $child) {
                    if (!$visited->contains($child)) {
                        if ($child instanceof Box) {
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

                continue 1;
            }

            if ($current instanceof Neighborhood) {
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
     * Run a range search over every cluster within radius and return
     * the labels and distances in a 2-tuple.
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

            if ($current instanceof Hypercube) {
                foreach ($current->children() as $child) {
                    if ($child instanceof Box) {
                        foreach ($child->sides() as $side) {
                            $distance = $this->kernel->compute($sample, $side);

                            if ($distance <= $radius) {
                                $stack[] = $child;

                                continue 2;
                            }
                        }
                    }
                }

                continue 1;
            }

            if ($current instanceof Neighborhood) {
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

            if ($current instanceof Hypercube) {
                if ($sample[$current->column()] < $current->value()) {
                    $current = $current->left();
                } else {
                    $current = $current->right();
                }

                continue 1;
            }

            if ($current instanceof Neighborhood) {
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
