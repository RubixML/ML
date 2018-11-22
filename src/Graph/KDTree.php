<?php

namespace Rubix\ML\Graph;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Spatial;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Coordinate;
use Rubix\ML\Graph\Nodes\Neighborhood;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;
use SplObjectStorage;

/**
 * K-d Tree
 *
 * A multidimensional binary search tree or *K-d* Tree is a structure for fast
 * retrieval (log n) by associative search.
 *
 * [1] J. L. Bentley. (1975). Multidimensional Binary Seach Trees Used for
 * Associative Searching.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KDTree implements Tree
{
    /**
     * The root node of the tree.
     *
     * @var \Rubix\ML\Graph\Nodes\Coordinate|null
     */
    protected $root;

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
     * The number of dimensions this tree encodes.
     *
     * @var int|null
     */
    protected $dimensionality;

    /**
     * @param  int  $maxLeafSize
     * @param  \Rubix\ML\Kernels\Distance\Distance|null  $kernel
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxLeafSize = 20, ?Distance $kernel = null)
    {
        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException("At least one sample is required"
                . " to form a neighborhood, $maxLeafSize given.");
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->maxLeafSize = $maxLeafSize;
        $this->kernel = $kernel;
    }

    /**
     * @return \Rubix\ML\Graph\Nodes\Coordinate|null
     */
    public function root() : ?Coordinate
    {
        return $this->root;
    }

    /**
     * Insert a root node into the tree and recursively split the training data
     * until a terminating condition is met.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function grow(Dataset $dataset) : void
    {
        $this->dimensionality = $dataset->numColumns();

        $depth = 1;

        $this->root = $this->findBestSplit($dataset, $depth);

        $stack = [[$this->root, $depth]];

        while($stack) {
            list($current, $depth) = array_pop($stack) ?? [];

            list($left, $right) = $current->groups();

            $depth++;

            if ($left->numRows() > $this->maxLeafSize) {
                $node = $this->findBestSplit($left, $depth);
    
                $current->attachLeft($node);

                $stack[] = [$node, $depth];
            } else {
                $current->attachLeft(new Neighborhood($left));
            }
    
            if ($right->numRows() > $this->maxLeafSize) {
                $node = $this->findBestSplit($right, $depth);
    
                $current->attachRight($node);

                $stack[] = [$node, $depth];
            } else {
                $current->attachRight(new Neighborhood($right));
            }

            $current->cleanup();
        }
    }

    /**
     * Run a k nearest neighbors search of every neighborhood and return
     * the labels and distances in a tuple.
     * 
     * @param  array  $sample
     * @param  int  $k
     * @throws \InvalidArgumentException
     * @return array[]
     */
    public function neighbors(array $sample, int $k = 1) : array
    {
        if ($k < 1) {
            throw new InvalidArgumentException('The number of nearest'
                . " neighbors must be greater than 0, $k given.");
        }

        $neighborhood = $this->search($sample);

        if (is_null($neighborhood)) {
            return [[], []];
        }

        list($samples, $labels) = $neighborhood->neighbors();

        $distances = [];

        foreach ($samples as $neighbor) {
            $distances[] = $this->kernel->compute($sample, $neighbor);
        }

        array_multisort($distances, $labels);

        $visited = new SplObjectStorage();
        $stack = [];

        $visited->attach($neighborhood);
        $stack[] = $neighborhood->parent();

        while ($stack) {
            $current = array_pop($stack);

            if (!$visited->contains($current)) {
                $visited->attach($current);

                if ($current instanceof Neighborhood) {
                    list($sHat, $lHat) = $current->neighbors();

                    foreach ($sHat as $neighbor) {
                        $distances[] = $this->kernel->compute($sample, $neighbor);
                    }

                    $labels = array_merge($labels, $lHat);

                    array_multisort($distances, $labels);
                }

                foreach ($current->children() as $child) {
                    if (!$visited->contains($child)) {
                        if ($child instanceof Spatial) {
                            foreach ($child->box() as $side) {
                                $distance = $this->kernel->compute($sample, $side);
        
                                if ($distance < ($distances[$k - 1] ?? INF)) {
                                    $stack[] = $child;
        
                                    continue 2;
                                }
                            }
                        }

                        $visited->attach($child);
                    }
                }
            }

            $parent = $current->parent();

            if (!isset($parent)) {
                continue 1;
            }

            $stack[] = $parent;
        }

        $distances = array_slice($distances, 0, $k);
        $labels = array_slice($labels, 0, $k);

        return [$distances, $labels];
    }

    /**
     * Search the tree for a neighborhood and return an array of samples and
     * labels.
     *
     * @param  array  $sample
     * @return \Rubix\ML\Graph\Nodes\Neighborhood|null
     */
    public function search(array $sample) : ?Neighborhood
    {
        $current = $this->root;

        while ($current) {
            if ($current instanceof Coordinate) {
                if ($sample[$current->column()] < $current->value()) {
                    $current = $current->left();
                } else {
                    $current = $current->right();
                }

                continue 1;
            }

            if ($current instanceof Neighborhood) {
                return $current;
            }
        }

        return null;
    }

    /**
     * Find the best split of a given subset of the training data.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Coordinate
     */
    protected function findBestSplit(Dataset $dataset, int $depth) : Coordinate
    {
        $column = $depth % $this->dimensionality;

        $value = Stats::median($dataset->column($column));

        return new Coordinate($column, $value, $dataset);
    }

    /**
     * Is the tree bare?
     *
     * @return bool
     */
    public function bare() : bool
    {
        return is_null($this->root);
    }
}
