<?php

namespace Rubix\ML\Graph\Trees;

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
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return void
     */
    public function grow(Labeled $dataset) : void
    {
        $this->dimensionality = $dataset->numColumns();

        $this->root = $this->findBestSplit($dataset, 1);

        $this->split($this->root, 1);
    }

    /**
     * Recursive function to split the training data adding coordinate nodes along
     * the way.
     *
     * @param  \Rubix\ML\Graph\Nodes\Coordinate  $current
     * @param  int  $depth
     * @return void
     */
    protected function split(Coordinate $current, int $depth) : void
    {
        list($left, $right) = $current->groups();

        $current->cleanup();

        if ($left->numRows() > $this->maxLeafSize) {
            $node = $this->findBestSplit($left, $depth);

            $current->attachLeft($node);

            $this->split($node, $depth + 1);
        } else {
            $current->attachLeft(new Neighborhood($left));
        }

        if ($right->numRows() > $this->maxLeafSize) {
            $node = $this->findBestSplit($right, $depth);

            $current->attachRight($node);

            $this->split($node, $depth + 1);
        } else {
            $current->attachRight(new Neighborhood($right));
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
            throw new InvalidArgumentException("The number of nearest"
                . " neighbors must be greater than 0, $k given.");
        }

        $neighborhood = $this->search($sample);

        if (is_null($neighborhood)) {
            return [];
        }
        
        $distances = [];

        foreach ($neighborhood->samples() as $neighbor) {
            $distances[] = $this->kernel->compute($sample, $neighbor);
        }

        $labels = $neighborhood->labels();

        array_multisort($distances, $labels);
        
        $visited = new SplObjectStorage();
        $visited->attach($neighborhood);

        $current = $neighborhood->parent();

        while ($current) {
            if (!$visited->contains($current)) {
                $visited->attach($current);

                if ($current instanceof Neighborhood) {
                    foreach ($current->samples() as $neighbor) {
                        $distances[] = $this->kernel->compute($sample, $neighbor);
                    }

                    $labels = array_merge($labels, $current->labels());

                    array_multisort($distances, $labels);
                }

                foreach ($current->children() as $child) {
                    if (!$visited->contains($child)) {
                        if ($child instanceof Spatial) {
                            foreach ($child->box() as $edge) {
                                $distance = $this->kernel->compute($sample, $edge);
        
                                if ($distance < $distances[$k - 1]) {
                                    $current = $child;
        
                                    continue 2;
                                }
                            }
                        }

                        $visited->attach($child);
                    }
                }
            }

            $current = $current->parent();
        }

        $labels = array_slice($labels, 0, $k);
        $distances = array_slice($distances, 0, $k);

        return [$labels, $distances];
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
                if ($sample[$current->index()] < $current->value()) {
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
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Coordinate
     */
    protected function findBestSplit(Labeled $dataset, int $depth) : Coordinate
    {
        $index = $depth % $this->dimensionality;

        $value = Stats::median($dataset->column($index));

        return new Coordinate($value, $index, $dataset);
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
