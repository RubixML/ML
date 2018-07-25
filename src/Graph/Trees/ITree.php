<?php

namespace Rubix\ML\Graph\Trees;


use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Nodes\Isolation;
use Rubix\ML\Graph\Nodes\Comparison;
use InvalidArgumentException;

/**
 * I Tree
 *
 * The base Isloation Tree.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ITree implements Tree
{
    /**
     * The root node of the tree.
     *
     * @var \Rubix\ML\Graph\Nodes\Comparison|null
     */
    protected $root;

    /**
     * The maximum depth of a branch before it is forced to terminate.
     *
     * @var int
     */
    protected $maxDepth;

    /**
     * The minimum number of samples that each node must contain in order to
     * form a consensus to make a prediction.
     *
     * @var int
     */
    protected $minSamples;

    /**
     * The C factor represents the average length of the path of a search.
     *
     * @var float
     */
    protected $c;

    /**
     * @param  int  $maxDepth
     * @param  int  $minSamples
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, int $minSamples = 5)
    {
        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have depth less'
                . ' than 1.');
        }

        if ($minSamples < 1) {
            throw new InvalidArgumentException('At least one sample is required'
                . ' to make a decision.');
        }

        $this->maxDepth = $maxDepth;
        $this->minSamples = $minSamples;
    }

    /**
     * Return the root node of the tree.
     *
     * @return \Rubix\ML\Graph\Nodes\Comparison|null
     */
    public function root() : ?Comparison
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
        $this->c = $this->calculateCFactor($dataset->numRows());

        $this->root = $this->findBestSplit($dataset, 1);

        $this->split($this->root, 1);
    }

    /**
     * Recursive function to split the training data adding decision nodes along
     * the way. The terminating conditions are a) split would make node
     * responsible for less values than $minSamples or b) the max depth of the
     * branch has been reached.
     *
     * @param  \Rubix\ML\Graph\Nodes\Comparison  $current
     * @param  int  $depth
     * @return void
     */
    protected function split(Comparison $current, int $depth) : void
    {
        list($left, $right) = $current->groups();

        $current->cleanup();

        if ($depth >= $this->maxDepth) {
            $current->attachLeft($this->terminate($left, $depth));
            $current->attachRight($this->terminate($right, $depth));
            return;
        }

        if ($left->numRows() > $this->minSamples) {
            $node = $this->findBestSplit($left, $depth);

            $current->attachLeft($node);

            $this->split($node, $depth + 1);
        } else {
            $current->attachLeft($this->terminate($left, $depth));
        }

        if ($right->numRows() > $this->minSamples) {
            $node = $this->findBestSplit($right, $depth);

            $current->attachRight($node);

            $this->split($node, $depth + 1);
        } else {
            $current->attachRight($this->terminate($right, $depth));
        }
    }

    /**
     * Search the tree for a terminal node.
     *
     * @param  array  $sample
     * @return \Rubix\ML\Graph\Nodes\Isolation|null
     */
    public function search(array $sample) : ?Isolation
    {
        $current = $this->root;

        while (isset($current)) {
            if ($current instanceof Isolation) {
                return $current;
            }

            if ($current instanceof Comparison) {
                if (is_string($current->value())) {
                    if ($sample[$current->index()] === $current->value()) {
                        $current = $current->left();
                    } else {
                        $current = $current->right();
                    }
                } else {
                    if ($sample[$current->index()] < $current->value()) {
                        $current = $current->left();
                    } else {
                        $current = $current->right();
                    }
                }
            }
        }

        return null;
    }

    /**
     * Randomized algorithm to find a split point in the data.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function findBestSplit(Dataset $dataset, int $depth) : Comparison
    {
        $index = rand(0, $dataset->numColumns() - 1);

        $sample = $dataset->row(rand(0, count($dataset) - 1));

        $value = $sample[$index];

        $groups = $dataset->partition($index, $value);

        $c = $this->calculateCFactor($dataset->numRows());

        $score = 2.0 ** -(($depth + $c) / $this->c);

        return new Comparison($index, $value, $score, $groups);
    }

    /**
     * Terminate the branch.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Isolation
     */
    protected function terminate(Dataset $dataset, int $depth) : Isolation
    {
        $c = $this->calculateCFactor($dataset->numRows());

        $score = 2.0 ** -(($depth + $c) / $this->c);

        return new Isolation($score);
    }

    /**
     * Calculate the average path length of an unsuccessful search for n nodes.
     *
     * @param  int  $n
     * @return float
     */
    protected function calculateCFactor(int $n) : float
    {
        if ($n <= 1) {
            return 0.0;
        }

        return 2.0 * (log($n - 1) + M_EULER) - (2.0 * ($n - 1) / $n);
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
