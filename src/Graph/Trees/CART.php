<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Graph\Nodes\BinaryNode;
use InvalidArgumentException;
use RuntimeException;
use Generator;

use const Rubix\ML\EPSILON;

/**
 * CART
 *
 * *Classification and Regression Tree* or CART is a binary search tree that
 * uses *decision* nodes at every split in the training data to locate a
 * purified leaf node.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
abstract class CART implements DecisionTree
{
    protected const DOWNSAMPLE_RATIO = 0.25;

    protected const IMPURITY_TOLERANCE = 1e-4;

    protected const MIN_PERCENTILES = 3;
    protected const MAX_PERCENTILES = 200;

    protected const BRANCH_INDENTER = '|---';

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
     * The maximum number of samples that a leaf node can contain.
     *
     * @var int
     */
    protected $maxLeafSize;

    /**
     * The maximum number of features to consider when determining a split.
     *
     * @var int|null
     */
    protected $maxFeatures;

    /**
     * Should we determine max features on the fly?
     *
     * @var bool
     */
    protected $fitMaxFeatures;

    /**
     * The minimum increase in purity necessary for a node not to be post pruned.
     *
     * @var float
     */
    protected $minPurityIncrease;

    /**
     * The number of feature columns in the training set.
     *
     * @var int
     */
    protected $featureCount;

    /**
     * The memoized random column indices.
     *
     * @var array
     */
    protected $columns = [
        //
    ];

    /**
     * @param int $maxDepth
     * @param int $maxLeafSize
     * @param int|null $maxFeatures
     * @param float $minPurityIncrease
     * @throws \InvalidArgumentException
     */
    public function __construct(int $maxDepth, int $maxLeafSize, ?int $maxFeatures, float $minPurityIncrease)
    {
        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have'
                . " depth of less than 1, $maxDepth given.");
        }

        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException('At least one sample'
                . ' is required to create a leaf node, '
                . " $maxLeafSize given.");
        }

        if ($minPurityIncrease < 0.) {
            throw new InvalidArgumentException('Min purity increase'
                . ' must be greater than or equal to 0,'
                . " $minPurityIncrease given.");
        }

        if (isset($maxFeatures) and $maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . " feature to determine a split, $maxFeatures given.");
        }

        $this->maxDepth = $maxDepth;
        $this->maxLeafSize = $maxLeafSize;
        $this->minPurityIncrease = $minPurityIncrease;
        $this->maxFeatures = $maxFeatures;
        $this->fitMaxFeatures = is_null($maxFeatures);
    }

    /**
     * Terminate a branch with a leaf node.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\Outcome
     */
    abstract protected function terminate(Labeled $dataset) : Outcome;

    /**
     * Compute the impurity of a labeled dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return float
     */
    abstract protected function impurity(Labeled $dataset) : float;

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

        $this->featureCount = $n = $dataset->numColumns();

        $this->columns = range(0, $n - 1);

        if ($this->fitMaxFeatures) {
            $this->maxFeatures = (int) round(sqrt($n));
        }

        $this->root = $this->split($dataset);

        $stack = [[$this->root, 1]];

        while ($stack) {
            [$current, $depth] = array_pop($stack) ?? [];

            [$left, $right] = $current->groups();

            $current->cleanup();

            ++$depth;

            if ($left->empty() or $right->empty()) {
                $node = $this->terminate($left->append($right));
    
                $current->attachLeft($node);
                $current->attachRight($node);

                continue 1;
            }
    
            if ($depth >= $this->maxDepth) {
                $current->attachLeft($this->terminate($left));
                $current->attachRight($this->terminate($right));
                
                continue 1;
            }

            if ($left->numRows() > $this->maxLeafSize) {
                $node = $this->split($left);

                if ($node->purityIncrease() >= $this->minPurityIncrease) {
                    $current->attachLeft($node);

                    $stack[] = [$node, $depth];
                } else {
                    $current->attachLeft($this->terminate($left));
                }
            } else {
                $current->attachLeft($this->terminate($left));
            }
    
            if ($right->numRows() > $this->maxLeafSize) {
                $node = $this->split($right);
    
                if ($node->purityIncrease() >= $this->minPurityIncrease) {
                    $current->attachRight($node);

                    $stack[] = [$node, $depth];
                } else {
                    $current->attachRight($this->terminate($right));
                }
            } else {
                $current->attachRight($this->terminate($right));
            }
        }

        $this->columns = [];
    }

    /**
     * Search the decision tree for a leaf node and return it.
     *
     * @param array $sample
     * @return \Rubix\ML\Graph\Nodes\Outcome|null
     */
    public function search(array $sample) : ?Outcome
    {
        $current = $this->root;

        while ($current) {
            if ($current instanceof Comparison) {
                $value = $current->value();

                if (is_string($value)) {
                    if ($sample[$current->column()] === $value) {
                        $current = $current->left();
                    } else {
                        $current = $current->right();
                    }
                } else {
                    if ($sample[$current->column()] < $value) {
                        $current = $current->left();
                    } else {
                        $current = $current->right();
                    }
                }

                continue 1;
            }

            if ($current instanceof Outcome) {
                return $current;
            }
        }

        return null;
    }

    /**
     * Return an array indexed by feature column that contains the normalized
     * importance score of that feature.
     *
     * @throws \RuntimeException
     * @return array
     */
    public function featureImportances() : array
    {
        if ($this->bare()) {
            throw new RuntimeException('Tree has not been constructed.');
        }

        $importances = array_fill(0, $this->featureCount, 0.);

        foreach ($this->dump() as $node) {
            if ($node instanceof Comparison) {
                $importances[$node->column()] += exp($node->purityIncrease());
            }
        }

        $total = array_sum($importances) ?: EPSILON;

        foreach ($importances as &$importance) {
            $importance /= $total;
        }

        return $importances;
    }

    /**
     * Print a human readable text representation of the decision tree.
     *
     * @throws RuntimeException
     */
    public function printRules() : void
    {
        if (!$this->root) {
            throw new RuntimeException('Tree has not been constructed.');
        }

        $this->_printrules($this->root);
    }

    /**
     * Recursive function to print out the decision rule at each node
     * using preorder traversal.
     *
     * @param \Rubix\ML\Graph\Nodes\BinaryNode $node
     * @param int $depth
     */
    protected function _printRules(BinaryNode $node, int $depth = 0) : void
    {
        ++$depth;

        $prefix = str_repeat(self::BRANCH_INDENTER, $depth) . ' ';

        if ($node instanceof Comparison) {
            if ($node->left() !== null) {
                $operator = is_string($node->value()) ? '==' : '<';

                echo $prefix . "Column_{$node->column()} $operator {$node->value()}" . PHP_EOL;

                $this->_printrules($node->left(), $depth);
            }
            
            if ($node->right() !== null) {
                $operator = is_string($node->value()) ? '!=' : '>=';

                echo $prefix . "Column_{$node->column()} $operator {$node->value()}" . PHP_EOL;

                $this->_printrules($node->right(), $depth);
            }
        }

        if ($node instanceof Outcome) {
            echo $prefix . "Outcome={$node->outcome()} Impurity={$node->impurity()}" . PHP_EOL;
        }
    }

    /**
     * Greedy algorithm to choose the best split point for a given dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function split(Labeled $dataset) : Comparison
    {
        $n = (int) ceil($dataset->numRows() * self::DOWNSAMPLE_RATIO);

        $k = max(self::MIN_PERCENTILES, min(self::MAX_PERCENTILES, $n));

        $p = range(0, 100, 100 / ($k - 1));

        shuffle($this->columns);

        $columns = array_slice($this->columns, 0, $this->maxFeatures);

        $bestImpurity = INF;
        $bestColumn = $bestValue = null;
        $bestGroups = [];

        foreach ($columns as $column) {
            $values = $dataset->column($column);

            if ($dataset->columnType($column) === DataType::CONTINUOUS) {
                $values = Stats::percentiles($values, $p);
            } else {
                $values = array_unique($values);
            }

            foreach ($values as $value) {
                $groups = $dataset->partition($column, $value);

                $impurity = $this->splitImpurity($groups);

                if ($impurity < $bestImpurity) {
                    $bestColumn = $column;
                    $bestValue = $value;
                    $bestGroups = $groups;
                    $bestImpurity = $impurity;
                }

                if ($impurity <= self::IMPURITY_TOLERANCE) {
                    break 2;
                }
            }
        }

        return new Comparison(
            $bestColumn,
            $bestValue,
            $bestGroups,
            $bestImpurity
        );
    }

    /**
     * Calculate the impurity of a given split.
     *
     * @param array $groups
     * @return float
     */
    protected function splitImpurity(array $groups) : float
    {
        $n = array_sum(array_map('count', $groups));

        $impurity = 0.;

        foreach ($groups as $dataset) {
            $m = $dataset->numRows();

            if ($m <= 1) {
                continue 1;
            }

            $impurity += ($m / $n) * $this->impurity($dataset);
        }

        return $impurity;
    }

    /**
     * Return a generator for all the nodes in the tree starting at the root.
     *
     * @return \Generator
     */
    protected function dump() : Generator
    {
        $stack = [$this->root];

        while ($stack) {
            yield $current = array_pop($stack);

            if ($current instanceof BinaryNode) {
                foreach ($current->children() as $child) {
                    $stack[] = $child;
                }
            }
        }
    }

    /**
     * Destroy the tree.
     */
    public function destroy() : void
    {
        unset($this->root);
    }
}
