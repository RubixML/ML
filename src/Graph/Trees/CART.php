<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Graph\Nodes\BinaryNode;
use InvalidArgumentException;
use RuntimeException;
use Generator;

use function array_slice;
use function is_string;
use function is_null;

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
abstract class CART
{
    /**
     * The glyph that denotes a branch of the tree.
     *
     * @var string
     */
    protected const BRANCH_INDENTER = '├───';

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
    protected $maxHeight;

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
     * The memorized column types of the dataset.
     *
     * @var \Rubix\ML\DataType[]
     */
    protected $types = [];

    /**
     * The memorized column offsets of the dataset.
     *
     * @var int[]
     */
    protected $columns = [];

    /**
     * @param int $maxHeight
     * @param int $maxLeafSize
     * @param int|null $maxFeatures
     * @param float $minPurityIncrease
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $maxHeight,
        int $maxLeafSize,
        ?int $maxFeatures,
        float $minPurityIncrease
    ) {
        if ($maxHeight < 1) {
            throw new InvalidArgumentException('Tree must have'
                . " depth greater than 0, $maxHeight given.");
        }

        if ($maxLeafSize < 1) {
            throw new InvalidArgumentException('At least one sample is'
                . " required to form a leaf node, $maxLeafSize given.");
        }

        if ($minPurityIncrease < 0.0) {
            throw new InvalidArgumentException('Min purity increase'
                . " must be greater than 0, $minPurityIncrease given.");
        }

        if (isset($maxFeatures) and $maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . " feature to determine a split, $maxFeatures given.");
        }

        $this->maxHeight = $maxHeight;
        $this->maxLeafSize = $maxLeafSize;
        $this->minPurityIncrease = $minPurityIncrease;
        $this->maxFeatures = $maxFeatures;
        $this->fitMaxFeatures = is_null($maxFeatures);
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
     * Insert a root node and recursively split the dataset a terminating
     * condition is met.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @throws \InvalidArgumentException
     */
    public function grow(Labeled $dataset) : void
    {
        $this->featureCount = $dataset->numColumns();

        if ($this->fitMaxFeatures) {
            $this->maxFeatures = (int) round(sqrt($this->featureCount));
        }

        $this->types = $dataset->columnTypes();

        $this->columns = array_keys($this->types);

        $this->root = $this->split($dataset);

        $stack = [[$this->root, 1]];

        while ([$current, $depth] = array_pop($stack)) {
            [$left, $right] = $current->groups();

            $current->cleanup();

            ++$depth;

            if ($left->empty() or $right->empty()) {
                $node = $this->terminate($left->merge($right));

                $current->attachLeft($node);
                $current->attachRight($node);

                continue 1;
            }

            if ($depth >= $this->maxHeight) {
                $current->attachLeft($this->terminate($left));
                $current->attachRight($this->terminate($right));

                continue 1;
            }

            if ($left->numRows() > $this->maxLeafSize) {
                $leftNode = $this->split($left);
            } else {
                $leftNode = $this->terminate($left);
            }

            if ($right->numRows() > $this->maxLeafSize) {
                $rightNode = $this->split($right);
            } else {
                $rightNode = $this->terminate($right);
            }

            $current->attachLeft($leftNode);
            $current->attachRight($rightNode);

            if ($current->purityIncrease() >= $this->minPurityIncrease) {
                if ($leftNode instanceof Comparison) {
                    $stack[] = [$leftNode, $depth];
                }

                if ($rightNode instanceof Comparison) {
                    $stack[] = [$rightNode, $depth];
                }
            } else {
                $node = $this->terminate($left->merge($right));

                $current->attachLeft($node);
                $current->attachRight($node);
            }
        }

        $this->types = $this->columns = [];
    }

    /**
     * Search the decision tree for a leaf node and return it.
     *
     * @param list<string|int|float> $sample
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
     * Return the normalized importance scores of each feature column of the training set.
     *
     * @throws \RuntimeException
     * @return float[]
     */
    public function featureImportances() : array
    {
        if ($this->bare()) {
            throw new RuntimeException('Tree has not been constructed.');
        }

        $importances = array_fill(0, $this->featureCount, 0.0);

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
     * @param string[] $header
     * @throws RuntimeException
     * @return string
     */
    public function rules(?array $header = null) : string
    {
        if (!$this->root) {
            throw new RuntimeException('Tree has not been constructed.');
        }

        if (isset($header) and count($header) !== $this->featureCount) {
            throw new InvalidArgumentException('Header must have the'
                . ' same number of columns as the training set, '
                . "{$this->featureCount} expected but "
                . count($header) . ' given.');
        }

        $carry = '';

        $this->_rules($carry, $this->root, $header);

        return $carry;
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
     * Greedy algorithm to choose the best split point for a given dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function split(Labeled $dataset) : Comparison
    {
        $m = $dataset->numRows();

        shuffle($this->columns);

        $columns = array_slice($this->columns, 0, $this->maxFeatures);

        $bestImpurity = INF;
        $bestColumn = 0;
        $bestValue = null;
        $bestGroups = [];

        foreach ($columns as $column) {
            $values = $dataset->column($column);

            if ($this->types[$column]->isContinuous()) {
                if (!isset($q)) {
                    $step = 1.0 / max(2.0, sqrt($m));

                    $q = array_slice(range(0.0, 1.0, $step), 1, -1);
                }

                $values = Stats::quantiles($values, $q);
            } else {
                $values = array_unique($values);
            }

            foreach ($values as $value) {
                $groups = $dataset->partitionByColumn($column, $value);

                $impurity = $this->splitImpurity($groups, $m);

                if ($impurity < $bestImpurity) {
                    $bestColumn = $column;
                    $bestValue = $value;
                    $bestGroups = $groups;
                    $bestImpurity = $impurity;
                }

                if ($impurity <= 0.0) {
                    break 2;
                }
            }
        }

        return new Comparison(
            $bestColumn,
            $bestValue,
            $bestGroups,
            $bestImpurity,
            $m
        );
    }

    /**
     * Calculate the impurity of a given split.
     *
     * @param \Rubix\ML\Datasets\Labeled[] $groups
     * @param int $n
     * @return float
     */
    protected function splitImpurity(array $groups, int $n) : float
    {
        $impurity = 0.0;

        foreach ($groups as $dataset) {
            $nHat = $dataset->numRows();

            if ($nHat <= 1) {
                continue 1;
            }

            $impurity += ($nHat / $n) * $this->impurity($dataset);
        }

        return $impurity;
    }

    /**
     * Recursive function to print out the decision rule at each node
     * using preorder traversal.
     *
     * @param string $carry
     * @param \Rubix\ML\Graph\Nodes\BinaryNode $node
     * @param string[]|null $header
     * @param int $depth
     */
    protected function _rules(string &$carry, BinaryNode $node, ?array $header = null, int $depth = 0) : void
    {
        ++$depth;

        $prefix = str_repeat(self::BRANCH_INDENTER, $depth) . ' ';

        if ($node instanceof Comparison) {
            $identifier = $header ? $header[$node->column()] : "Feature {$node->column()}";

            if ($node->left() !== null) {
                $operator = is_string($node->value()) ? '==' : '<';

                $carry .= $prefix . "$identifier $operator {$node->value()}" . PHP_EOL;

                $this->_rules($carry, $node->left(), $header, $depth);
            }

            if ($node->right() !== null) {
                $operator = is_string($node->value()) ? '!=' : '>=';

                $carry .= $prefix . "$identifier $operator {$node->value()}" . PHP_EOL;

                $this->_rules($carry, $node->right(), $header, $depth);
            }
        }

        if ($node instanceof Outcome) {
            $carry .= $prefix . $node . PHP_EOL;
        }
    }

    /**
     * Return a generator for all the nodes in the tree starting at the root.
     *
     * @return \Generator<\Rubix\ML\Graph\Nodes\Decision>
     */
    protected function dump() : Generator
    {
        $stack = [$this->root];

        while ($current = array_pop($stack)) {
            yield $current;

            foreach ($current->children() as $child) {
                $stack[] = $child;
            }
        }
    }
}
