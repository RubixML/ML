<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use IteratorAggregate;
use Generator;

use function array_slice;
use function is_string;
use function is_int;
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
 *
 * @implements IteratorAggregate<int,\Rubix\ML\Graph\Nodes\Decision>
 */
abstract class CART implements IteratorAggregate
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
     * @var \Rubix\ML\Graph\Nodes\Split|null
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
    protected $types = [
        //
    ];

    /**
     * @internal
     *
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
     * @internal
     *
     * @return bool
     */
    public function bare() : bool
    {
        return !$this->root;
    }

    /**
     * Insert a root node and recursively split the dataset a terminating condition is met.
     *
     * @internal
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

                continue;
            }

            if ($depth >= $this->maxHeight) {
                $current->attachLeft($this->terminate($left));
                $current->attachRight($this->terminate($right));

                continue;
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
                if ($leftNode instanceof Split) {
                    $stack[] = [$leftNode, $depth];
                }

                if ($rightNode instanceof Split) {
                    $stack[] = [$rightNode, $depth];
                }
            } else {
                if ($leftNode instanceof Split) {
                    $current->attachLeft($this->terminate($left));
                }

                if ($rightNode instanceof Split) {
                    $current->attachRight($this->terminate($right));
                }
            }
        }

        unset($this->types);
    }

    /**
     * Search the decision tree for a leaf node and return it.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return \Rubix\ML\Graph\Nodes\Outcome|null
     */
    public function search(array $sample) : ?Outcome
    {
        $current = $this->root;

        while ($current) {
            if ($current instanceof Split) {
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

                continue;
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

        foreach ($this as $node) {
            if ($node instanceof Split) {
                $importances[$node->column()] += $node->purityIncrease();
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
     * Return a generator for all the nodes in the tree starting at the root and traversing depth first.
     *
     * @return \Generator<\Rubix\ML\Graph\Nodes\Decision>
     */
    public function getIterator() : Generator
    {
        $stack = [$this->root];

        while ($current = array_pop($stack)) {
            yield $current;

            foreach ($current->children() as $child) {
                if ($child instanceof Decision) {
                    $stack[] = $child;
                }
            }
        }
    }

    /**
     * Terminate a branch with an outcome node.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\Outcome
     */
    abstract protected function terminate(Labeled $dataset);

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
     * @return \Rubix\ML\Graph\Nodes\Split
     */
    protected function split(Labeled $dataset) : Split
    {
        $columns = array_keys($this->types ?? []);

        shuffle($columns);

        $columns = array_slice($columns, 0, $this->maxFeatures);

        $n = $dataset->numRows();

        $bestColumn = $bestValue = $bestGroups = null;
        $bestImpurity = INF;

        foreach ($columns as $column) {
            $values = $dataset->column($column);

            $type = $this->types[$column];

            if ($type->isContinuous()) {
                if (!isset($q)) {
                    $step = 1.0 / max(2.0, sqrt($n));

                    $q = array_slice(range(0.0, 1.0, $step), 1, -1);
                }

                $values = Stats::quantiles($values, $q);
            }

            $values = array_unique($values);

            foreach ($values as $value) {
                $groups = $dataset->splitByColumn($column, $value);

                $impurity = $this->splitImpurity($groups, $n);

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

        if (!is_int($bestColumn) or $bestValue === null or $bestGroups === null) {
            throw new RuntimeException('Could not split dataset.');
        }

        return new Split(
            $bestColumn,
            $bestValue,
            $bestGroups,
            $bestImpurity,
            $n
        );
    }

    /**
     * Calculate the impurity of a given split.
     *
     * @param array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled} $groups
     * @param int $n
     * @return float
     */
    protected function splitImpurity(array $groups, int $n) : float
    {
        $impurity = 0.0;

        foreach ($groups as $dataset) {
            $nHat = $dataset->numRows();

            if ($nHat <= 1) {
                continue;
            }

            $impurity += ($nHat / $n) * $this->impurity($dataset);
        }

        return $impurity;
    }

    /**
     * Recursive function to print out the decision rule at each node using preorder traversal.
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

        if ($node instanceof Split) {
            $left = $node->left();
            $right = $node->right();

            $identifier = $header ? $header[$node->column()] : "Feature {$node->column()}";

            if ($left) {
                $value = $node->value();

                $operator = is_string($value) ? '==' : '<';

                $carry .= "$prefix $identifier $operator $value" . PHP_EOL;

                $this->_rules($carry, $left, $header, $depth);
            }

            if ($right) {
                $value = $node->value();

                $operator = is_string($value) ? '!=' : '>=';

                $carry .= "$prefix $identifier $operator $value" . PHP_EOL;

                $this->_rules($carry, $right, $header, $depth);
            }
        }

        if ($node instanceof Outcome) {
            $carry .= $prefix . $node . PHP_EOL;
        }
    }
}
