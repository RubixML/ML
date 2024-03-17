<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Encoding;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\HasBinaryChildren;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use IteratorAggregate;
use Traversable;

use function strlen;
use function substr;
use function array_pop;
use function is_string;
use function array_fill;
use function array_map;
use function array_sum;
use function hexdec;

/**
 * Decision Tree
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements IteratorAggregate<int,\Rubix\ML\Graph\Nodes\Decision>
 */
abstract class DecisionTree implements BinaryTree, IteratorAggregate
{
    /**
     * The maximum number of characters before a node label is truncated.
     *
     * @var int
     */
    protected const MAX_NODE_LABEL_LENGTH = 30;

    /**
     * The maximum depth of a branch before it is forced to terminate.
     *
     * @var int
     */
    protected int $maxHeight;

    /**
     * The maximum number of samples that a leaf node can contain.
     *
     * @var int
     */
    protected int $maxLeafSize;

    /**
     * The minimum increase in purity necessary for a node not to be post pruned.
     *
     * @var float
     */
    protected float $minPurityIncrease;

    /**
     * The root node of the tree.
     *
     * @var Split|null
     */
    protected ?Split $root = null;

    /**
     * The number of feature columns in the training set.
     *
     * @var int<0,max>
     */
    protected ?int $featureCount = null;

    /**
     * Return the brightness of a color in hex format.
     *
     * @param string $color
     * @return int
     */
    protected static function brightness(string $color) : int
    {
        $brightness = hexdec(substr($color, 0, 2));
        $brightness += hexdec(substr($color, 2, 2));
        $brightness += hexdec(substr($color, 4, 2));

        $brightness /= 3;

        return (int) round($brightness);
    }

    /**
     * @internal
     *
     * @param int $maxHeight
     * @param int $maxLeafSize
     * @param float $minPurityIncrease
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $maxHeight,
        int $maxLeafSize,
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

        $this->maxHeight = $maxHeight;
        $this->maxLeafSize = $maxLeafSize;
        $this->minPurityIncrease = $minPurityIncrease;
    }

    /**
     * Return the number of levels in the tree.
     *
     * @return int
     */
    public function height() : int
    {
        return $this->root ? $this->root->height() : 0;
    }

    /**
     * Return a factor that quantifies the skewness of the distribution of nodes in the tree.
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
     * @param Labeled $dataset
     * @throws \InvalidArgumentException
     */
    public function grow(Labeled $dataset) : void
    {
        $n = $dataset->numFeatures();

        $this->featureCount = $n;

        $this->root = $this->split($dataset);

        $stack = [[$this->root, 0]];

        while ($stack) {
            [$current, $depth] = array_pop($stack);

            [$left, $right] = $current->subsets();

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

            if ($left->numSamples() > $this->maxLeafSize) {
                $leftNode = $this->split($left);
            } else {
                $leftNode = $this->terminate($left);
            }

            if ($right->numSamples() > $this->maxLeafSize) {
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
    }

    /**
     * Search the decision tree for a leaf node and return it.
     *
     * @internal
     *
     * @param list<string|int|float> $sample
     * @return Outcome|null
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
                    if ($sample[$current->column()] <= $value) {
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
     * Return the importance scores of each feature column of the training set.
     *
     * @throws \RuntimeException
     * @return float[]
     */
    public function featureImportances() : array
    {
        if ($this->bare() or !$this->featureCount) {
            throw new RuntimeException('Tree has not been constructed.');
        }

        $importances = array_fill(0, $this->featureCount, 0.0);

        foreach ($this as $node) {
            if ($node instanceof Split) {
                $importances[$node->column()] += $node->purityIncrease();
            }
        }

        return $importances;
    }

    /**
     * Return an iterator for all the nodes in the tree starting at the root and traversing depth first.
     *
     * @return \Generator<\Rubix\ML\Graph\Nodes\BinaryNode>
     */
    public function getIterator() : Traversable
    {
        $stack = [$this->root];

        while ($current = array_pop($stack)) {
            yield $current;

            if ($current instanceof HasBinaryChildren) {
                foreach ($current->children() as $child) {
                    $stack[] = $child;
                }
            }
        }
    }

    /**
     * Print a representation of the decision tree suitable to render with the
     * graphviz tool. For example, writing it to graph.dot then executing:
     *
     * dot -Tpng graph.dot
     *
     * @param string[]|null $featureNames
     * @param int $maxDepth
     * @throws RuntimeException
     * @return Encoding
     */
    public function exportGraphviz(?array $featureNames = null, ?int $maxDepth = null) : Encoding
    {
        if (!$this->root) {
            throw new RuntimeException('Tree has not been constructed, train learner first.');
        }

        $carry = 'digraph Tree {' . PHP_EOL;
        $carry .= '  node [shape=box, fontname=helvetica];' . PHP_EOL;
        $carry .= '  edge [fontname=helvetica];' . PHP_EOL;

        $nodeCounter = 0;

        $this->_exportGraphviz($carry, $nodeCounter, $this->root, $maxDepth, $featureNames);

        $carry .= '}';

        return new Encoding($carry);
    }

    /**
     * Find a split point for a given subset of the training set.
     *
     * @param Labeled $dataset
     * @return Split
     */
    abstract protected function split(Labeled $dataset) : Split;

    /**
     * Terminate a branch with an outcome node.
     *
     * @param Labeled $dataset
     * @return Outcome
     */
    abstract protected function terminate(Labeled $dataset);

    /**
     * Calculate the impurity of a set of labels.
     *
     * @param list<string|int> $labels
     * @return float
     */
    abstract protected function impurity(array $labels) : float;

    /**
     * Calculate the impurity of a given split.
     *
     * @param array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled} $subsets
     * @return float
     */
    protected function splitImpurity(array $subsets) : float
    {
        $n = array_sum(array_map('count', $subsets));

        $impurity = 0.0;

        foreach ($subsets as $dataset) {
            $nHat = $dataset->numSamples();

            if ($nHat === 0) {
                continue;
            }

            $impurity += ($nHat / $n) * $this->impurity($dataset->labels());
        }

        return $impurity;
    }

    /**
     * Recursive function to print out the decision rule at each node using preorder traversal.
     *
     * @param string $carry
     * @param int $nodesCounter
     * @param BinaryNode $node
     * @param int $maxDepth
     * @param string[]|null $featureNames
     * @param int|null $parentId
     * @param int|null $leftRight
     * @param int $depth
     */
    protected function _exportGraphviz(
        string &$carry,
        int &$nodesCounter,
        BinaryNode $node,
        ?int $maxDepth = null,
        ?array $featureNames = null,
        ?int $parentId = null,
        ?int $leftRight = null,
        int $depth = 0
    ) : void {
        ++$depth;

        $thisNode = $nodesCounter++;

        if ($depth === $maxDepth) {
            $carry .= "  N$thisNode [label=\"...\"];" . PHP_EOL;
        } elseif ($node instanceof Split) {
            $column = $node->column();
            $value = $node->value();

            $carry .= "  N$thisNode [label=\"";

            if ($featureNames) {
                $name = $featureNames[$column];

                if (strlen($name) > self::MAX_NODE_LABEL_LENGTH) {
                    $name = substr($name, 0, self::MAX_NODE_LABEL_LENGTH) . '...';
                }

                $carry .= $name;
            } else {
                $carry .= "Feature {$column}";
            }

            $operator = is_string($value) ? '==' : '<=';

            $carry .= " $operator {$value}\"";

            $carry .= '];' . PHP_EOL;

            if ($node->left() !== null) {
                $this->_exportGraphviz($carry, $nodesCounter, $node->left(), $maxDepth, $featureNames, $thisNode, 1, $depth);
            }

            if ($node->right() !== null) {
                $this->_exportGraphviz($carry, $nodesCounter, $node->right(), $maxDepth, $featureNames, $thisNode, 2, $depth);
            }
        } elseif ($node instanceof Outcome) {
            $outcome = $node->outcome();
            $impurity = $node->impurity();

            $carry .= "  N$thisNode [label=\"{$outcome}";

            if ($impurity > 0.0) {
                $carry .= "\\nImpurity: {$impurity}";
            }

            $carry .= '"';

            if (is_string($outcome)) {
                $fillColor = substr(hash('crc32b', $outcome), -6);

                if (self::brightness($fillColor) > 128) {
                    $fontColor = '000000';
                } else {
                    $fontColor = 'ffffff';
                }
            } else {
                $fillColor = 'cccccc';
                $fontColor = '000000';
            }

            $carry .= ',style="rounded,filled"';
            $carry .= ",fontcolor=\"#{$fontColor}\"";
            $carry .= ",fillcolor=\"#{$fillColor}\"";

            $carry .= ']' . PHP_EOL;
        }

        if ($parentId !== null) {
            $carry .= "  N$parentId -> N$thisNode";

            if ($parentId === 0) {
                $carry .= ' [labeldistance=2.5';

                if ($leftRight === 1) {
                    $carry .= ',labelangle=45';
                    $carry .= ',headlabel="True"';
                } else {
                    $carry .= ',labelangle=-45';
                    $carry .= ',headlabel="False"';
                }

                $carry .= ']';
            }

            $carry .= ';' . PHP_EOL;
        }
    }
}
