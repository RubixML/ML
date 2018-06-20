<?php

namespace Rubix\ML\AnomalyDetection;

use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Graph\BinaryNode;
use Rubix\ML\Graph\BinaryTree;
use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

class IsolationTree extends BinaryTree implements Detector, Probabilistic, Persistable
{
    /**
     * The maximum depth of a branch before it is forced to terminate.
     *
     * @var int
     */
    protected $maxDepth;

    /**
     * The threshold isolation score. Score is a value between 0 and 1 where
     * 0.5 is nominal, 1 is certain to be an outlier, and 0 is an extremely
     * dense region.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The C factor represents the average length of the path of a search.
     *
     * @var float
     */
    protected $c;

    /**
     * The type of each feature column. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $columnTypes = [
        //
    ];

    /**
     * The number of times the tree has split. i.e. a comparison is made.
     *
     * @var int
     */
    protected $splits;

    /**
     * @param  int  $maxDepth
     * @param  float  $threshold
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, float $threshold = 0.5)
    {
        if ($maxDepth < 1) {
            throw new InvalidArgumentException('A tree cannot have depth less'
                . ' than 1.');
        }

        if ($threshold < 0 or $threshold > 1) {
            throw new InvalidArgumentException('Threshold isolation score must'
                . ' be between 0 and 1.');
        }

        $this->maxDepth = $maxDepth;
        $this->threshold = $threshold;
        $this->splits = 0;
    }

    /**
     * The complexity of the tree i.e. the number of splits.
     *
     * @return int
     */
    public function complexity() : int
    {
        return $this->splits;
    }

    /**
     * Train the isolation tree by randomly isolating individual data points.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        $this->columnTypes = $dataset->columnTypes();

        $this->c = $this->calculateCFactor($dataset->numRows());

        $this->setRoot($this->findRandomSplit($dataset->samples()));

        $this->splits = 1;

        $this->split($this->root);
    }

    /**
     * Make a prediction based on the score of a terminal node in the tree.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($dataset as $sample) {
            $predictions[] = $this->search($sample)->get('outcome');
        }

        return $predictions;
    }

    /**
     * Return the probabilities of a sample being an outllier.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $probabilities = [];

        foreach ($dataset as $sample) {
            $probabilities[] = $this->search($sample)->get('probability');
        }

        return $probabilities;
    }

    /**
     * Search the tree for a terminal node.
     *
     * @param  array  $sample
     * @return \Rubix\ML\Graph\BinaryNode
     */
    public function search(array $sample) : BinaryNode
    {
        return $this->_search($sample, $this->root);
    }

    /**
     * Recursive function to traverse the tree and return a terminal node.
     *
     * @param  array  $sample
     * @param  \Rubix\ML\BinaryNode  $root
     * @return \Rubix\ML\BinaryNode
     */
    protected function _search(array $sample, BinaryNode $root) : BinaryNode
    {
        if ($root->terminal) {
            return $root;
        }

        if ($root->type === self::CATEGORICAL) {
            if ($sample[$root->index] === $root->value) {
                return $this->_search($sample, $root->left());
            } else {
                return $this->_search($sample, $root->right());
            }
        } else {
            if ($sample[$root->index] < $root->value) {
                return $this->_search($sample, $root->left());
            } else {
                return $this->_search($sample, $root->right());
            }
        }
    }

    /**
     * Recursive function to split the training data adding decision nodes along the
     * way. The terminating conditions are a) split would make node responsible
     * for less values than $minSamples or b) the max depth of the branch has been reached.
     *
     * @param  \Rubix\ML\BinaryNode  $root
     * @param  int  $depth
     * @return void
     */
    protected function split(BinaryNode $root, int $depth = 0) : void
    {
        list($left, $right) = $root->groups;

        $root->remove('groups');

        if ($depth >= $this->maxDepth) {
            $root->attachLeft($this->terminate($left, $depth));
            $root->attachRight($this->terminate($right, $depth));
            return;
        }

        if (count($left) > 1) {
            $root->attachLeft($this->findRandomSplit($left));

            $this->splits++;

            $this->split($root->left(), ++$depth);
        } else {
            $root->attachLeft($this->terminate($left, $depth));
        }

        if (count($right) > 1) {
            $root->attachRight($this->findRandomSplit($right));

            $this->splits++;

            $this->split($root->right(), ++$depth);

        } else {
            $root->attachRight($this->terminate($left, $depth));
        }
    }

    /**
     * Randomized algorithm to find a split point in a dataset.
     *
     * @param  array  $data
     * @return \Rubix\ML\BinaryNode
     */
    protected function findRandomSplit(array $data) : BinaryNode
    {
        $index = array_rand($data[0]);

        $value = $data[array_rand($data)][$index];

        $groups = $this->partition($data, $index, $value);

        return new BinaryNode([
            'index' => $index,
            'value' => $value,
            'type' => $this->columnTypes[$index],
            'groups' => $groups,
        ]);
    }

    /**
     * Terminate the branch.
     *
     * @param  array  $data
     * @param  int  $depth
     * @return \Rubix\ML\Graph\BinaryNode
     */
    protected function terminate(array $data, int $depth) : BinaryNode
    {
        $n = count($data);

        $c = $this->calculateCFactor($n);

        $probability = 2.0 ** -(($depth + $c) / $this->c);

        $outcome = $probability > $this->threshold ? 1 : 0;

        return new BinaryNode([
            'outcome' => $outcome,
            'probability' => $probability,
            'terminal' => true,
        ]);
    }

    /**
     * Partition a dataset into left and right subsets.
     *
     * @param  array  $data
     * @param  int  $index
     * @param  mixed  $value
     * @return array
     */
    protected function partition(array $data, int $index, $value) : array
    {
        $left = $right = [];

        foreach ($data as $row) {
            if ($this->columnTypes[$index] === self::CATEGORICAL) {
                if ($row[$index] !== $value) {
                    $left[] = $row;
                } else {
                    $right[] = $row;
                }
            } else {
                if ($row[$index] < $value) {
                    $left[] = $row;
                } else {
                    $right[] = $row;
                }
            }
        }

        return [$left, $right];
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

        return 2.0 * (log($n - 1) + M_EULER)
            - (2.0 * ($n - 1) / ($n  + self::EPSILON));
    }
}
