<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\CART;
use Rubix\ML\Graph\Nodes\Best;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Other\Traits\LoggerAware;
use InvalidArgumentException;
use RuntimeException;

/**
 * Classification Tree
 *
 * A Leaf Tree-based classifier that minimizes gini impurity to greedily
 * search for the best splits in a training set.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ClassificationTree extends CART implements Learner, Probabilistic, Verbose, Persistable
{
    use LoggerAware;

    /**
     * The maximum number of features to consider when determining a split.
     *
     * @var int|null
     */
    protected $maxFeatures;

    /**
     * A small amount of impurity to tolerate when choosing a perfect split.
     *
     * @var float
     */
    protected $tolerance;

    /**
     * The possible class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * The memoized random column indices.
     *
     * @var array
     */
    protected $columns = [
        //
    ];

    /**
     * @param  int  $maxDepth
     * @param  int  $maxLeafSize
     * @param  int|null  $maxFeatures
     * @param  float  $minImpurityIncrease
     * @param  float  $tolerance
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, int $maxLeafSize = 3, ?int $maxFeatures = null,
                                float $minImpurityIncrease = 0., float $tolerance = 1e-3)
    {   
        if (isset($maxFeatures) and $maxFeatures < 1) {
            throw new InvalidArgumentException("Tree must consider at least 1"
                . " feature to determine a split, $maxFeatures given.");
        }

        if ($tolerance < 0.) {
            throw new InvalidArgumentException("Impurity tolerance must be 0"
                . " or greater, $tolerance given.");
        }

        $this->maxFeatures = $maxFeatures;
        $this->tolerance = $tolerance;

        parent::__construct($maxDepth, $maxLeafSize, $minImpurityIncrease);
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLASSIFIER;
    }

    /**
     * Train the Leaf tree by learning the most optimal splits in the
     * training set.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        $k = $dataset->numColumns();

        $this->classes = $dataset->possibleOutcomes();
        $this->columns = $dataset->axes();
        $this->maxFeatures = $this->maxFeatures ?? (int) round(sqrt($k));

        if ($this->logger) $this->logger->info('Learner initialized w/ params: '
            . Params::stringify([
                'max_depth' => $this->maxDepth,
                'max_leaf_size' => $this->maxLeafSize,
                'max_features' => $this->maxFeatures,
                'min_purity_increase' => $this->minPurityIncrease,
                'tolerance' => $this->tolerance,
            ]));

        $this->grow($dataset);

        if ($this->logger) $this->logger->info('Training completed');

        $this->columns = [];
    }

    /**
     * Make predictions from a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset as $sample) {
            $node = $this->search($sample);

            $predictions[] = $node instanceof Best
                ? $node->outcome()
                : null;
        }

        return $predictions;
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $probabilities = [];

        foreach ($dataset as $sample) {
            $node = $this->search($sample);

            $probabilities[] = $node instanceof Best
                ? $node->probabilities()
                : null;
        }

        return $probabilities;
    }

    /**
     * Greedy algorithm to choose the best split point for a given dataset.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function findBestSplit(Labeled $dataset, int $depth) : Comparison
    {
        $bestGini = INF;
        $bestColumn = $bestValue = null;
        $bestGroups = [];

        shuffle($this->columns);

        foreach (array_slice($this->columns, 0, $this->maxFeatures) as $column) {
            foreach ($dataset as $sample) {
                $value = $sample[$column];

                $groups = $dataset->partition($column, $value);

                $gini = $this->gini($groups);

                if ($gini < $bestGini) {
                    $bestColumn = $column;
                    $bestValue = $value;
                    $bestGroups = $groups;
                    $bestGini = $gini;
                }

                if ($gini < $this->tolerance) {
                    break 2;
                }
            }
        }

        if ($this->logger) $this->logger->info('Best split at '
            . Params::stringify([
                'column' => $bestColumn,
                'value' => $bestValue,
                'impurity' => $bestGini,
                'depth' => $depth,
            ]));

        return new Comparison($bestColumn, $bestValue, $bestGroups, $bestGini);
    }

    /**
     * Terminate the branch by selecting the class outcome with the highest
     * probability.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\BinaryNode
     */
    protected function terminate(Labeled $dataset, int $depth) : BinaryNode
    {
        $n = $dataset->numRows();

        $counts = array_count_values($dataset->labels());

        $outcome = Argmax::compute($counts);

        $probabilities = [];

        foreach ($counts as $class => $count) {
            $probabilities[$class] = $count / $n;
        }
    
        $gini = $this->gini([$dataset]);

        if ($this->logger) $this->logger->info('Branch terminated w/ '
            . Params::stringify([
                'outcome' => $outcome,
                'impurity' => $gini,
                'depth' => $depth,
            ]));

        return new Best($outcome, $probabilities, $gini, $n);
    }

    /**
     * Calculate the Gini impurity index for a given split.
     *
     * @param  array  $groups
     * @return float
     */
    protected function gini(array $groups) : float
    {
        $n = array_sum(array_map('count', $groups));

        $impurity = 0.;

        foreach ($groups as $group) {
            $k = $group->numRows();

            if ($k < 2) {
                continue 1;
            }

            $counts = array_count_values($group->labels());

            $gini = 0;

            foreach ($counts as $count) {
                $gini += 1 - ($count / $n) ** 2;
            }

            $impurity += ($k / $n) * $gini;
        }

        return $impurity;
    }
}
