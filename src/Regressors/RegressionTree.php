<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\CART;
use Rubix\ML\Graph\Nodes\Average;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Comparison;
use Rubix\ML\Other\Traits\LoggerAware;
use InvalidArgumentException;
use RuntimeException;

/**
 * Regression Tree
 *
 * A Leaf Tree learning algorithm that performs greedy splitting by
 * minimizing the variance (*impurity*) among Leaf node splits.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RegressionTree extends CART implements Learner, Verbose, Persistable
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
     * The memoized random column indices.
     *
     * @var array
     */
    protected $indices = [
        //
    ];

    /**
     * @param  int  $maxDepth
     * @param  int  $maxLeafSize
     * @param  int  $maxFeatures
     * @param  float  $tolerance
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, int $maxLeafSize = 3,
                                ?int $maxFeatures = null, float $tolerance = 1e-4)
    {
        parent::__construct($maxDepth, $maxLeafSize);

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
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::REGRESSOR;
    }

    /**
     * Train the regression tree by learning the optimal splits in the
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

        $this->indices = $dataset->axes();

        if ($this->logger) $this->logger->info('Training started');

        $this->grow($dataset);

        if ($this->logger) $this->logger->info('Training completed');

        $this->indices = [];
    }

    /**
     * Make a prediction based on the value of a terminal node in the tree.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->bare() === true) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset as $sample) {
            $node = $this->search($sample);

            $predictions[] = $node instanceof Average
                ? $node->outcome()
                : null;
        }

        return $predictions;
    }

    /**
     * Greedy algorithm to chose the best split for a given dataset as
     * determined by the variance of the split.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function findBestSplit(Labeled $dataset, int $depth) : Comparison
    {
        $bestVariance = INF;
        $bestIndex = $bestValue = null;
        $bestGroups = [];
        
        $maxFeatures = $this->maxFeatures
            ?? (int) round(sqrt($dataset->numColumns()));

        shuffle($this->indices);

        foreach (array_slice($this->indices, 0, $maxFeatures) as $index) {
            foreach ($dataset as $sample) {
                $value = $sample[$index];

                $groups = $dataset->partition($index, $value);

                $variance = $this->variance($groups);

                if ($variance < $bestVariance) {
                    $bestValue = $value;
                    $bestIndex = $index;
                    $bestGroups = $groups;
                    $bestVariance = $variance;
                }

                if ($variance < $this->tolerance) {
                    break 2;
                }
            }
        }

        if ($this->logger) $this->logger->info("Best split: column=$bestIndex"
            . " value=$bestValue impurity=$bestVariance depth=$depth");

        return new Comparison($bestValue, $bestIndex, $bestGroups, $bestVariance);
    }

    /**
     * Terminate the branch with the most likely outcome.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\BinaryNode
     */
    protected function terminate(Labeled $dataset, int $depth) : BinaryNode
    {
        list($mean, $variance) = Stats::meanVar($dataset->labels());

        if ($this->logger) $this->logger->info("Leaf node: outcome=$mean,"
            . " impurity=$variance depth=$depth");

        return new Average($mean, $variance, $dataset->numRows());
    }

    /**
     * Calculate the mean squared error for each group in a split.
     *
     * @param  array  $groups
     * @return float
     */
    protected function variance(array $groups) : float
    {
        $n = array_sum(array_map('count', $groups));

        $variance = 0.;

        foreach ($groups as $group) {
            $k = $group->numRows();

            if ($k < 2) {
                continue 1;
            }

            $density = $k / $n;

            $variance += $density * Stats::variance($group->labels());
        }

        return $variance;
    }
}
