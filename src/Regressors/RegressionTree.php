<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\CART;
use MathPHP\Statistics\Average;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\Comparison;
use MathPHP\Statistics\RandomVariable;
use InvalidArgumentException;

/**
 * Regression Tree
 *
 * A Decision Tree learning algorithm that performs greedy splitting by
 * minimizing the sum of squared errors between decision node splits.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RegressionTree extends CART implements Regressor, Persistable
{
    /**
     * The maximum number of features to consider when determining a split.
     *
     * @var int
     */
    protected $maxFeatures;

    /**
     * A small amount of impurity to tolerate when choosing a perfect split.
     *
     * @var float
     */
    protected $tolerance;

    /**
     * The randomized column indices.
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
    public function __construct(int $maxDepth = PHP_INT_MAX, int $maxLeafSize = 5,
                            int $maxFeatures = PHP_INT_MAX, float $tolerance = 1e-4)
    {
        if ($maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . ' feature to determine a split.');
        }

        if ($tolerance < 0) {
            throw new InvalidArgumentException('Impurity tolerance must be 0 or'
                . ' greater.');
        }

        $this->maxFeatures = $maxFeatures;
        $this->tolerance = $tolerance;

        parent::__construct($maxDepth, $maxLeafSize);
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
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        $this->indices = $dataset->indices();

        $this->grow($dataset);

        $this->indices = [];
    }

    /**
     * Make a prediction based on the value of a terminal node in the tree.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($dataset as $sample) {
            $predictions[] = $this->search($sample)->outcome();
        }

        return $predictions;
    }

    /**
     * Greedy algorithm to chose the best split point for a given set of data
     * as determined by its sum of squared error. The algorithm will terminate
     * early if it finds a perfect split. i.e. a sse score of 0.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function findBestSplit(Labeled $dataset) : Comparison
    {
        $best = [
            'ssd' => INF, 'index' => null, 'value' => null, 'groups' => [],
        ];

        shuffle($this->indices);

        foreach (array_slice($this->indices, 0, $this->maxFeatures) as $index) {
            foreach ($dataset as $sample) {
                $value = $sample[$index];

                $groups = $dataset->partition($index, $value);

                $ssd = 0.0;

                foreach ($groups as $group) {
                    if ($group->numRows() === 0) {
                        continue 1;
                    }

                    $ssd += RandomVariable::sumOfSquaresDeviations($group->labels());
                }

                if ($ssd < $best['ssd']) {
                    $best['ssd'] = $ssd;
                    $best['index'] = $index;
                    $best['value'] = $value;
                    $best['groups'] = $groups;
                }

                if ($ssd < $this->tolerance) {
                    break 2;
                }
            }
        }

        return new Comparison($best['index'], $best['value'], $best['groups'],
            $best['ssd']);
    }

    /**
     * Terminate the branch with the most likely outcome.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return \Rubix\ML\Graph\Nodes\Decision
     */
    protected function terminate(Labeled $dataset) : Decision
    {
        return new Decision(Average::mean($dataset->labels()));
    }
}
