<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\DecisionTree;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\Comparison;
use InvalidArgumentException;

/**
 * Isolation Tree
 *
 * Isolation Trees separate anomalous samples from dense clusters using an
 * extremely randomized splitting process that isolates outliers into their own
 * nodes. Note that this Estimator is considered a weak learner and is typically
 * used within the context of an ensemble (such as Isolation Forest).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class IsolationTree extends DecisionTree implements Detector, Probabilistic, Persistable
{
    /**
     * The threshold isolation score betweeen 0 and 1 where 0 is not likely to
     * be an outlier and 1 is very likely to be an outlier.
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
     * @param  int  $maxDepth
     * @param  int  $minSamples
     * @param  float  $threshold
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $maxDepth = PHP_INT_MAX, int $minSamples = 5, float $threshold = 0.5)
    {
        if ($threshold < 0 or $threshold > 1) {
            throw new InvalidArgumentException('Threshold isolation score must'
                . ' be between 0 and 1.');
        }

        parent::__construct($maxDepth, $minSamples);

        $this->threshold = $threshold;
    }

    /**
     * Train the isolation tree by randomly isolating individual data points.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        $this->c = $this->calculateCFactor($dataset->numRows());

        $this->grow($dataset->samples());
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
            $predictions[] = $this->search($sample)->outcome();
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
            $probabilities[] = $this->search($sample)->meta('probability');
        }

        return $probabilities;
    }

    /**
     * Randomized algorithm to find a split point in the data.
     *
     * @param  array  $data
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function findBestSplit(array $data) : Comparison
    {
        $index = rand(0, count($data[0]) - 1);

        $value = $data[rand(0, count($data) - 1)][$index];

        $score = count($data);

        $groups = $this->partition($data, $index, $value);

        return new Comparison($index, $value, $score, $groups);
    }

    /**
     * Terminate the branch.
     *
     * @param  array  $data
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Decision
     */
    protected function terminate(array $data, int $depth) : Decision
    {
        $c = $this->calculateCFactor(count($data));

        $probability = 2.0 ** -(($depth + $c) / $this->c);

        $prediction = $probability > $this->threshold ? 1 : 0;

        return new Decision($prediction, [
            'probability' => $probability,
        ]);
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
}
