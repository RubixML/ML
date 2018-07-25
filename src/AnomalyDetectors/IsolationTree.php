<?php
namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Trees\ITree;
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
class IsolationTree extends ITree implements Detector, Probabilistic, Persistable
{
    /**
     * The threshold isolation score betweeen 0 and 1 where 0 is not likely to
     * be an outlier and 1 is very likely to be an outlier.
     *
     * @var float
     */
    protected $threshold;

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

        $this->threshold = $threshold;

        parent::__construct($maxDepth, $minSamples);
    }

    /**
     * Train the isolation tree by randomly isolating individual data points.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        $this->grow($dataset);
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
            $predictions[] = $this->search($sample)->score()
                > $this->threshold ? 1 : 0;
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
            $probabilities[] = $this->search($sample)->score();
        }

        return $probabilities;
    }
}
