<?php
namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Trees\ITree;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\Comparison;
use InvalidArgumentException;
use RuntimeException;

/**
 * Isolation Tree
 *
 * Isolation Trees separate anomalous samples from dense clusters using an
 * extremely randomized splitting process that isolates outliers into their own
 * nodes. Note that this Estimator is considered a weak learner and is typically
 * used within the context of an ensemble (such as Isolation Forest).
 *
 * References:
 * [1] Fei Tony Liu Et Al. (2008). Isolation Forest.
 * [2] Fei Tony Liu Et Al. (2011). Isolation-based Anomaly Detection.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class IsolationTree extends ITree implements Estimator, Probabilistic, Persistable
{
    /**
     * The amount of contamination (outliers) that is presumed to be in
     * the training set as a percentage.
     *
     * @var float
     */
    protected $contamination;

    /**
     * @param  int  $maxDepth
     * @param  int  $maxLeafSize
     * @param  float  $contamination
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(?int $maxDepth = null, int $maxLeafSize = 3, float $contamination = 0.1)
    {
        parent::__construct($maxDepth, $maxLeafSize);

        if ($contamination < 0. or $contamination > 0.5) {
            throw new InvalidArgumentException("The contamination factor must"
                . " be between 0 and 0.5, $contamination given.");
        }

        $this->contamination = $contamination;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::DETECTOR;
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
        if ($this->bare() === true) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($this->proba($dataset) as $probability) {
            $predictions[] = $probability > 0.5 ? 1 : 0;
        }

        return $predictions;
    }

    /**
     * Return the probabilities of a sample being an outllier.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if ($this->bare() === true) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $probabilities = [];

        foreach ($dataset as $sample) {
            $score = $this->search($sample)->score();
            
            $score -= $this->contamination;

            $probabilities[] = $score;
        }

        return $probabilities;
    }
}
