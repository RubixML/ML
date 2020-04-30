<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Radius Neighbors Regressor
 *
 * This is the regressor version of Radius Neighbors implementing a binary spatial tree under
 * the hood for fast radius queries. The prediction is a weighted average of each label from
 * the training set that is within a fixed user-defined radius.
 *
 * > **Note**: Unknown samples with no training samples within radius are labeled
 * *NaN*. As such, Radius Neighbors is also a quasi anomaly detector.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RadiusNeighborsRegressor implements Estimator, Learner, Persistable
{
    use PredictsSingle;
    
    /**
     * The value to assign to outliers when making a prediction.
     *
     * @var float
     */
    public const OUTLIER = NAN;

    /**
     * The radius within which points are considered neighboors.
     *
     * @var float
     */
    protected $radius;

    /**
     * Should we consider the distances of our nearest neighbors when making predictions?
     *
     * @var bool
     */
    protected $weighted;

    /**
     * The spatial tree used to run range searches.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected $tree;

    /**
     * @param float $radius
     * @param bool $weighted
     * @param \Rubix\ML\Graph\Trees\Spatial|null $tree
     * @throws \InvalidArgumentException
     */
    public function __construct(float $radius = 1.0, bool $weighted = true, ?Spatial $tree = null)
    {
        if ($radius <= 0.0) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        $this->radius = $radius;
        $this->weighted = $weighted;
        $this->tree = $tree ?? new BallTree();
    }

    /**
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
    }

    /**
     * Return the data types that the model is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->tree->kernel()->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'radius' => $this->radius,
            'weighted' => $this->weighted,
            'tree' => $this->tree,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !$this->tree->bare();
    }

    /**
     * Return the base spatial tree instance.
     *
     * @return \Rubix\ML\Graph\Trees\Spatial
     */
    public function tree() : Spatial
    {
        return $this->tree;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' Labeled training set.');
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $this->tree->grow($dataset);
    }

    /**
     * Make a prediction based on the nearest neighbors.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return (int|float)[]
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->tree->bare()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            [$samples, $labels, $distances] = $this->tree->range($sample, $this->radius);

            if (empty($labels)) {
                $predictions[] = self::OUTLIER;

                continue 1;
            }
            
            if ($this->weighted) {
                $weights = [];

                foreach ($distances as $distance) {
                    $weights[] = 1.0 / (1.0 + $distance);
                }

                $predictions[] = Stats::weightedMean($labels, $weights);
            } else {
                $predictions[] = Stats::mean($labels);
            }
        }

        return $predictions;
    }
}
