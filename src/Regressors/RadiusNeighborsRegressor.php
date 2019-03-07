<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Graph\BallTree;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Radius Neighbors Regressor
 *
 * This is the regressor version of Radius Neighbors classifier implementing
 * a binary spatial tree under the hood for fast radius queries. The prediction
 * is a weighted average of each label from the training set that is within
 * a fixed user-defined radius.
 *
 * > **Note**: Unknown samples with 0 samples from the training set that are
 * within radius will be labeled *NaN*.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RadiusNeighborsRegressor extends BallTree implements Learner, Persistable
{
    public const OUTLIER = NAN;

    /**
     * The radius within which points are considered neighboors.
     *
     * @var float
     */
    protected $radius;

    /**
     * Should we use the inverse distances as confidence scores when
     * making predictions?
     *
     * @var bool
     */
    protected $weighted;

    /**
     * @param float $radius
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param bool $weighted
     * @param int $maxLeafSize
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $radius = 1.0,
        ?Distance $kernel = null,
        bool $weighted = true,
        int $maxLeafSize = 30
    ) {
        if ($radius <= 0.) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        $this->radius = $radius;
        $this->weighted = $weighted;

        parent::__construct($maxLeafSize, $kernel);
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::REGRESSOR;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return !$this->bare();
    }

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This estimator requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->grow($dataset);
    }

    /**
     * Make a prediction based on the nearest neighbors.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('Estimator has not been trainied.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset as $sample) {
            [$labels, $distances] = $this->range($sample, $this->radius);

            if (empty($labels)) {
                $predictions[] = self::OUTLIER;

                continue 1;
            }
            
            if ($this->weighted) {
                $weights = [];

                foreach ($distances as $distance) {
                    $weights[] = 1. / (1. + $distance);
                }

                $predictions[] = Stats::weightedMean($labels, $weights);
            } else {
                $predictions[] = Stats::mean($labels);
            }
        }

        return $predictions;
    }
}
