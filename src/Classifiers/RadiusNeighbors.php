<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Graph\BallTree;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Radius Neighbors
 *
 * Radius Neighbors is a spatial tree-based classifier that takes the weighted vote
 * of each neighbor within a fixed user-defined radius measured by a kernel distance
 * function.
 *
 * > **Note**: Unknown samples that have 0 samples from the training set that are
 * within radius will be labeled as outliers (-1).
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RadiusNeighbors extends BallTree implements Learner, Probabilistic, Persistable
{
    public const OUTLIER = -1;

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
     * The unique class outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * @param float $radius
     * @param int $maxLeafSize
     * @param \Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @param bool $weighted
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $radius = 1.0,
        int $maxLeafSize = 20,
        ?Distance $kernel = null,
        bool $weighted = true
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
        return self::CLASSIFIER;
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
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Estimator requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->classes = $dataset->possibleOutcomes();

        $this->grow($dataset);
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('The learner has not'
                . ' been trained.');
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
                $weights = array_fill_keys($labels, 0.);

                foreach ($labels as $i => $label) {
                    $weights[$label] += 1. / (1. + $distances[$i]);
                }
            } else {
                $weights = array_count_values($labels);
            }

            $predictions[] = Argmax::compute($weights);
        }

        return $predictions;
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if ($this->bare()) {
            throw new RuntimeException('The learner has not'
                . ' been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $template = array_fill_keys($this->classes, 0.);

        $probabilities = [];

        foreach ($dataset as $sample) {
            [$labels, $distances] = $this->range($sample, $this->radius);

            if (empty($labels)) {
                $probabilities[] = null;

                continue 1;
            }

            if ($this->weighted) {
                $weights = array_fill_keys($labels, 0.);

                foreach ($labels as $i => $label) {
                    $weights[$label] += 1. / (1. + $distances[$i]);
                }
            } else {
                $weights = array_count_values($labels);
            }

            $total = array_sum($weights) ?: self::EPSILON;

            $dist = $template;

            foreach ($weights as $class => $weight) {
                $dist[$class] = $weight / $total;
            }

            $probabilities[] = $dist;
        }

        return $probabilities;
    }
}
