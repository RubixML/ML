<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function Rubix\ML\argmax;

use const Rubix\ML\EPSILON;

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
class RadiusNeighbors implements Estimator, Learner, Probabilistic, Persistable
{
    public const OUTLIER = -1;

    /**
     * The radius within which points are considered neighboors.
     *
     * @var float
     */
    protected $radius;

    /**
     * The spatial tree used for range queries.
     *
     * @var \Rubix\ML\Graph\Trees\Spatial
     */
    protected $tree;

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
     * @param \Rubix\ML\Graph\Trees\Spatial|null $tree
     * @param bool $weighted
     * @throws \InvalidArgumentException
     */
    public function __construct(float $radius = 1.0, ?Spatial $tree = null, bool $weighted = true)
    {
        if ($radius <= 0.) {
            throw new InvalidArgumentException('Radius must be'
                . " greater than 0, $radius given.");
        }

        $this->radius = $radius;
        $this->tree = $tree ?? new BallTree();
        $this->weighted = $weighted;
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
            throw new InvalidArgumentException('Estimator requires a'
                . ' labeled training set.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $this->classes = $dataset->possibleOutcomes();

        $this->tree->grow($dataset);
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
        if ($this->tree->bare()) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset as $sample) {
            [$samples, $labels, $distances] = $this->tree->range($sample, $this->radius);

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

            $predictions[] = argmax($weights);
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
        if ($this->tree->bare()) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $template = array_fill_keys($this->classes, 0.);

        $probabilities = [];

        foreach ($dataset as $sample) {
            [$samples, $labels, $distances] = $this->tree->range($sample, $this->radius);

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

            $total = array_sum($weights) ?: EPSILON;

            $dist = $template;

            foreach ($weights as $class => $weight) {
                $dist[$class] = $weight / $total;
            }

            $probabilities[] = $dist;
        }

        return $probabilities;
    }
}
