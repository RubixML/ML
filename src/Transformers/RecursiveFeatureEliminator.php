<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\RanksFeatures;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Classifiers\ClassificationTree;
use InvalidArgumentException;
use RuntimeException;

use function count;
use function is_null;

/**
 * Recursive Feature Eliminator
 *
 * Recursive Feature Eliminator (RFE) is a supervised feature selector that uses the
 * importance scores returned by a learner implementing the RanksFeatures interface to
 * recursively drop feature columns with the lowest importance until a terminating
 * condition is met.
 *
 * References:
 * [1] I. Guyon et al. (2002). Gene Selection for Cancer Classification using Support Vector
 * Machines.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RecursiveFeatureEliminator implements Transformer, Stateful, Verbose
{
    use LoggerAware;

    /**
     * The minimum number of features to select from the dataset.
     *
     * @var int
     */
    protected $minFeatures;

    /**
     * The maximum number of features to drop from the dataset per iteration.
     *
     * @var int
     */
    protected $maxDroppedFeatures;

    /**
     * The maximum importance to drop from the dataset per iteration.
     *
     * @var float
     */
    protected $maxDroppedImportance;

    /**
     * The base feature ranking learner.
     *
     * @var \Rubix\ML\RanksFeatures|null
     */
    protected $estimator;

    /**
     * Should the base feature ranking learner be fitted?
     *
     * @var bool
     */
    protected $fitBase;

    /**
     * The final importances of the selected feature columns.
     *
     * @var float[]|null
     */
    protected $importances;

    /**
     * @param int $minFeatures
     * @param int $maxDroppedFeatures
     * @param float $maxDroppedImportance
     * @param \Rubix\ML\RanksFeatures|null $estimator
     */
    public function __construct(
        int $minFeatures,
        int $maxDroppedFeatures = 3,
        float $maxDroppedImportance = 0.2,
        ?RanksFeatures $estimator = null
    ) {
        if ($minFeatures < 1) {
            throw new InvalidArgumentException('Maximum features must'
                . " be greater than 0, $minFeatures given.");
        }

        if ($maxDroppedFeatures < 1) {
            throw new InvalidArgumentException('Maximum dropped features'
                . " must be greater than 0, $maxDroppedFeatures given.");
        }

        if ($maxDroppedImportance < 0.0 or $maxDroppedImportance > 1.0) {
            throw new InvalidArgumentException('Maximum dropped importance'
                . " must be between 0 and 1, $maxDroppedImportance given.");
        }

        $this->minFeatures = $minFeatures;
        $this->maxDroppedFeatures = $maxDroppedFeatures;
        $this->maxDroppedImportance = $maxDroppedImportance;
        $this->estimator = $estimator;
        $this->fitBase = is_null($estimator);
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->importances);
    }

    /**
     * Return the final importances of the selected feature columns.
     *
     * @return float[]|null
     */
    public function importances() : ?array
    {
        return $this->importances;
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Transformer requires a'
                . ' Labeled training set.');
        }

        if ($this->fitBase or is_null($this->estimator)) {
            switch ($dataset->labelType()) {
                case DataType::categorical():
                    $this->estimator = new ClassificationTree();

                    break 1;

                case DataType::continuous():
                    $this->estimator = new RegressionTree();

                    break 1;

                default:
                    throw new InvalidArgumentException('No compatible base'
                        . " learner for {$dataset->labelType()} label type.");
            }
        }

        if ($this->logger) {
            $this->logger->info("$this initialized");
        }

        $n = $dataset->numColumns();

        $subset = clone $dataset;

        $importances = array_fill(0, $n, 1.0 / $n);
        $selected = range(0, $n - 1);
        $epoch = 0;

        while (count($selected) > $this->minFeatures) {
            ++$epoch;

            $this->estimator->train($subset);

            $importances = $this->estimator->featureImportances();

            asort($importances);

            $dropped = [];
            $total = 0.0;

            foreach ($importances as $column => $importance) {
                $total += $importance;

                if ($total >= $this->maxDroppedImportance) {
                    break 1;
                }

                $dropped[] = (int) $column;

                unset($selected[$column], $importances[$column]);

                if (count($dropped) >= $this->maxDroppedFeatures) {
                    break 1;
                }

                if (count($selected) <= $this->minFeatures) {
                    break 1;
                }
            }

            $selected = array_values($selected);

            $subset->dropColumns($dropped);

            if ($this->logger) {
                $this->logger->info("Epoch $epoch - Dropped "
                    . count($dropped) . "/{$subset->numColumns()} columns,"
                    . " Total Dropped Importance: $total");
            }

            if (empty($dropped)) {
                break 1;
            }
        }

        $this->importances = array_combine($selected, array_values($importances)) ?: [];

        if ($this->logger) {
            $this->logger->info('Fitting complete');
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->importances)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $sample = array_values(array_intersect_key($sample, $this->importances));
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Recursive Feature Eliminator (min_features: {$this->minFeatures},"
            . " max_dropped_features: {$this->maxDroppedFeatures},"
            . " max_dropped_importance: {$this->maxDroppedImportance},"
            . " estimator: {$this->estimator})";
    }
}
