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
 * Recursive Feature Eliminator or *RFE* is a supervised feature selector that uses the
 * importance scores returned by a learner implementing the RanksFeatures interface to
 * recursively drop feature columns with the lowest importance until the minimum number
 * of features has been reached.
 *
 * References:
 * [1] I. Guyon et al. (2002). Gene Selection for Cancer Classification using Support
 * Vector Machines.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RecursiveFeatureEliminator implements Transformer, Stateful, Verbose
{
    use LoggerAware;

    /**
     * The minimum number of features to select.
     *
     * @var int
     */
    protected $minFeatures;

    /**
     * The maximum number of features to drop from the dataset per iteration.
     *
     * @var int
     */
    protected $maxDropFeatures;

    /**
     * The maximum importance to drop from the dataset per iteration.
     *
     * @var float
     */
    protected $maxDropImportance;

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
     * @param int $maxDropFeatures
     * @param float $maxDropImportance
     * @param \Rubix\ML\RanksFeatures|null $estimator
     */
    public function __construct(
        int $minFeatures,
        int $maxDropFeatures = 3,
        float $maxDropImportance = 0.2,
        ?RanksFeatures $estimator = null
    ) {
        if ($minFeatures < 1) {
            throw new InvalidArgumentException('Maximum features must'
                . " be greater than 0, $minFeatures given.");
        }

        if ($maxDropFeatures < 1) {
            throw new InvalidArgumentException('Maximum dropped features'
                . " must be greater than 0, $maxDropFeatures given.");
        }

        if ($maxDropImportance < 0.0 or $maxDropImportance > 1.0) {
            throw new InvalidArgumentException('Maximum dropped importance'
                . " must be between 0 and 1, $maxDropImportance given.");
        }

        $this->minFeatures = $minFeatures;
        $this->maxDropFeatures = $maxDropFeatures;
        $this->maxDropImportance = $maxDropImportance;
        $this->estimator = $estimator;
        $this->fitBase = is_null($estimator);
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return \Rubix\ML\DataType[]
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

        $subset = clone $dataset;

        $selected = range(0, $dataset->numColumns() - 1);
        $epoch = 0;

        while (count($selected) > $this->minFeatures) {
            ++$epoch;

            $this->estimator->train($subset);

            $importances = $this->estimator->featureImportances();

            asort($importances);

            $dropped = [];
            $total = 0.0;

            foreach ($importances as $column => $importance) {
                if ($importance >= $this->maxDropImportance - $total) {
                    break 1;
                }

                $dropped[] = (int) $column;
                $total += $importance;

                unset($selected[$column]);

                if (count($dropped) >= $this->maxDropFeatures) {
                    break 1;
                }

                if (count($selected) <= $this->minFeatures) {
                    break 1;
                }
            }

            $subset->dropColumns($dropped);

            $selected = array_values($selected);

            if ($this->logger) {
                $this->logger->info("Epoch $epoch - Dropped "
                    . count($dropped) . "/{$subset->numColumns()} columns,"
                    . " Total Dropped Importance: $total");
            }

            if (empty($dropped)) {
                break 1;
            }
        }

        $this->estimator->train($subset);

        $importances = $this->estimator->featureImportances();

        $this->importances = array_combine($selected, $importances) ?: [];

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
            . " max_drop_features: {$this->maxDropFeatures},"
            . " max_drop_importance: {$this->maxDropImportance},"
            . " estimator: {$this->estimator})";
    }
}
