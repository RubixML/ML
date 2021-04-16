<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\RanksFeatures;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\array_unset;
use function count;
use function is_null;
use function array_slice;
use function array_values;
use function asort;

use const Rubix\ML\EPSILON;

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
class RecursiveFeatureEliminator implements Transformer, Stateful, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

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
     * The maximum importance to drop from the dataset per epoch.
     *
     * @var float
     */
    protected $maxDroppedImportance;

    /**
     * The base feature scorer.
     *
     * @var \Rubix\ML\RanksFeatures|null
     */
    protected $scorer;

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
     * Lambda function to drop feature columns from a sample given their offsets.
     *
     * @internal
     *
     * @param mixed[] $sample
     * @param int|string $index
     * @param mixed $offsets
     */
    public static function dropColumns(array &$sample, $index, $offsets) : void
    {
        array_unset($sample, $offsets);

        $sample = array_values($sample);
    }

    /**
     * @param int $minFeatures
     * @param int $maxDroppedFeatures
     * @param float $maxDroppedImportance
     * @param \Rubix\ML\RanksFeatures|null $scorer
     */
    public function __construct(
        int $minFeatures,
        int $maxDroppedFeatures = 5,
        float $maxDroppedImportance = 0.2,
        ?RanksFeatures $scorer = null
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
        $this->scorer = $scorer;
        $this->fitBase = is_null($scorer);
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
        return [
            DataType::categorical(),
            DataType::continuous(),
        ];
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
     * Return the normalized importances of the selected features indexed by column offset.
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
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Transformer requires a'
                . ' Labeled training set.');
        }

        if ($this->fitBase or $this->scorer === null) {
            switch ($dataset->labelType()) {
                case DataType::categorical():
                    $this->scorer = new ClassificationTree();

                    break;

                case DataType::continuous():
                    $this->scorer = new RegressionTree();

                    break;

                default:
                    throw new InvalidArgumentException('A compatible base'
                        . ' feature scorer cannot be determined.');
            }
        }

        if ($this->logger) {
            $this->logger->info("$this initialized");
        }

        $subset = clone $dataset;

        $selected = range(0, $subset->numColumns() - 1);
        $epoch = 1;

        do {
            $this->scorer->train($subset);

            $importances = $this->scorer->featureImportances();

            $total = array_sum($importances) ?: EPSILON;

            foreach ($importances as &$importance) {
                $importance /= $total;
            }

            asort($importances);

            $k = min($this->maxDroppedFeatures, count($selected) - $this->minFeatures);

            $candidates = array_slice($importances, 0, $k, true);

            $totalDroppedImportance = 0.0;
            $dropped = [];

            foreach ($candidates as $column => $importance) {
                $totalDroppedImportance += $importance;

                if ($totalDroppedImportance > $this->maxDroppedImportance) {
                    break;
                }

                unset($selected[$column], $importances[$column]);

                $dropped[] = $column;
            }

            $selected = array_values($selected);

            if ($this->logger) {
                $this->logger->info("Epoch $epoch - Dropped "
                    . count($dropped) . ' features with'
                    . " $totalDroppedImportance importance");
            }

            if (count($selected) <= $this->minFeatures) {
                break;
            }

            if (empty($dropped)) {
                break;
            }

            $subset->apply(new LambdaFunction([self::class, 'dropColumns'], $dropped));

            ++$epoch;
        } while (true);

        $this->importances = array_combine($selected, array_values($importances)) ?: [];

        if ($this->logger) {
            $this->logger->info('Fitting complete');
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->importances === null) {
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
        return "Recursive Feature Eliminator (min features: {$this->minFeatures},"
            . " max dropped features: {$this->maxDroppedFeatures},"
            . " max dropped importance: {$this->maxDroppedImportance},"
            . " scorer: {$this->scorer})";
    }
}
