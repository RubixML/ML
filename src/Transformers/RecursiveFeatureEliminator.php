<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\RanksFeatures;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Classifiers\ClassificationTree;
use InvalidArgumentException;
use RuntimeException;

use function count;
use function is_null;

/**
 * Recursive Feature Eliminator
 *
 * Recursive Feature Eliminator or RFE is a supervised feature selector that uses the importance
 * scores returned by a learner implementing the RanksFeatures interface to recursively drop
 * feature columns with the lowest importance until max features is reached.
 *
 * References:
 * [1] I. Guyon et al. (2002). Gene Selection for Cancer Classification using Support Vector
 * Machines.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RecursiveFeatureEliminator implements Transformer, Stateful
{
    /**
     * The maximum number of features to select.
     *
     * @var int
     */
    protected $maxFeatures;

    /**
     * The maximum number of iterations to recurse upon the dataset.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The base feature ranking learner instance.
     *
     * @var \Rubix\ML\RanksFeatures|null
     */
    protected $base;

    /**
     * Should the base learner be fitted?
     *
     * @var bool
     */
    protected $fitBase;

    /**
     * The final feature importances of the selected columns.
     *
     * @var float[]|null
     */
    protected $importances;

    /**
     * @param int $maxFeatures
     * @param int $epochs
     * @param \Rubix\ML\RanksFeatures|null $base
     */
    public function __construct(int $maxFeatures, int $epochs = 10, ?RanksFeatures $base = null)
    {
        if ($maxFeatures < 1) {
            throw new InvalidArgumentException('Maximum features must'
                . " be greater than 0, $maxFeatures given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Number of epochs must'
                . " be greater than 0, $epochs given.");
        }

        $this->maxFeatures = $maxFeatures;
        $this->epochs = $epochs;
        $this->base = $base;
        $this->fitBase = is_null($base);
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
     * Return the final importances scores of the selected feature columns.
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

        if ($this->fitBase or is_null($this->base)) {
            switch ($dataset->labelType()) {
                case DataType::categorical():
                    $this->base = new ClassificationTree();

                    break 1;

                case DataType::continuous():
                    $this->base = new RegressionTree();

                    break 1;

                default:
                    throw new InvalidArgumentException('Label type is'
                        . ' not compatible with base learner.');
            }
        }

        $n = $dataset->numColumns();

        $selected = range(0, $n - 1);

        $k = (int) max(round(max($n - $this->maxFeatures, 0) / $this->epochs), 1);

        $subset = clone $dataset;

        do {
            $this->base->train($subset);

            $importances = $this->base->featureImportances();

            asort($importances);

            $dropped = array_slice($importances, 0, $k, true);

            $selected = array_values(array_diff_key($selected, $dropped));

            $subset->dropColumns(array_keys($dropped));
        } while (count($selected) > $this->maxFeatures);

        $importances = array_diff_key($importances, $dropped ?? []);

        $this->importances = array_combine($selected, $importances) ?: [];
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
        return "Recursive Feature Eliminator {max features: {$this->maxFeatures}}"
            . " {epochs: {$this->epochs} base: {$this->base}}";
    }
}
