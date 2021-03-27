<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\RanksFeatures;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function is_null;

/**
 * K Best Feature Selector
 *
 * A supervised feature selector that picks the top K ranked features.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KBestFeatureSelector implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

    /**
     * The maximum number of features to select from the dataset.
     *
     * @var int
     */
    protected $k;

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
     * @param int $k
     * @param \Rubix\ML\RanksFeatures|null $scorer
     * @throws \InvalidArgumentException
     */
    public function __construct(int $k, ?RanksFeatures $scorer = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be'
                . " greater than 0, $k given.");
        }

        $this->k = $k;
        $this->scorer = $scorer;
        $this->fitBase = is_null($scorer);
    }

    /**
     * Return the data types that this transformer is compatible with.
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

        if ($this->fitBase or is_null($this->scorer)) {
            switch ($dataset->labelType()) {
                case DataType::categorical():
                    $this->scorer = new ClassificationTree();

                    break;

                case DataType::continuous():
                    $this->scorer = new RegressionTree();

                    break;

                default:
                    throw new InvalidArgumentException('No compatible base'
                        . " learner for {$dataset->labelType()} label type.");
            }
        }

        $this->scorer->train($dataset);

        $importances = $this->scorer->featureImportances();

        asort($importances);

        $this->importances = array_slice($importances, 0, $this->k, true);
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
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
        return "K Best Selector (k: {$this->k}, scorer: {$this->scorer})";
    }
}
