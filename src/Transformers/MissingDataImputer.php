<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Strategies\Mean;
use Rubix\ML\Other\Strategies\Continuous;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Other\Strategies\KMostFrequent;
use InvalidArgumentException;
use RuntimeException;

/**
 * Missing Data Imputer
 *
 * In the real world, it is common to have data with missing values here and
 * there. The Missing Data Imputer replaces missing value placeholders with a
 * guess based on a given imputer Strategy.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MissingDataImputer implements Stateful
{
    /**
     * The placeholder of a missing value.
     *
     * @var mixed
     */
    protected $placeholder;

    /**
     * The imputer to use when imputing continuous values.
     *
     * @var \Rubix\ML\Other\Strategies\Continuous
     */
    protected $continuous;

    /**
     * The imputer to use when imputing categorical values.
     *
     * @var \Rubix\ML\Other\Strategies\Categorical
     */
    protected $categorical;

    /**
     * The guessing strategy per feature column.
     *
     * @var array|null
     */
    protected $strategies;

    /**
     * @param  mixed  $placeholder
     * @param  \Rubix\ML\Other\Strategies\Continuous|null  $continuous
     * @param  \Rubix\ML\Other\Strategies\Categorical|null  $categorical
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($placeholder = '?', ?Continuous $continuous = null, ?Categorical $categorical = null)
    {
        if (!is_numeric($placeholder) and !is_string($placeholder)) {
            throw new InvalidArgumentException('Placeholder1D must be a string or'
                . ' numeric type, ' . gettype($placeholder) . ' found.');
        }

        if (is_null($continuous)) {
            $continuous = new Mean();
        }

        if (is_null($categorical)) {
            $categorical = new KMostFrequent();
        }

        $this->placeholder = $placeholder;
        $this->continuous = $continuous;
        $this->categorical = $categorical;
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->strategies);
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->strategies = [];

        foreach ($dataset->types() as $column => $type) {
            $values = [];

            foreach ($dataset->column($column) as $value) {
                if ($value !== $this->placeholder) {
                    $values[] = $value;
                }
            }

            switch ($type) {
                case DataType::CATEGORICAL:
                    $strategy = clone $this->categorical;
                    break 1;

                case DataType::CONTINUOUS:
                    $strategy = clone $this->continuous;
                    break 1;

                default:
                    throw new InvalidArgumentException('This transformer'
                        . ' only handles categorical and continuous'
                        . ' features, ' . DataType::TYPES[$type]
                        . ' found.');
            }

            $strategy->fit($values);

            $this->strategies[$column] = $strategy;
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->strategies)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as $row => &$sample) {
            foreach ($sample as $column => &$feature) {
                if ($feature === $this->placeholder) {
                    $strategy = $this->strategies[$column];

                    $feature = $strategy->guess();
                }
            }
        }
    }
}
