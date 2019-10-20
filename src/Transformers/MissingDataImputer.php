<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Strategies\Mean;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Strategies\Continuous;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Other\Strategies\KMostFrequent;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;

/**
 * Missing Data Imputer
 *
 * The Missing Data Imputer replaces missing values denoted by NaN for
 * continuous features or a placeholder variable for categorical ones
 * with a guess based on user-defined strategy.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MissingDataImputer implements Transformer, Stateful
{
    /**
     * The categorical placeholder variable denoting the category that
     * contains missing values.
     *
     * @var mixed
     */
    protected $placeholder;

    /**
     * The guessing strategy to use when imputing continuous values.
     *
     * @var \Rubix\ML\Other\Strategies\Continuous
     */
    protected $continuous;

    /**
     * The guessing strategy to use when imputing categorical values.
     *
     * @var \Rubix\ML\Other\Strategies\Categorical
     */
    protected $categorical;

    /**
     * The fitted guessing strategy for each feature column.
     *
     * @var array|null
     */
    protected $strategies;

    /**
     * @param string $placeholder
     * @param \Rubix\ML\Other\Strategies\Continuous|null $continuous
     * @param \Rubix\ML\Other\Strategies\Categorical|null $categorical
     * @throws \InvalidArgumentException
     */
    public function __construct(
        string $placeholder = '?',
        ?Continuous $continuous = null,
        ?Categorical $categorical = null
    ) {
        $this->placeholder = $placeholder;
        $this->continuous = $continuous ?? new Mean();
        $this->categorical = $categorical ?? new KMostFrequent();
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
            DataType::CATEGORICAL,
        ];
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
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        DatasetIsCompatibleWithTransformer::check($dataset, $this);
        
        $this->strategies = [];

        foreach ($dataset->types() as $column => $type) {
            $donors = [];

            foreach ($dataset->column($column) as $value) {
                switch (true) {
                    case is_float($value) and is_nan($value):
                        continue 2;

                    case $value === $this->placeholder:
                        continue 2;

                    default:
                        $donors[] = $value;
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
                        . ' given.');
            }

            $strategy->fit($donors);

            $this->strategies[$column] = $strategy;
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param array $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->strategies === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($sample as $column => &$value) {
                if ((is_float($value) and is_nan($value)) or $value === $this->placeholder) {
                    $value = $this->strategies[$column]->guess();
                }
            }
        }
    }
}
