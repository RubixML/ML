<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Strategies\Mean;
use Rubix\ML\Other\Strategies\Strategy;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Other\Strategies\KMostFrequent;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function is_null;

/**
 * Missing Data Imputer
 *
 * Missing Data Imputer replaces missing continuous (denoted by `NaN`) or categorical values
 * (denoted by special placeholder category) with a guess based on user-defined Strategy.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MissingDataImputer implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

    /**
     * The guessing strategy to use when imputing continuous values.
     *
     * @var \Rubix\ML\Other\Strategies\Strategy
     */
    protected $continuous;

    /**
     * The guessing strategy to use when imputing categorical values.
     *
     * @var \Rubix\ML\Other\Strategies\Strategy
     */
    protected $categorical;

    /**
     * The placeholder category that denotes missing values.
     *
     * @var string
     */
    protected $categoricalPlaceholder;

    /**
     * The fitted guessing strategy for each feature column.
     *
     * @var list<\Rubix\ML\Other\Strategies\Strategy>|null
     */
    protected $strategies;

    /**
     * The data types of the fitted feature columns.
     *
     * @var list<\Rubix\ML\DataType>|null
     */
    protected $types;

    /**
     * @param \Rubix\ML\Other\Strategies\Strategy|null $continuous
     * @param \Rubix\ML\Other\Strategies\Strategy|null $categorical
     * @param string $categoricalPlaceholder
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        ?Strategy $continuous = null,
        ?Strategy $categorical = null,
        string $categoricalPlaceholder = '?'
    ) {
        if ($continuous and !$continuous->type()->isContinuous()) {
            throw new InvalidArgumentException('Continuous strategy must'
                . ' be compatible with continuous data types.');
        }

        if ($categorical and !$categorical->type()->isCategorical()) {
            throw new InvalidArgumentException('Categorical strategy must'
                . ' be compatible with categorical data types.');
        }

        $this->continuous = $continuous ?? new Mean();
        $this->categorical = $categorical ?? new KMostFrequent(1);
        $this->categoricalPlaceholder = $categoricalPlaceholder;
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
        return isset($this->strategies) and isset($this->types);
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->strategies = $this->types = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            $donors = [];

            switch ($type->code()) {
                case DataType::CONTINUOUS:
                    $strategy = clone $this->continuous;

                    foreach ($dataset->column($column) as $value) {
                        if (is_float($value) and is_nan($value)) {
                            continue;
                        }

                        $donors[] = $value;
                    }

                    break;

                case DataType::CATEGORICAL:
                    $strategy = clone $this->categorical;

                    foreach ($dataset->column($column) as $value) {
                        if ($value !== $this->categoricalPlaceholder) {
                            $donors[] = $value;
                        }
                    }

                    break;
            }

            if (!isset($strategy)) {
                continue;
            }

            if (empty($donors)) {
                throw new InvalidArgumentException('Dataset must contain'
                    . ' at least 1 donor per feature column.');
            }

            $strategy->fit($donors);

            $this->strategies[$column] = $strategy;
            $this->types[$column] = $type;
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
        if (is_null($this->strategies) or is_null($this->types)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->types as $column => $type) {
                $value = &$sample[$column];

                switch ($type->code()) {
                    case DataType::CONTINUOUS:
                        if (is_float($value) and is_nan($value)) {
                            $value = $this->strategies[$column]->guess();
                        }

                        break;

                    case DataType::CATEGORICAL:
                        if ($value === $this->categoricalPlaceholder) {
                            $value = $this->strategies[$column]->guess();
                        }

                        break;
                }
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Missing Data Imputer (continuous_strategy: {$this->continuous},"
            . " categorical_strategy: {$this->categorical},"
            . " categorical_placeholder: {$this->categoricalPlaceholder})";
    }
}
