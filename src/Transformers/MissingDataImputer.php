<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Strategies\Continuous;
use Rubix\ML\Other\Strategies\BlurryMean;
use Rubix\ML\Other\Strategies\Categorical;
use Rubix\ML\Other\Strategies\PopularityContest;
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
class MissingDataImputer implements Transformer, Stateful
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
    public function __construct($placeholder = '?', Continuous $continuous = null, Categorical $categorical = null)
    {
        if (!is_numeric($placeholder) and !is_string($placeholder)) {
            throw new InvalidArgumentException('Placeholder must be a string or'
                . ' numeric type, ' . gettype($placeholder) . ' found.');
        }

        if (is_null($continuous)) {
            $continuous = new BlurryMean();
        }

        if (is_null($categorical)) {
            $categorical = new PopularityContest();
        }

        $this->placeholder = $placeholder;
        $this->continuous = $continuous;
        $this->categorical = $categorical;
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
            $values = $dataset->column($column);

            $values = array_filter($values, function ($value) {
                return $value !== $this->placeholder;
            });

            if ($type === DataFrame::CATEGORICAL) {
                $strategy = clone $this->categorical;
            } else {
                $strategy = clone $this->continuous;
            }

            $strategy->fit($values);

            $this->strategies[$column] = $strategy;
        }
    }

    /**
     * Transform the sample matrix.
     *
     * @param  array  $samples
     * @throws \RuntimeException
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
